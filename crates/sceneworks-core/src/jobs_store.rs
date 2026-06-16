use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use parking_lot::Mutex;
use rusqlite::{
    params, params_from_iter, Connection, OptionalExtension, Row, ToSql, TransactionBehavior,
};
use serde::de::DeserializeOwned;
use serde_json::{json, Map, Number, Value};

use crate::contracts::{
    ContractNumber, JobSnapshot, JobStatus, JobType, ProgressStage, QueueSummary, WorkerCapability,
    WorkerSnapshot, WorkerStatus, WorkerUtilizationSnapshot,
};
use crate::store_util::{ensure_column, parse_string_enum};
use crate::time::{format_unix_seconds, now_unix_seconds, utc_now};

pub const ACTIVE_STATUSES: &[&str] = &[
    "preparing",
    "downloading",
    "loading_model",
    "running",
    "saving",
];
pub const TERMINAL_STATUSES: &[&str] = &["completed", "failed", "canceled", "interrupted"];
pub const JOB_STATUSES: &[&str] = &[
    "queued",
    "preparing",
    "downloading",
    "loading_model",
    "running",
    "saving",
    "completed",
    "failed",
    "canceled",
    "interrupted",
];
pub const NON_GPU_JOB_TYPES: &[&str] = &[
    "model_download",
    "model_import",
    "model_convert",
    "lora_import",
    "lora_download",
];
pub const MAX_JOB_ATTEMPTS: u32 = 5;

/// The non-GPU job types as a quoted SQL list for `type in (...)` / `type not in
/// (...)` dispatch filters, derived once from [`NON_GPU_JOB_TYPES`]. This keeps
/// the SQL from drifting away from the declared contract — the drift this fixes
/// was `model_convert` living in the const but missing from the hard-coded SQL
/// lists (sc-1629). Values are crate constants, never user input, so direct
/// interpolation is safe.
fn non_gpu_job_types_sql() -> &'static str {
    static SQL: OnceLock<String> = OnceLock::new();
    SQL.get_or_init(|| {
        NON_GPU_JOB_TYPES
            .iter()
            .map(|job_type| format!("'{job_type}'"))
            .collect::<Vec<_>>()
            .join(", ")
    })
}

/// The active (non-terminal, non-queued) statuses as a quoted SQL list for
/// `status in (...)` stale-sweep / claim-guard filters, derived once from
/// [`ACTIVE_STATUSES`] — same anti-drift rationale as [`non_gpu_job_types_sql`]
/// (sc-4207 / F-CORE-3): the list was copy-pasted into five SQL statements, so
/// adding/renaming an active status risked missing one. Values are crate
/// constants, never user input, so direct interpolation is safe.
fn active_statuses_sql() -> &'static str {
    static SQL: OnceLock<String> = OnceLock::new();
    SQL.get_or_init(|| {
        ACTIVE_STATUSES
            .iter()
            .map(|status| format!("'{status}'"))
            .collect::<Vec<_>>()
            .join(", ")
    })
}

/// The terminal statuses as a quoted SQL list for `status not in (...)` filters,
/// derived once from [`TERMINAL_STATUSES`] — same anti-drift rationale as
/// [`active_statuses_sql`]. Used to select the non-terminal (still in-flight,
/// including `queued`) jobs for the queue summary. Values are crate constants,
/// never user input, so direct interpolation is safe.
fn terminal_statuses_sql() -> &'static str {
    static SQL: OnceLock<String> = OnceLock::new();
    SQL.get_or_init(|| {
        TERMINAL_STATUSES
            .iter()
            .map(|status| format!("'{status}'"))
            .collect::<Vec<_>>()
            .join(", ")
    })
}
const DISPATCH_MEMORY_NOT_WORSE_TOLERANCE_MB: f64 = 512.0;
const DISPATCH_MEMORY_RELIEF_THRESHOLD_MB: f64 = 1024.0;
const DISPATCH_LOW_MEMORY_THRESHOLD_MB: f64 = 2048.0;
const DISPATCH_HEALTHY_MEMORY_THRESHOLD_MB: f64 = 4096.0;
const DISPATCH_LOAD_NOT_WORSE_TOLERANCE_PERCENT: f64 = 10.0;
const DISPATCH_LOAD_RELIEF_THRESHOLD_PERCENT: f64 = 15.0;
const DISPATCH_HIGH_LOAD_THRESHOLD_PERCENT: f64 = 85.0;
const DISPATCH_RECOVERED_LOAD_THRESHOLD_PERCENT: f64 = 75.0;
const DISPATCH_MEMORY_USAGE_NOT_WORSE_TOLERANCE_PERCENT: f64 = 10.0;
const DISPATCH_MEMORY_USAGE_RELIEF_THRESHOLD_PERCENT: f64 = 10.0;
const DISPATCH_HIGH_MEMORY_USAGE_THRESHOLD_PERCENT: f64 = 90.0;
const DISPATCH_RECOVERED_MEMORY_USAGE_THRESHOLD_PERCENT: f64 = 80.0;

pub type JobsStoreResult<T> = Result<T, JobsStoreError>;

#[derive(Debug)]
pub enum JobsStoreError {
    Io(std::io::Error),
    Sqlite(rusqlite::Error),
    Json(serde_json::Error),
    NotFound(String),
    InvalidStatus(String),
    InvalidNumber(String),
    InvalidRequestedGpu(String),
    RetryLimit {
        max_attempts: u32,
    },
    /// A progress report tried to change a job that already reached a terminal
    /// status. Terminal jobs are immutable; only an idempotent re-report of the
    /// same terminal status succeeds (sc-4172).
    TerminalJobImmutable {
        job_id: String,
        status: String,
    },
    /// A progress report came from a worker that no longer owns the job — the
    /// job was swept/canceled (worker_id cleared) or reclaimed. The worker
    /// should abandon the job (sc-4172).
    NotJobOwner {
        job_id: String,
    },
}

impl std::fmt::Display for JobsStoreError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Sqlite(error) => write!(formatter, "{error}"),
            Self::Json(error) => write!(formatter, "{error}"),
            Self::NotFound(id) => write!(formatter, "Record not found: {id}"),
            Self::InvalidStatus(status) => write!(formatter, "Unsupported job status: {status}"),
            Self::InvalidNumber(field) => write!(formatter, "Invalid numeric value for {field}"),
            Self::InvalidRequestedGpu(detail) => write!(formatter, "{detail}"),
            Self::RetryLimit { max_attempts } => {
                write!(
                    formatter,
                    "Job retry limit reached after {max_attempts} attempts."
                )
            }
            Self::TerminalJobImmutable { job_id, status } => {
                write!(
                    formatter,
                    "Job {job_id} is already {status}; terminal jobs cannot be updated."
                )
            }
            Self::NotJobOwner { job_id } => {
                write!(
                    formatter,
                    "Progress rejected: the reporting worker no longer owns job {job_id}."
                )
            }
        }
    }
}

impl std::error::Error for JobsStoreError {}

impl From<std::io::Error> for JobsStoreError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<rusqlite::Error> for JobsStoreError {
    fn from(error: rusqlite::Error) -> Self {
        Self::Sqlite(error)
    }
}

impl From<serde_json::Error> for JobsStoreError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

#[derive(Debug)]
pub struct JobsStore {
    db_path: PathBuf,
    lock: Mutex<()>,
}

#[derive(Debug, Clone)]
pub struct CreateJob {
    pub job_type: JobType,
    pub project_id: Option<String>,
    pub project_name: Option<String>,
    pub payload: Map<String, Value>,
    pub requested_gpu: String,
    pub source_job_id: Option<String>,
    pub duplicate_of_job_id: Option<String>,
    pub attempts: u32,
}

#[derive(Debug, Clone)]
pub struct DuplicateJob {
    pub payload_changes: Map<String, Value>,
    pub requested_gpu: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RetryJob {
    pub payload_changes: Map<String, Value>,
}

#[derive(Debug, Clone)]
pub struct RegisterWorker {
    pub worker_id: String,
    pub gpu_id: String,
    pub gpu_name: Option<String>,
    pub capabilities: Vec<WorkerCapability>,
    pub loaded_models: Vec<String>,
    pub utilization: Option<WorkerUtilizationSnapshot>,
}

#[derive(Debug, Clone)]
pub struct WorkerHeartbeat {
    pub worker_id: String,
    pub status: WorkerStatus,
    pub current_job_id: Option<String>,
    pub loaded_models: Vec<String>,
    pub utilization: Option<WorkerUtilizationSnapshot>,
}

#[derive(Debug, Clone)]
pub struct ProgressUpdate {
    pub status: JobStatus,
    pub stage: ProgressStage,
    pub progress: f64,
    pub message: String,
    pub error: Option<String>,
    pub result: Option<Map<String, Value>>,
    pub eta_seconds: Option<f64>,
    /// Sampled GPU memory percentage observed by the worker at this progress
    /// point (0..100). The store keeps a running max across a job's progress
    /// updates (sc-2086) so completed-row meters render the peak.
    pub peak_gpu_memory_pct: Option<f64>,
    /// Sampled GPU load percentage observed at this progress point (0..100).
    /// Same running-max semantics as peak_gpu_memory_pct.
    pub peak_gpu_load_pct: Option<f64>,
    /// Runtime backend label the worker reports for this job
    /// ("mlx" / "mps" / "cuda" / "cpu"). First non-null value sticks — once a
    /// worker tells us which backend ran the job, subsequent status-only
    /// progress updates can't accidentally clear it. Drives the
    /// WorkerProgressCard arch pill.
    pub backend: Option<String>,
    /// Id of the worker reporting this progress. When set, the store rejects
    /// the update unless the job's `worker_id` still matches — a zombie worker
    /// whose job was swept to `interrupted` (worker_id cleared) or reclaimed by
    /// another worker can no longer resurrect or corrupt it (sc-4172). `None`
    /// keeps legacy trusted-caller behavior.
    pub worker_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StaleSweep {
    pub workers: Vec<WorkerSnapshot>,
    pub jobs: Vec<JobSnapshot>,
}

impl JobsStore {
    pub fn new(db_path: impl Into<PathBuf>) -> Self {
        Self {
            db_path: db_path.into(),
            lock: Mutex::new(()),
        }
    }

    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    pub fn initialize(&self) -> JobsStoreResult<()> {
        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        transaction.execute_batch(
            "
            create table if not exists jobs (
              id text primary key,
              type text not null,
              status text not null,
              project_id text,
              project_name text,
              payload_json text not null,
              result_json text not null default '{}',
              requested_gpu text not null default 'auto',
              assigned_gpu text,
              worker_id text,
              progress real not null default 0,
              stage text not null default 'queued',
              message text not null default '',
              error text,
              eta_seconds real,
              attempts integer not null default 1,
              source_job_id text,
              duplicate_of_job_id text,
              cancel_requested integer not null default 0,
              created_at text not null,
              updated_at text not null,
              started_at text,
              completed_at text,
              canceled_at text,
              last_heartbeat_at text
            );

            create index if not exists idx_jobs_status_created
              on jobs(status, created_at);
            create index if not exists idx_jobs_project_created
              on jobs(project_id, created_at);
            create index if not exists idx_jobs_assigned_gpu_status
              on jobs(assigned_gpu, status);

            create table if not exists workers (
              id text primary key,
              gpu_id text not null,
              gpu_name text,
              status text not null,
              current_job_id text,
              capabilities_json text not null,
              loaded_models_json text not null,
              utilization_json text,
              registered_at text not null,
              last_seen_at text not null
            );
            ",
        )?;
        ensure_column(&transaction, "workers", "utilization_json", "text")?;
        // sc-2086: per-job peak GPU memory % and load %, written by the worker
        // along with progress so a completed row shows the peak the run reached.
        ensure_column(&transaction, "jobs", "peak_gpu_memory_pct", "real")?;
        ensure_column(&transaction, "jobs", "peak_gpu_load_pct", "real")?;
        // Runtime backend label written by the worker ("mlx" / "mps" / "cuda"
        // / "cpu"). First-non-null wins so the WorkerProgressCard's arch pill
        // stays stable across the run.
        ensure_column(&transaction, "jobs", "backend", "text")?;
        transaction.commit()?;
        Ok(())
    }

    pub fn mark_interrupted_on_startup(&self) -> JobsStoreResult<Vec<JobSnapshot>> {
        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let interrupted = self.list_jobs_by_status_on_connection(&transaction, ACTIVE_STATUSES)?;
        let interrupted_ids = interrupted
            .iter()
            .map(|job| job.id.clone())
            .collect::<Vec<_>>();
        let now = utc_now();
        transaction.execute(
            &format!(
                "
            update jobs
               set status = 'interrupted',
                   stage = 'interrupted',
                   message = 'Job was interrupted by a backend restart.',
                   error = 'The backend restarted before this job finished.',
                   completed_at = ?1,
                   updated_at = ?1,
                   worker_id = null
             where status in ({active})
            ",
                active = active_statuses_sql()
            ),
            params![now],
        )?;
        transaction.execute(
            "update workers set status = 'offline', current_job_id = null where status != 'offline'",
            [],
        )?;
        let updated_jobs = interrupted_ids
            .iter()
            .map(|job_id| self.get_job_on_connection(&transaction, job_id))
            .collect::<JobsStoreResult<Vec<_>>>()?;
        transaction.commit()?;
        Ok(updated_jobs)
    }

    pub fn create_job(&self, request: CreateJob) -> JobsStoreResult<JobSnapshot> {
        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let job = self.create_job_on_connection(&transaction, request, None)?;
        transaction.commit()?;
        Ok(job)
    }

    /// Create a job under a caller-supplied id. Used when the payload must
    /// reference its own job id before insertion — e.g. a `lora_train` job whose
    /// resolved [`crate::training::TrainingPlan`] embeds `jobId`/`sourceJobId`.
    /// The id must be unique; a collision surfaces as a SQLite error.
    pub fn create_job_with_id(
        &self,
        id: String,
        request: CreateJob,
    ) -> JobsStoreResult<JobSnapshot> {
        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let job = self.create_job_on_connection(&transaction, request, Some(id))?;
        transaction.commit()?;
        Ok(job)
    }

    pub fn list_jobs(
        &self,
        project_id: Option<&str>,
        status: Option<&str>,
        limit: u32,
    ) -> JobsStoreResult<Vec<JobSnapshot>> {
        let _guard = self.lock.lock();
        let connection = self.connect()?;
        let limit = limit.clamp(1, 500);
        let mut conditions: Vec<&str> = Vec::new();
        let mut bindings: Vec<Box<dyn ToSql>> = Vec::new();
        if let Some(project_id) = project_id {
            conditions.push("project_id = ?");
            bindings.push(Box::new(project_id.to_owned()));
        }
        if let Some(status) = status {
            conditions.push("status = ?");
            bindings.push(Box::new(status.to_owned()));
        }
        let mut sql = String::from("select * from jobs");
        if !conditions.is_empty() {
            sql.push_str(" where ");
            sql.push_str(&conditions.join(" and "));
        }
        sql.push_str(" order by created_at desc limit ?");
        bindings.push(Box::new(limit));
        let mut statement = connection.prepare(&sql)?;
        let jobs =
            collect_jobs(statement.query_map(params_from_iter(bindings.iter()), row_to_job)?)?;
        Ok(jobs)
    }

    pub fn get_job(&self, job_id: &str) -> JobsStoreResult<JobSnapshot> {
        let _guard = self.lock.lock();
        let connection = self.connect()?;
        self.get_job_on_connection(&connection, job_id)
    }

    pub fn cancel_job(&self, job_id: &str) -> JobsStoreResult<JobSnapshot> {
        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let job = self.get_job_on_connection(&transaction, job_id)?;
        if is_terminal_status(job.status.as_str()) {
            return Ok(job);
        }

        let now = utc_now();
        if job.status == JobStatus::Queued {
            transaction.execute(
                "
                update jobs
                   set status = 'canceled',
                       stage = 'canceled',
                       progress = 1,
                       cancel_requested = 1,
                       message = 'Canceled before a worker started.',
                       canceled_at = ?1,
                       completed_at = ?1,
                       updated_at = ?1
                 where id = ?2
                ",
                params![now, job_id],
            )?;
        } else {
            transaction.execute(
                "
                update jobs
                   set cancel_requested = 1,
                       message = 'Cancellation requested. Waiting for worker acknowledgement.',
                       updated_at = ?1
                 where id = ?2
                ",
                params![now, job_id],
            )?;
        }
        let job = self.get_job_on_connection(&transaction, job_id)?;
        transaction.commit()?;
        Ok(job)
    }

    pub fn retry_job(&self, job_id: &str, request: RetryJob) -> JobsStoreResult<JobSnapshot> {
        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let job = self.get_job_on_connection(&transaction, job_id)?;
        if job.attempts >= MAX_JOB_ATTEMPTS {
            return Err(JobsStoreError::RetryLimit {
                max_attempts: MAX_JOB_ATTEMPTS,
            });
        }
        let mut payload = job.payload;
        payload.extend(request.payload_changes);
        let job = self.create_job_on_connection(
            &transaction,
            CreateJob {
                job_type: job.job_type,
                project_id: job.project_id,
                project_name: job.project_name,
                payload,
                requested_gpu: job.requested_gpu,
                source_job_id: Some(job.id),
                duplicate_of_job_id: None,
                attempts: job.attempts + 1,
            },
            None,
        )?;
        transaction.commit()?;
        Ok(job)
    }

    pub fn duplicate_job(
        &self,
        job_id: &str,
        request: DuplicateJob,
    ) -> JobsStoreResult<JobSnapshot> {
        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let job = self.get_job_on_connection(&transaction, job_id)?;
        let mut payload = job.payload;
        payload.extend(request.payload_changes);
        let job = self.create_job_on_connection(
            &transaction,
            CreateJob {
                job_type: job.job_type,
                project_id: job.project_id,
                project_name: job.project_name,
                payload,
                requested_gpu: request.requested_gpu.unwrap_or(job.requested_gpu),
                source_job_id: None,
                duplicate_of_job_id: Some(job.id),
                attempts: 1,
            },
            None,
        )?;
        transaction.commit()?;
        Ok(job)
    }

    pub fn register_worker(&self, request: RegisterWorker) -> JobsStoreResult<WorkerSnapshot> {
        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let now = utc_now();
        transaction.execute(
            "
            insert into workers (
              id, gpu_id, gpu_name, status, current_job_id, capabilities_json,
              loaded_models_json, utilization_json, registered_at, last_seen_at
            ) values (?1, ?2, ?3, 'idle', null, ?4, ?5, ?6, ?7, ?7)
            on conflict(id) do update set
              gpu_id = excluded.gpu_id,
              gpu_name = excluded.gpu_name,
              status = case when workers.current_job_id is null then 'idle' else workers.status end,
              capabilities_json = excluded.capabilities_json,
              loaded_models_json = excluded.loaded_models_json,
              utilization_json = excluded.utilization_json,
              last_seen_at = excluded.last_seen_at
            ",
            params![
                request.worker_id,
                request.gpu_id,
                request.gpu_name,
                dumps(&request.capabilities)?,
                dumps(&request.loaded_models)?,
                optional_dumps(request.utilization.as_ref())?,
                now,
            ],
        )?;
        let worker = self.get_worker_on_connection(&transaction, &request.worker_id)?;
        transaction.commit()?;
        Ok(worker)
    }

    pub fn heartbeat_worker(&self, request: WorkerHeartbeat) -> JobsStoreResult<WorkerSnapshot> {
        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let worker = self.get_worker_on_connection(&transaction, &request.worker_id)?;
        let now = utc_now();
        if request.current_job_id.is_none() {
            if let Some(previous_job_id) = worker.current_job_id {
                let previous_job = self.get_job_on_connection(&transaction, &previous_job_id)?;
                // Only interrupt a worker's previous active job on an idle heartbeat
                // if that job has already heartbeated at least once. A job that was
                // *just* claimed (no heartbeat yet) may be one another incarnation of
                // the same worker_id claimed microseconds ago — an idle heartbeat
                // racing the claim must not kill it. The time-based stale sweep still
                // reclaims a job abandoned before its first heartbeat.
                if is_active_status(previous_job.status.as_str())
                    && previous_job.last_heartbeat_at.is_some()
                {
                    transaction.execute(
                        "
                        update jobs
                           set status = 'interrupted',
                               stage = 'interrupted',
                               message = 'Job was interrupted after its worker restarted.',
                               error = 'Worker heartbeat no longer referenced the active job.',
                               completed_at = ?1,
                               updated_at = ?1,
                               worker_id = null
                         where id = ?2
                        ",
                        params![now, previous_job_id],
                    )?;
                }
            }
        }
        transaction.execute(
            "
            update workers
               set status = ?1,
                   current_job_id = ?2,
                   loaded_models_json = ?3,
                   utilization_json = ?4,
                   last_seen_at = ?5
             where id = ?6
            ",
            params![
                request.status.as_str(),
                request.current_job_id,
                dumps(&request.loaded_models)?,
                optional_dumps(request.utilization.as_ref())?,
                now,
                request.worker_id,
            ],
        )?;
        if let Some(job_id) = request.current_job_id {
            transaction.execute(
                "update jobs set last_heartbeat_at = ?1, updated_at = ?1 where id = ?2",
                params![now, job_id],
            )?;
        }
        let worker = self.get_worker_on_connection(&transaction, &request.worker_id)?;
        transaction.commit()?;
        Ok(worker)
    }

    pub fn mark_stale_workers_interrupted(
        &self,
        timeout_seconds: u64,
    ) -> JobsStoreResult<StaleSweep> {
        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let now = now_unix_seconds();
        let timeout = i64::try_from(timeout_seconds.max(1)).unwrap_or(i64::MAX);
        let cutoff = format_unix_seconds(now.saturating_sub(timeout));
        let now_text = format_unix_seconds(now);
        let mut statement = transaction.prepare(
            "
            select * from workers
             where status != 'offline'
               and last_seen_at < ?1
            ",
        )?;
        let stale_workers = collect_workers(statement.query_map(params![cutoff], row_to_worker)?)?;
        if stale_workers.is_empty() {
            return Ok(StaleSweep {
                workers: Vec::new(),
                jobs: Vec::new(),
            });
        }

        let worker_ids = stale_workers
            .iter()
            .map(|worker| worker.id.clone())
            .collect::<Vec<_>>();
        drop(statement);
        let active_jobs = self.active_jobs_for_workers(&transaction, &worker_ids)?;
        let placeholders = placeholders_from(2, worker_ids.len());
        let mut job_params = vec![now_text.as_str()];
        job_params.extend(worker_ids.iter().map(String::as_str));
        transaction.execute(
            &format!(
                "
                update jobs
                   set status = 'interrupted',
                       stage = 'interrupted',
                       message = 'Job was interrupted after its worker stopped sending heartbeats.',
                       error = 'Worker heartbeat timed out.',
                       completed_at = ?1,
                       updated_at = ?1,
                       worker_id = null
                 where worker_id in ({placeholders})
                   and status in ({active})
                ",
                active = active_statuses_sql()
            ),
            params_from_iter(job_params),
        )?;

        let mut worker_params = vec![now_text.as_str()];
        worker_params.extend(worker_ids.iter().map(String::as_str));
        transaction.execute(
            &format!(
                "
                update workers
                   set status = 'offline',
                       current_job_id = null,
                       last_seen_at = ?1
                 where id in ({placeholders})
                "
            ),
            params_from_iter(worker_params),
        )?;

        let updated_workers = self.workers_by_ids(&transaction, &worker_ids)?;
        let updated_jobs = active_jobs
            .iter()
            .map(|job| self.get_job_on_connection(&transaction, &job.id))
            .collect::<JobsStoreResult<Vec<_>>>()?;
        transaction.commit()?;
        Ok(StaleSweep {
            workers: updated_workers,
            jobs: updated_jobs,
        })
    }

    /// Surface a worker's death-by-uncatchable-signal (SIGKILL/OOM, SIGABRT,
    /// SIGSEGV, …) as a terminal job FAILURE, instead of letting the heartbeat
    /// sweep later mark it the generic `interrupted` (which reads to the user like
    /// a frozen progress bar). The supervisor that reaped the child observes the
    /// terminating signal — the only layer that can, since the death is
    /// uncatchable in-process — and calls this with that signal. We fail the
    /// worker's still-active job with an actionable, signal-attributed error and
    /// release the worker so the UI doesn't show it pinned to a dead job. Returns
    /// the failed job if the worker had an active one (else `None` — it died idle
    /// between jobs). (sc-4881)
    pub fn fail_worker_job_killed_by_signal(
        &self,
        worker_id: &str,
        signal: i32,
    ) -> JobsStoreResult<Option<JobSnapshot>> {
        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let now = utc_now();
        let worker_ids = [worker_id.to_owned()];
        let active_jobs = self.active_jobs_for_workers(&transaction, &worker_ids)?;
        let mut failed = None;
        if let Some(job) = active_jobs.into_iter().next() {
            // Tailor the OOM/signal hint to the dead job's kind so the guidance is
            // actionable (sc-5567): an image-batch SIGKILL points at count/resolution,
            // not the training-only gradient-checkpointing remediation.
            let error = signal_failure_error(signal, Some(&job.job_type));
            transaction.execute(
                &format!(
                    "
                    update jobs
                       set status = 'failed',
                           stage = 'failed',
                           message = 'Worker process terminated unexpectedly.',
                           error = ?2,
                           completed_at = ?1,
                           updated_at = ?1,
                           worker_id = null
                     where id = ?3
                       and status in ({active})
                    ",
                    active = active_statuses_sql()
                ),
                params![now, error, job.id],
            )?;
            failed = Some(self.get_job_on_connection(&transaction, &job.id)?);
        }
        // Release the worker so it isn't shown pinned to a now-failed job; the
        // supervisor restarts the child, which re-registers itself fresh.
        transaction.execute(
            "
            update workers
               set status = 'offline',
                   current_job_id = null,
                   last_seen_at = ?1
             where id = ?2
            ",
            params![now, worker_id],
        )?;
        transaction.commit()?;
        Ok(failed)
    }

    /// macOS "MLX-required" grace sweep (epic 3482 / sc-3483). When `mlx_required`, the
    /// non-mlx (MPS) worker never claims an MLX-eligible job — it defers unconditionally
    /// to the in-process `mlx` worker (see `should_defer_*`). If no **live** `mlx` worker
    /// claims such a job within the grace window — because the worker is down, never
    /// started, or has been crashed longer than the supervisor's auto-restart can
    /// self-heal — the job would otherwise sit queued forever. This fails those jobs
    /// terminal (`status = failed`) with an actionable `mlx_unavailable` error naming the
    /// model + job type, so the failure is loud and points at the real gap instead of
    /// silently falling back to MPS.
    ///
    /// "Live `mlx` worker" = a `gpu_id = 'mlx'` worker that is not offline and has
    /// heartbeat within the grace window. While one exists (even if it is merely busy),
    /// this is a no-op and the job waits to be claimed; a transient `mlx` crash that the
    /// supervisor restarts inside the window therefore never fails a job. `grace_seconds`
    /// reuses the stale-worker timeout for exactly that reason.
    ///
    /// Off (`mlx_required == false`) it returns immediately, so Windows/Linux/Docker and
    /// the Mac build before the final cutover (sc-3492) are completely unaffected. Returns
    /// the jobs it failed so the caller can surface the structured event in System → Logs
    /// and publish their updates.
    pub fn fail_stranded_mlx_jobs(
        &self,
        mlx_required: bool,
        grace_seconds: u64,
    ) -> JobsStoreResult<Vec<JobSnapshot>> {
        if !mlx_required {
            return Ok(Vec::new());
        }
        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let now = now_unix_seconds();
        let grace = i64::try_from(grace_seconds.max(1)).unwrap_or(i64::MAX);
        let cutoff = format_unix_seconds(now.saturating_sub(grace));

        // A live `mlx` worker (not offline, heartbeat within the window) means MLX-eligible
        // jobs should wait for it — it may simply be busy. Only when none has checked in
        // within the window do we treat MLX as unavailable and fail the stranded jobs.
        let live_mlx_worker = transaction
            .query_row(
                "
                select 1 from workers
                 where gpu_id = 'mlx'
                   and status != 'offline'
                   and last_seen_at >= ?1
                 limit 1
                ",
                params![cutoff],
                |_row| Ok(()),
            )
            .optional()?
            .is_some();
        if live_mlx_worker {
            return Ok(Vec::new());
        }

        // Candidates: still queued and old enough to have outlived the grace window. A job
        // newer than the cutoff keeps waiting (bounded), so a job created mid-outage isn't
        // failed instantly — it gets the full window for an `mlx` worker to appear.
        let mut statement = transaction.prepare(
            "
            select * from jobs
             where status = 'queued'
               and created_at < ?1
             order by created_at asc
            ",
        )?;
        let candidates = collect_jobs(statement.query_map(params![cutoff], row_to_job)?)?;
        drop(statement);

        let now_text = format_unix_seconds(now);
        let mut failed_ids = Vec::new();
        for job in candidates {
            if !job_is_any_mlx_eligible(&job) {
                continue;
            }
            let error = mlx_unavailable_error(&job, grace_seconds);
            transaction.execute(
                "
                update jobs
                   set status = 'failed',
                       stage = 'failed',
                       message = 'MLX worker unavailable.',
                       error = ?2,
                       completed_at = ?1,
                       updated_at = ?1,
                       worker_id = null
                 where id = ?3 and status = 'queued'
                ",
                params![now_text, error, job.id],
            )?;
            failed_ids.push(job.id.clone());
        }
        let failed = failed_ids
            .iter()
            .map(|id| self.get_job_on_connection(&transaction, id))
            .collect::<JobsStoreResult<Vec<_>>>()?;
        transaction.commit()?;
        Ok(failed)
    }

    /// macOS "MLX-unsupported" enforce sweep (epic 3482 / sc-3484). When `mlx_required` AND
    /// `enforce`, fails every queued job the Rust/MLX flow can't run (`mac_rust_supported`
    /// returns `Err`) terminal with a feature-precise `mlx_unsupported` error — the forcing
    /// function that turns "still on torch" into a loud, named failure instead of a silent
    /// fallback. Unlike the stranded sweep there is no grace window: an unsupported job is
    /// permanently unsupported until its surface is ported or dropped, so it fails immediately.
    ///
    /// Default mode is **warn** (`enforce == false`) → this is a no-op and the gap is logged
    /// at claim time instead (the job still runs on torch), so flipping `mlx_required` on for
    /// observation surfaces the gap list without breaking anything. Off (`!mlx_required`) →
    /// immediate no-op, so Windows/Linux/Docker are unaffected. MLX-*eligible* jobs are
    /// `Ok` here and handled by `fail_stranded_mlx_jobs`/routing — the two sweeps partition
    /// the queue and never touch the same job. Returns `(job, reason)` pairs so the caller can
    /// emit the structured event.
    pub fn fail_unsupported_mlx_jobs(
        &self,
        mlx_required: bool,
        enforce: bool,
    ) -> JobsStoreResult<Vec<(JobSnapshot, UnsupportedReason)>> {
        if !mlx_required || !enforce {
            return Ok(Vec::new());
        }
        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let mut statement = transaction
            .prepare("select * from jobs where status = 'queued' order by created_at asc")?;
        let candidates = collect_jobs(statement.query_map([], row_to_job)?)?;
        drop(statement);

        let now_text = format_unix_seconds(now_unix_seconds());
        let mut failed = Vec::new();
        for job in candidates {
            let Err(reason) = mac_rust_supported(&job) else {
                continue;
            };
            transaction.execute(
                "
                update jobs
                   set status = 'failed',
                       stage = 'failed',
                       message = 'Not supported by the Rust/MLX flow on macOS.',
                       error = ?2,
                       completed_at = ?1,
                       updated_at = ?1,
                       worker_id = null
                 where id = ?3 and status = 'queued'
                ",
                params![now_text, reason.error_message(), job.id],
            )?;
            let updated = self.get_job_on_connection(&transaction, &job.id)?;
            failed.push((updated, reason));
        }
        transaction.commit()?;
        Ok(failed)
    }

    pub fn claim_next_job(&self, worker_id: &str) -> JobsStoreResult<Option<JobSnapshot>> {
        Ok(self.claim_next_job_routed(worker_id, false)?.0)
    }

    /// Like [`Self::claim_next_job`], but also reports the MLX↔torch routing decision
    /// so the caller (the API claim handler) can log *why* a job landed where it did —
    /// the single most useful line for diagnosing "MLX-eligible job ran on torch"
    /// (sc-3449). A `None` decision means the claim was routing-neutral: no job was
    /// available, an unrelated balancing deferral fired, or the job is one no `mlx`
    /// worker would ever want.
    pub fn claim_next_job_routed(
        &self,
        worker_id: &str,
        mlx_required: bool,
    ) -> JobsStoreResult<(Option<JobSnapshot>, Option<RouteDecision>)> {
        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        // BEGIN IMMEDIATE: take the write lock up front. The claim reads the worker, the
        // active-gpu-job guard and the full queued set before deciding, then writes. A
        // DEFERRED transaction holds only a read lock through those reads and tries to
        // upgrade at the first UPDATE — and SQLite returns SQLITE_BUSY *immediately* on a
        // lock upgrade (busy_timeout does not retry upgrades, to avoid deadlock), so two
        // overlapping claims would race and one would fail. IMMEDIATE serializes claimers.
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let worker = self.get_worker_on_connection(&transaction, worker_id)?;
        let worker_gpu_id = worker.gpu_id.clone();
        let has_active_gpu_job = active_gpu_job_exists(&transaction, &worker.gpu_id)?;

        let mut statement = transaction.prepare(&format!(
            "
            select * from jobs
             where status = 'queued'
               and (type in ({list}) or requested_gpu = 'auto' or requested_gpu = ?1)
               and (?2 = 0 or type in ({list}))
             order by created_at asc
            ",
            list = non_gpu_job_types_sql()
        ))?;
        let queued_rows = collect_jobs(statement.query_map(
            params![worker.gpu_id, i64::from(has_active_gpu_job)],
            row_to_job,
        )?)?;
        // No row cap (sc-1630): choose_claimable_job must see every gpu/type-gated queued row,
        // or a capability-incompatible prefix (e.g. 50+ jobs the worker can't run) would hide a
        // later compatible job and the worker would sit idle. It also needs the whole compatible
        // set for its priority pass (an explicit-GPU / loaded-model job jumps ahead of an earlier
        // auto-GPU one), so a bounded scan can't preserve that anyway. The WHERE above already
        // narrows rows to this worker's gpu/type lane; pushing the capability filter into SQL is
        // the scale lever if queues ever grow large enough for the full scan to matter.
        let queued = choose_claimable_job(queued_rows, &worker);
        let Some(queued) = queued else {
            return Ok((None, None));
        };
        drop(statement);
        if should_defer_auto_gpu_claim(&transaction, &queued, &worker)? {
            return Ok((None, None));
        }
        if should_defer_image_to_mlx_worker(&transaction, &queued, &worker, mlx_required)?
            || should_defer_video_to_mlx_worker(&transaction, &queued, &worker, mlx_required)?
            || should_defer_training_to_mlx_worker(&transaction, &queued, &worker, mlx_required)?
            || should_defer_caption_to_mlx_worker(&transaction, &queued, &worker, mlx_required)?
            || should_defer_understanding_to_mlx_worker(
                &transaction,
                &queued,
                &worker,
                mlx_required,
            )?
        {
            // A non-mlx worker is yielding this MLX-eligible job to an idle mlx worker.
            let decision = RouteDecision::new(
                &queued,
                &worker_gpu_id,
                worker_id,
                "deferred_to_mlx",
                "idle_mlx_available",
            );
            return Ok((None, Some(decision)));
        }

        let assigned_gpu = if is_non_gpu_job_type(queued.job_type.as_str()) {
            "cpu".to_owned()
        } else {
            worker_gpu_id.clone()
        };
        let now = utc_now();
        transaction.execute(
            "
            update jobs
               set status = 'preparing',
                   assigned_gpu = ?1,
                   worker_id = ?2,
                   stage = 'preparing',
                   message = 'Worker claimed job.',
                   started_at = coalesce(started_at, ?3),
                   updated_at = ?3
             where id = ?4 and status = 'queued'
            ",
            params![assigned_gpu, worker_id, now, queued.id],
        )?;
        transaction.execute(
            "update workers set status = 'busy', current_job_id = ?1, last_seen_at = ?2 where id = ?3",
            params![queued.id, now, worker_id],
        )?;
        let job = self.get_job_on_connection(&transaction, &queued.id)?;
        transaction.commit()?;
        let decision = route_decision_for_claim(&queued, &worker_gpu_id, worker_id);
        Ok((Some(job), decision))
    }

    pub fn update_job_progress(
        &self,
        job_id: &str,
        update: ProgressUpdate,
    ) -> JobsStoreResult<JobSnapshot> {
        if !JOB_STATUSES.contains(&update.status.as_str()) {
            return Err(JobsStoreError::InvalidStatus(
                update.status.as_str().to_owned(),
            ));
        }

        if !update.progress.is_finite() {
            return Err(JobsStoreError::InvalidNumber("progress".to_owned()));
        }
        if update.eta_seconds.is_some_and(|value| !value.is_finite()) {
            return Err(JobsStoreError::InvalidNumber("etaSeconds".to_owned()));
        }
        if update
            .peak_gpu_memory_pct
            .is_some_and(|value| !value.is_finite())
        {
            return Err(JobsStoreError::InvalidNumber("peakGpuMemoryPct".to_owned()));
        }
        if update
            .peak_gpu_load_pct
            .is_some_and(|value| !value.is_finite())
        {
            return Err(JobsStoreError::InvalidNumber("peakGpuLoadPct".to_owned()));
        }

        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        // Guard against zombie-worker writes (sc-4172): a worker that went
        // silent long enough for the stale sweep to mark its job `interrupted`
        // (or whose job the user canceled) must not resurrect it with a late
        // progress report — that's exactly the failure mode the heartbeat
        // machinery exists to handle.
        let current = self.get_job_on_connection(&transaction, job_id)?;
        if is_terminal_status(current.status.as_str()) {
            // Idempotent re-report of the same terminal status (e.g. a retried
            // "canceled" POST) succeeds without touching the row.
            if current.status == update.status {
                return Ok(current);
            }
            return Err(JobsStoreError::TerminalJobImmutable {
                job_id: job_id.to_owned(),
                status: current.status.as_str().to_owned(),
            });
        }
        match (update.worker_id.as_deref(), current.worker_id.as_deref()) {
            (Some(reporter), Some(owner)) if reporter == owner => {}
            (None, None) => {}
            _ => {
                return Err(JobsStoreError::NotJobOwner {
                    job_id: job_id.to_owned(),
                });
            }
        }
        let now = utc_now();
        let completed_at = is_terminal_status(update.status.as_str()).then_some(now.clone());
        let canceled_at = (update.status == JobStatus::Canceled).then_some(now.clone());
        let progress = update.progress.clamp(0.0, 1.0);
        // Peaks are clamped to 0..100 and persisted as a running max so a stale
        // progress report (lower sample) can't ratchet the peak down (sc-2086).
        let peak_memory = update
            .peak_gpu_memory_pct
            .map(|value| value.clamp(0.0, 100.0));
        let peak_load = update
            .peak_gpu_load_pct
            .map(|value| value.clamp(0.0, 100.0));
        let mut result = update.result;
        if let Some(result) = result.as_mut() {
            // Reuse the result we already read above (same transaction/row) rather
            // than re-selecting result_json each update (sc-4274 / F-CORE-14).
            merge_training_sample_history(Some(&current.result), result);
        }
        transaction.execute(
            "
            update jobs
               set status = ?1,
                   stage = ?2,
                   progress = ?3,
                   message = ?4,
                   error = ?5,
                   result_json = coalesce(?6, result_json),
                   eta_seconds = ?7,
                   completed_at = coalesce(?8, completed_at),
                   canceled_at = coalesce(?9, canceled_at),
                   updated_at = ?10,
                   peak_gpu_memory_pct = case
                       when ?11 is null then peak_gpu_memory_pct
                       else max(coalesce(peak_gpu_memory_pct, 0), ?11)
                   end,
                   peak_gpu_load_pct = case
                       when ?12 is null then peak_gpu_load_pct
                       else max(coalesce(peak_gpu_load_pct, 0), ?12)
                   end,
                   backend = coalesce(backend, ?13)
             where id = ?14
            ",
            params![
                update.status.as_str(),
                update.stage.as_str(),
                progress,
                update.message,
                update.error,
                optional_dumps(result.as_ref())?,
                update.eta_seconds,
                completed_at,
                canceled_at,
                now,
                peak_memory,
                peak_load,
                update.backend,
                job_id,
            ],
        )?;
        let job = self.get_job_on_connection(&transaction, job_id)?;
        if is_terminal_status(update.status.as_str()) {
            if let Some(worker_id) = &job.worker_id {
                transaction.execute(
                    "update workers set status = 'idle', current_job_id = null, last_seen_at = ?1 where id = ?2",
                    params![now, worker_id],
                )?;
            }
        }
        transaction.commit()?;
        Ok(job)
    }

    pub fn list_workers(&self) -> JobsStoreResult<Vec<WorkerSnapshot>> {
        let _guard = self.lock.lock();
        let connection = self.connect()?;
        let mut statement = connection.prepare("select * from workers order by gpu_id, id")?;
        let workers = collect_workers(statement.query_map([], row_to_worker)?)?;
        Ok(workers)
    }

    pub fn get_worker(&self, worker_id: &str) -> JobsStoreResult<WorkerSnapshot> {
        let _guard = self.lock.lock();
        let connection = self.connect()?;
        self.get_worker_on_connection(&connection, worker_id)
    }

    pub fn queue_summary(&self) -> JobsStoreResult<QueueSummary> {
        // list_workers takes the store lock itself, so resolve it before we take
        // the lock below to avoid a nested (potential self-deadlock) acquisition.
        let workers = self.list_workers()?;
        let _guard = self.lock.lock();
        let connection = self.connect()?;

        // Per-status counts over the WHOLE table — never a capped/newest-N sample.
        // Filtering an already-capped list silently undercounts once a project
        // exceeds the cap (sc-4208 / F-CORE-4). Seed every known status at 0 so
        // the map shape is stable for callers regardless of what rows exist.
        let mut counts = JOB_STATUSES
            .iter()
            .map(|status| (parse_string_enum::<JobStatus>(status), 0u32))
            .collect::<std::collections::BTreeMap<_, _>>();
        let mut statement =
            connection.prepare("select status, count(*) from jobs group by status")?;
        let rows = statement.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?;
        for row in rows {
            let (status, count) = row?;
            // Writes are constrained to JOB_STATUSES so the seeded entry exists;
            // or_insert keeps an unexpected value counted rather than dropped.
            *counts
                .entry(parse_string_enum::<JobStatus>(&status))
                .or_insert(0) += u32::try_from(count).unwrap_or(u32::MAX);
        }

        // Active (non-terminal, includes `queued`) jobs come from a dedicated
        // uncapped query so an old still-queued/running job can't fall out of the
        // newest-N window and become invisible to the operator.
        let mut statement = connection.prepare(&format!(
            "select * from jobs where status not in ({terminal}) order by created_at desc",
            terminal = terminal_statuses_sql()
        ))?;
        let active_jobs = collect_jobs(statement.query_map([], row_to_job)?)?;

        Ok(QueueSummary {
            counts,
            active_jobs,
            workers,
            max_job_attempts: MAX_JOB_ATTEMPTS,
            extra: Default::default(),
        })
    }

    fn connect(&self) -> JobsStoreResult<Connection> {
        if let Some(parent) = self.db_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let connection = Connection::open(&self.db_path)?;
        // Wait (instead of failing instantly) when another connection/process holds the
        // database lock. rusqlite's default busy timeout is 0ms, so any cross-process
        // overlap — e.g. a sidecar restart where the old process hasn't fully released the
        // db, or a concurrent claim/heartbeat — surfaces as `database is locked` and the
        // job loses its claim (MLX-eligible jobs then fall through to the torch worker).
        // A 5s wait lets the holder finish; paired with BEGIN IMMEDIATE on write
        // transactions (below), writers queue cleanly rather than deadlocking on lock upgrade.
        connection.busy_timeout(Duration::from_millis(5000))?;
        match connection.pragma_update(None, "journal_mode", "wal") {
            Ok(()) => {}
            Err(error) => {
                // WAL almost always succeeds. When it can't be set, do NOT delete
                // the `-wal`/`-shm` sidecars: they may belong to a live connection
                // in another process, and removing them can corrupt that
                // connection's view. Nor do we silently force `delete` mode — the
                // 5s busy_timeout reasoning above assumes WAL lets writers queue,
                // so a silent drop to rollback-journal would change concurrency
                // semantics for the rest of the process with no signal. Leave the
                // connection in whatever mode it opened with and warn loudly
                // instead (sc-4275 / F-CORE-16).
                eprintln!(
                    "WARNING: SceneWorks could not enable SQLite WAL mode for {} ({error}); \
                     continuing in the default rollback-journal mode. Cross-process write \
                     concurrency will be more serialized than usual.",
                    self.db_path.display()
                );
            }
        }
        connection.pragma_update(None, "foreign_keys", "on")?;
        Ok(connection)
    }

    fn create_job_on_connection(
        &self,
        connection: &Connection,
        request: CreateJob,
        job_id: Option<String>,
    ) -> JobsStoreResult<JobSnapshot> {
        let requested_gpu = normalize_requested_gpu(&request.requested_gpu);
        if job_requires_gpu(&request.job_type) && requested_gpu == "cpu" {
            return Err(JobsStoreError::InvalidRequestedGpu(format!(
                "{} jobs cannot target CPU workers. Choose auto or a GPU id.",
                request.job_type.as_str()
            )));
        }
        let now = utc_now();
        let job_id = match job_id {
            Some(job_id) => job_id,
            None => {
                let job_hex: String =
                    connection
                        .query_row("select lower(hex(randomblob(16)))", [], |row| row.get(0))?;
                format!("job_{job_hex}")
            }
        };
        connection.execute(
            "
            insert into jobs (
              id, type, status, project_id, project_name, payload_json, result_json,
              requested_gpu, progress, stage, message, attempts, source_job_id,
              duplicate_of_job_id, created_at, updated_at
            ) values (?1, ?2, 'queued', ?3, ?4, ?5, '{}', ?6, 0, 'queued', ?7, ?8, ?9, ?10, ?11, ?11)
            ",
            params![
                job_id,
                request.job_type.as_str(),
                request.project_id,
                request.project_name,
                dumps(&request.payload)?,
                requested_gpu,
                "Waiting for an available worker.",
                request.attempts,
                request.source_job_id,
                request.duplicate_of_job_id,
                now,
            ],
        )?;
        self.get_job_on_connection(connection, &job_id)
    }

    fn list_jobs_by_status_on_connection(
        &self,
        connection: &Connection,
        statuses: &[&str],
    ) -> JobsStoreResult<Vec<JobSnapshot>> {
        let mut jobs = Vec::new();
        for status in statuses {
            let mut statement = connection.prepare("select * from jobs where status = ?1")?;
            jobs.extend(collect_jobs(
                statement.query_map(params![status], row_to_job)?,
            )?);
        }
        Ok(jobs)
    }

    fn active_jobs_for_workers(
        &self,
        connection: &Connection,
        worker_ids: &[String],
    ) -> JobsStoreResult<Vec<JobSnapshot>> {
        if worker_ids.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders = placeholders_from(1, worker_ids.len());
        let mut statement = connection.prepare(&format!(
            "
            select * from jobs
             where worker_id in ({placeholders})
               and status in ({active})
            ",
            active = active_statuses_sql()
        ))?;
        let jobs = collect_jobs(statement.query_map(
            params_from_iter(worker_ids.iter().map(String::as_str)),
            row_to_job,
        )?)?;
        Ok(jobs)
    }

    fn workers_by_ids(
        &self,
        connection: &Connection,
        worker_ids: &[String],
    ) -> JobsStoreResult<Vec<WorkerSnapshot>> {
        if worker_ids.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders = placeholders_from(1, worker_ids.len());
        let mut statement = connection.prepare(&format!(
            "select * from workers where id in ({placeholders}) order by gpu_id, id"
        ))?;
        let workers = collect_workers(statement.query_map(
            params_from_iter(worker_ids.iter().map(String::as_str)),
            row_to_worker,
        )?)?;
        Ok(workers)
    }

    fn get_job_on_connection(
        &self,
        connection: &Connection,
        job_id: &str,
    ) -> JobsStoreResult<JobSnapshot> {
        connection
            .query_row(
                "select * from jobs where id = ?1",
                params![job_id],
                row_to_job,
            )
            .optional()?
            .ok_or_else(|| JobsStoreError::NotFound(job_id.to_owned()))
    }

    fn get_worker_on_connection(
        &self,
        connection: &Connection,
        worker_id: &str,
    ) -> JobsStoreResult<WorkerSnapshot> {
        connection
            .query_row(
                "select * from workers where id = ?1",
                params![worker_id],
                row_to_worker,
            )
            .optional()?
            .ok_or_else(|| JobsStoreError::NotFound(worker_id.to_owned()))
    }
}

fn row_to_job(row: &Row<'_>) -> rusqlite::Result<JobSnapshot> {
    let progress: f64 = row.get("progress")?;
    let eta_seconds: Option<f64> = row.get("eta_seconds")?;
    let peak_memory: Option<f64> = row.get("peak_gpu_memory_pct").ok().flatten();
    let peak_load: Option<f64> = row.get("peak_gpu_load_pct").ok().flatten();
    let backend: Option<String> = row.get("backend").ok().flatten();
    let created_at: String = row.get("created_at")?;
    let started_at: Option<String> = row.get("started_at")?;
    let completed_at: Option<String> = row.get("completed_at")?;
    let elapsed_seconds = started_at
        .as_deref()
        .and_then(|started| elapsed_seconds(started, completed_at.as_deref()));
    let job_type: JobType = parse_string_enum(&row.get::<_, String>("type")?);
    let payload = loads_object(row.get::<_, Option<String>>("payload_json")?.as_deref());
    let title = derive_job_title(&job_type, &payload);
    Ok(JobSnapshot {
        id: row.get("id")?,
        job_type,
        status: parse_string_enum(&row.get::<_, String>("status")?),
        project_id: row.get("project_id")?,
        project_name: row.get("project_name")?,
        payload,
        result: loads_object(row.get::<_, Option<String>>("result_json")?.as_deref()),
        requested_gpu: row.get("requested_gpu")?,
        assigned_gpu: row.get("assigned_gpu")?,
        worker_id: row.get("worker_id")?,
        progress: number_from_f64(progress),
        stage: parse_string_enum(&row.get::<_, String>("stage")?),
        message: row.get("message")?,
        error: row.get("error")?,
        eta_seconds: eta_seconds.map(number_from_f64),
        elapsed_seconds,
        attempts: row.get::<_, u32>("attempts")?,
        source_job_id: row.get("source_job_id")?,
        duplicate_of_job_id: row.get("duplicate_of_job_id")?,
        cancel_requested: row.get::<_, i64>("cancel_requested")? != 0,
        created_at,
        updated_at: row.get("updated_at")?,
        started_at,
        completed_at,
        canceled_at: row.get("canceled_at")?,
        last_heartbeat_at: row.get("last_heartbeat_at")?,
        peak_gpu_memory_pct: peak_memory.map(number_from_f64),
        peak_gpu_load_pct: peak_load.map(number_from_f64),
        backend,
        title,
        extra: Default::default(),
    })
}

/// Server-side derivation of the human-readable job title surfaced in the
/// queue and WorkerProgressCard (sc-2087). Mirrors the Job Title table in
/// docs/design/worker-progress-card.md. Returns None for types where the
/// payload doesn't carry a meaningful subject — the frontend then falls back
/// to its own derivation, keeping the queue from ever showing only a raw job
/// id as the row identifier.
fn derive_job_title(job_type: &JobType, payload: &Map<String, Value>) -> Option<String> {
    /// Find the first string value at any of the candidate keys.
    fn first_str<'a>(payload: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a str> {
        keys.iter()
            .find_map(|key| payload.get(*key).and_then(Value::as_str))
            .filter(|value| !value.trim().is_empty())
    }
    /// Truncate a prompt to ~max chars on a word boundary, append an ellipsis
    /// when truncated. Mirrors the JS helper in WorkerProgressCard.jsx.
    fn truncate_prompt(prompt: &str, max: usize) -> String {
        if prompt.len() <= max {
            return prompt.to_owned();
        }
        let mut cut = prompt[..max].to_owned();
        if let Some(space) = cut.rfind(' ') {
            if space > (max * 6) / 10 {
                cut.truncate(space);
            }
        }
        format!("{}…", cut.trim_end())
    }

    match job_type {
        JobType::LoraTrain => {
            let subject = first_str(payload, &["loraName", "outputName", "targetName", "loraId"])
                .map(str::to_owned)
                .or_else(|| {
                    payload
                        .get("plan")
                        .and_then(|plan| plan.get("output"))
                        .and_then(|output| output.get("loraId"))
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .unwrap_or_else(|| "(unnamed LoRA)".to_owned());
            Some(format!("Training Run — {subject}"))
        }
        JobType::TrainingCaption => {
            let subject = first_str(payload, &["datasetName", "datasetId"])
                .unwrap_or("(unnamed dataset)")
                .to_owned();
            Some(format!("Dataset Captioning — {subject}"))
        }
        JobType::ImageGenerate
        | JobType::ImageEdit
        | JobType::ImageVqa
        | JobType::ImageInterleave => {
            // Character Turnaround override: a character generation has
            // characterId + characterName on the payload.
            if payload.get("characterId").and_then(Value::as_str).is_some() {
                if let Some(name) = first_str(payload, &["characterName"]) {
                    return Some(format!("Character Turnaround — {name}"));
                }
            }
            let prompt = first_str(payload, &["prompt"]).unwrap_or("(no prompt)");
            Some(format!("Generate Image — {}", truncate_prompt(prompt, 80)))
        }
        JobType::VideoGenerate | JobType::VideoExtend | JobType::VideoBridge => {
            let prompt = first_str(payload, &["prompt"]).unwrap_or("(no prompt)");
            Some(format!("Generate Video — {}", truncate_prompt(prompt, 80)))
        }
        JobType::PersonReplace => {
            let prompt = first_str(payload, &["prompt"]).unwrap_or("(no prompt)");
            Some(format!("Person Replace — {}", truncate_prompt(prompt, 80)))
        }
        JobType::ModelDownload | JobType::ModelImport | JobType::ModelConvert => {
            let subject =
                first_str(payload, &["modelName", "filename", "modelId", "repo"]).unwrap_or("");
            if subject.is_empty() {
                Some("Model Import".to_owned())
            } else {
                Some(format!("Model Import — {subject}"))
            }
        }
        JobType::LoraImport => {
            let subject = first_str(payload, &["loraName", "filename", "loraId"]).unwrap_or("");
            if subject.is_empty() {
                Some("LoRA Import".to_owned())
            } else {
                Some(format!("LoRA Import — {subject}"))
            }
        }
        JobType::LoraDownload => {
            let subject = first_str(payload, &["loraName", "loraId", "repo"]).unwrap_or("");
            if subject.is_empty() {
                Some("LoRA Download".to_owned())
            } else {
                Some(format!("LoRA Download — {subject}"))
            }
        }
        JobType::PromptRefine => {
            let prompt = first_str(payload, &["prompt"]).unwrap_or("(empty prompt)");
            Some(format!("Prompt Refine — {}", truncate_prompt(prompt, 60)))
        }
        // Person detect/track/segment + anything else — let the frontend
        // fall back to its own derivation.
        _ => None,
    }
}

fn row_to_worker(row: &Row<'_>) -> rusqlite::Result<WorkerSnapshot> {
    Ok(WorkerSnapshot {
        id: row.get("id")?,
        gpu_id: row.get("gpu_id")?,
        gpu_name: row.get("gpu_name")?,
        status: parse_string_enum(&row.get::<_, String>("status")?),
        current_job_id: row.get("current_job_id")?,
        capabilities: loads_vec(
            row.get::<_, Option<String>>("capabilities_json")?
                .as_deref(),
        ),
        loaded_models: loads_vec(
            row.get::<_, Option<String>>("loaded_models_json")?
                .as_deref(),
        ),
        utilization: loads_optional(row.get::<_, Option<String>>("utilization_json")?.as_deref()),
        registered_at: row.get("registered_at")?,
        last_seen_at: row.get("last_seen_at")?,
        extra: Default::default(),
    })
}

fn collect_jobs<F>(rows: rusqlite::MappedRows<'_, F>) -> JobsStoreResult<Vec<JobSnapshot>>
where
    F: FnMut(&Row<'_>) -> rusqlite::Result<JobSnapshot>,
{
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

fn collect_workers<F>(rows: rusqlite::MappedRows<'_, F>) -> JobsStoreResult<Vec<WorkerSnapshot>>
where
    F: FnMut(&Row<'_>) -> rusqlite::Result<WorkerSnapshot>,
{
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

fn dumps<T: serde::Serialize>(value: &T) -> JobsStoreResult<String> {
    let mut value = serde_json::to_value(value)?;
    sort_json_value(&mut value);
    serde_json::to_string(&value).map_err(Into::into)
}

fn optional_dumps<T: serde::Serialize>(value: Option<&T>) -> JobsStoreResult<Option<String>> {
    value.map(dumps).transpose()
}

fn loads_object(value: Option<&str>) -> Map<String, Value> {
    value
        .and_then(|text| serde_json::from_str::<Map<String, Value>>(text).ok())
        .unwrap_or_default()
}

/// Merge accumulated `trainingSamples` history into an incoming progress
/// result. `existing_result` is the job's current result, which
/// `update_job_progress` has already read in the same transaction — so this no
/// longer re-`select`s `result_json` per update (sc-4274 / F-CORE-14).
fn merge_training_sample_history(
    existing_result: Option<&Map<String, Value>>,
    incoming: &mut Map<String, Value>,
) {
    let has_training_samples = incoming
        .get("trainingSamples")
        .and_then(Value::as_array)
        .is_some();
    let has_latest_training_samples = incoming
        .get("latestTrainingSamples")
        .and_then(Value::as_array)
        .is_some();
    if !has_training_samples && !has_latest_training_samples {
        return;
    }

    let mut samples = Vec::new();
    let mut seen = std::collections::HashSet::new();
    append_training_samples(
        &mut samples,
        &mut seen,
        existing_result.and_then(|result| result.get("trainingSamples")),
    );
    append_training_samples(&mut samples, &mut seen, incoming.get("trainingSamples"));
    append_training_samples(
        &mut samples,
        &mut seen,
        incoming.get("latestTrainingSamples"),
    );

    if !samples.is_empty() {
        incoming.insert("trainingSamples".to_owned(), Value::Array(samples));
    }
}

fn append_training_samples(
    samples: &mut Vec<Value>,
    seen: &mut std::collections::HashSet<String>,
    value: Option<&Value>,
) {
    let Some(array) = value.and_then(Value::as_array) else {
        return;
    };
    for sample in array {
        let key = training_sample_key(sample, samples.len());
        if seen.insert(key) {
            samples.push(sample.clone());
        }
    }
}

fn training_sample_key(sample: &Value, fallback_index: usize) -> String {
    let Some(object) = sample.as_object() else {
        return format!("sample:{fallback_index}");
    };
    for key in ["relativePath", "path", "url"] {
        if let Some(value) = object.get(key).and_then(Value::as_str) {
            if !value.is_empty() {
                return format!("{key}:{value}");
            }
        }
    }
    let step = object
        .get("step")
        .map(Value::to_string)
        .unwrap_or_else(|| "unknown".to_owned());
    let prompt = object
        .get("prompt")
        .and_then(Value::as_str)
        .unwrap_or_default();
    format!("step:{step}:prompt:{prompt}:index:{fallback_index}")
}

fn loads_vec<T>(value: Option<&str>) -> Vec<T>
where
    T: DeserializeOwned,
{
    value
        .and_then(|text| serde_json::from_str::<Vec<T>>(text).ok())
        .unwrap_or_default()
}

fn loads_optional<T>(value: Option<&str>) -> Option<T>
where
    T: DeserializeOwned,
{
    // Best-effort worker telemetry should disappear rather than poison the queue.
    value.and_then(|text| serde_json::from_str::<T>(text).ok())
}

fn number_from_f64(value: f64) -> ContractNumber {
    Number::from_f64(value).unwrap_or_else(|| Number::from(0))
}

fn elapsed_seconds(started_at: &str, completed_at: Option<&str>) -> Option<ContractNumber> {
    let started = parse_utc_seconds(started_at)?;
    let ended = completed_at.map_or_else(|| Some(now_unix_seconds()), parse_utc_seconds)?;
    let seconds = ended.saturating_sub(started).max(0);
    Some(Number::from(seconds))
}

fn parse_utc_seconds(value: &str) -> Option<i64> {
    if value.len() < 20 {
        return None;
    }
    let year = value.get(0..4)?.parse::<i32>().ok()?;
    let month = value.get(5..7)?.parse::<u32>().ok()?;
    let day = value.get(8..10)?.parse::<u32>().ok()?;
    let hour = value.get(11..13)?.parse::<i64>().ok()?;
    let minute = value.get(14..16)?.parse::<i64>().ok()?;
    let second = value.get(17..19)?.parse::<i64>().ok()?;
    let suffix = value.get(19..)?;
    if value.get(4..5)? != "-"
        || value.get(7..8)? != "-"
        || value.get(10..11)? != "T"
        || value.get(13..14)? != ":"
        || value.get(16..17)? != ":"
        || month == 0
        || month > 12
        || day == 0
        || day > 31
        || hour > 23
        || minute > 59
        || second > 59
    {
        return None;
    }
    if suffix != "Z" {
        if !suffix.starts_with('.') || !suffix.ends_with('Z') {
            return None;
        }
        if !suffix[1..suffix.len() - 1]
            .chars()
            .all(|character| character.is_ascii_digit())
        {
            return None;
        }
    }
    Some(days_from_civil(year, month, day) * 86_400 + hour * 3_600 + minute * 60 + second)
}

fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    let adjusted_year = i64::from(year) - i64::from(month <= 2);
    let era = adjusted_year.div_euclid(400);
    let year_of_era = adjusted_year - era * 400;
    let month = i64::from(month);
    let day = i64::from(day);
    let month_prime = month + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * month_prime + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

fn is_active_status(status: &str) -> bool {
    ACTIVE_STATUSES.contains(&status)
}

fn is_terminal_status(status: &str) -> bool {
    TERMINAL_STATUSES.contains(&status)
}

fn is_non_gpu_job_type(job_type: &str) -> bool {
    NON_GPU_JOB_TYPES.contains(&job_type)
}

/// The MLX↔torch routing decision for a single claim, emitted as a structured log
/// event (`mlx_route_decision`) by the API so operators can see *why* a job ran where
/// it did (sc-3449) — the line that answers "why did this MLX-eligible job run on torch?".
#[derive(Debug, Clone, serde::Serialize)]
pub struct RouteDecision {
    pub job_id: String,
    pub job_type: String,
    pub model: Option<String>,
    pub requested_gpu: String,
    pub worker_id: String,
    pub gpu_id: String,
    /// `deferred_to_mlx` | `claimed_by_mlx` | `fell_back_to_torch` | `explicit_gpu`.
    pub decision: &'static str,
    /// Machine-readable cause: `idle_mlx_available`, `mlx_worker`, `no_idle_mlx_worker`,
    /// or `explicit_gpu`.
    pub reason: &'static str,
}

impl RouteDecision {
    fn new(
        job: &JobSnapshot,
        gpu_id: &str,
        worker_id: &str,
        decision: &'static str,
        reason: &'static str,
    ) -> Self {
        Self {
            job_id: job.id.clone(),
            job_type: job.job_type.as_str().to_owned(),
            model: job
                .payload
                .get("model")
                .and_then(Value::as_str)
                .map(str::to_owned),
            requested_gpu: job.requested_gpu.clone(),
            worker_id: worker_id.to_owned(),
            gpu_id: gpu_id.to_owned(),
            decision,
            reason,
        }
    }
}

/// True when *any* MLX-routing predicate (image/detail, video, or training) claims this
/// job — the union an `mlx` worker would want. Used both to classify a claim for routing
/// observability (sc-3449) and to identify the jobs the macOS grace sweep must fail when
/// no `mlx` worker is alive (sc-3483).
fn job_is_any_mlx_eligible(job: &JobSnapshot) -> bool {
    job_is_mlx_eligible(job)
        || video_job_is_mlx_eligible(job)
        || video_upscale_job_is_mlx_eligible(job)
        || training_job_is_mlx_eligible(job)
        || caption_job_is_mlx_eligible(job)
        || understanding_job_is_mlx_eligible(job)
}

/// Actionable terminal error for an MLX-eligible job stranded on macOS with no live `mlx`
/// worker (sc-3483). Names the model + job type so the job card and the System → Logs
/// surface point at the real gap, never a generic failure. Prefixed `mlx_unavailable:` so
/// the cause is greppable in logs and distinguishable from `mlx_unsupported` (sc-3484).
fn mlx_unavailable_error(job: &JobSnapshot, grace_seconds: u64) -> String {
    let model = job
        .payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("(unknown)");
    format!(
        "mlx_unavailable: the MLX GPU worker is required on macOS but no live worker \
         claimed this job within {grace_seconds}s (model={model}, type={job_type}). The \
         Python/MPS fallback is disabled on Mac — check System → Logs and confirm the MLX \
         worker is running.",
        job_type = job.job_type.as_str()
    )
}

/// Human, actionable terminal error attributing a worker's death to its
/// terminating signal (sc-4881). Signal 9 (an uncatchable SIGKILL — almost always an
/// OS memory-pressure OOM kill) carries a remediation hint tailored to the dead job's
/// kind (sc-5567): training points at gradient checkpointing (the sc-4874 first-step
/// OOM), image/video generation at the knobs that actually shrink the working set
/// (batch count, resolution, frame count). Other uncatchable deaths (SIGABRT GPU/Metal
/// abort, SIGSEGV) name themselves so the job card and System → Logs show a real cause
/// instead of a frozen progress bar. `job_type` is the failed job's kind when one was
/// active (`None` when the worker died idle).
fn signal_failure_error(signal: i32, job_type: Option<&JobType>) -> String {
    let hint = match signal {
        9 => oom_remediation_hint(job_type),
        6 => ", likely a GPU/Metal command-buffer abort or assertion",
        11 => " (segmentation fault)",
        _ => "",
    };
    match signal_name(signal) {
        Some(name) => format!("Worker terminated by signal {signal} ({name}){hint}."),
        None => format!("Worker terminated by signal {signal}{hint}."),
    }
}

/// Signal-9 (SIGKILL/OOM) remediation hint keyed to the dead job's kind so the guidance
/// is actionable rather than training-centric (sc-5567). The `_` arm covers the long tail
/// of non-generation job types (and is required anyway — `JobType` is `#[non_exhaustive]`).
fn oom_remediation_hint(job_type: Option<&JobType>) -> &'static str {
    match job_type {
        // LoRA training: the sc-4874 first-training-step OOM — gradient checkpointing is
        // the real lever; resolution is secondary.
        Some(JobType::LoraTrain) => {
            ", likely out-of-memory during the first training step \
             — enable Gradient Checkpointing or reduce resolution"
        }
        // Video generation/edit: working set scales with resolution AND frame count.
        Some(
            JobType::VideoGenerate
            | JobType::VideoExtend
            | JobType::VideoBridge
            | JobType::VideoUpscale
            | JobType::PersonReplace,
        ) => ", likely out-of-memory — reduce the resolution, frame count, or batch count",
        // Image generation/edit: a multi-image batch stacks per-image working set — count
        // is the first knob, then resolution (sc-5567).
        Some(
            JobType::ImageGenerate
            | JobType::ImageEdit
            | JobType::ImageUpscale
            | JobType::ImageDetail
            | JobType::ImageVqa
            | JobType::ImageInterleave,
        ) => ", likely out-of-memory — reduce the image count or resolution",
        _ => ", likely out-of-memory — reduce the resolution or batch count",
    }
}

/// Conventional name for the common terminating signals we attribute (sc-4881).
fn signal_name(signal: i32) -> Option<&'static str> {
    Some(match signal {
        1 => "SIGHUP",
        2 => "SIGINT",
        6 => "SIGABRT",
        9 => "SIGKILL",
        11 => "SIGSEGV",
        15 => "SIGTERM",
        _ => return None,
    })
}

/// Why the Rust/MLX flow can't run a job on macOS (epic 3482 / sc-3484) — the inverse of the
/// `*_mlx_eligible` predicates, extended across every job type. Feature-precise so the
/// `mlx_unsupported` Logs event + the gap inventory name the exact surface to port or drop.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UnsupportedReason {
    /// Model id involved, when the gap is model-specific (e.g. "kolors", "qwen_image").
    pub model: Option<String>,
    /// The specific capability that isn't in the Rust/MLX flow (e.g. "strict-pose ControlNet",
    /// "third-party LyCORIS LoRA", "image_upscale (Real-ESRGAN)").
    pub feature: String,
    /// Actionable human-readable explanation.
    pub detail: String,
    /// Closing story/epic ("epic 3401", "sc-3489"), `"drop-candidate"`, or `None` when not yet
    /// triaged — the roadmap pointer. "where known" per the story.
    pub suggested_epic: Option<String>,
}

impl UnsupportedReason {
    fn new(model: Option<&str>, feature: &str, detail: &str, suggested_epic: Option<&str>) -> Self {
        Self {
            model: model.map(str::to_owned),
            feature: feature.to_owned(),
            detail: detail.to_owned(),
            suggested_epic: suggested_epic.map(str::to_owned),
        }
    }

    /// Terminal job error for an enforced `mlx_unsupported` failure (sc-3484): greppable
    /// prefix, names feature + model + roadmap pointer.
    pub fn error_message(&self) -> String {
        let model = self
            .model
            .as_deref()
            .map(|m| format!(" ({m})"))
            .unwrap_or_default();
        let pointer = self
            .suggested_epic
            .as_deref()
            .map(|epic| format!(" [{epic}]"))
            .unwrap_or_default();
        format!(
            "mlx_unsupported: {feature}{model} is not in the Rust/MLX flow on macOS — {detail}{pointer}",
            feature = self.feature,
            detail = self.detail,
        )
    }
}

/// macOS "can the Rust/MLX flow run this?" oracle (sc-3484). `Ok(())` = the in-process mlx
/// worker — or an MLX-agnostic in-process path (downloads, ffmpeg, prompt refine) — runs it
/// with no Python torch dependency. `Err` names the exact Python-torch gap. This is the epic's
/// *forcing function*: under mlx-required **enforce** mode an `Err` job fails terminal with
/// `mlx_unsupported`, and the set of `Err`s IS the port-or-drop roadmap. Consistent with
/// routing by construction — anything `job_is_any_mlx_eligible` accepts is `Ok`.
pub fn mac_rust_supported(job: &JobSnapshot) -> Result<(), UnsupportedReason> {
    if job_is_any_mlx_eligible(job) {
        return Ok(());
    }
    let model = job.payload.get("model").and_then(Value::as_str);
    match job.job_type {
        // In-process macOS job types with no Python torch dependency: MLX-agnostic metadata/utility
        // work + ffmpeg, plus prompt refine — now served by the native MLX `prompt_refine` TextLlm
        // provider (sc-5552, the mlx twin of the candle sc-5525 cutover), so the worker advertises +
        // claims it and this `Ok` is backed by a real capability (no longer the pre-sc-5552 strand).
        JobType::Placeholder
        | JobType::ModelDownload
        | JobType::ModelImport
        | JobType::LoraImport
        | JobType::LoraDownload
        | JobType::FrameExtract
        | JobType::TimelineExport
        | JobType::PromptRefine => Ok(()),

        // Forward-compat: an unrecognized job type isn't a known Python-torch gap, so don't
        // enforce-fail it (it would otherwise break a newer job type this build doesn't model).
        JobType::Unknown(_) => Ok(()),

        JobType::ImageGenerate | JobType::ImageEdit => Err(classify_image_gap(&job.payload)),

        JobType::ImageDetail => Err(UnsupportedReason::new(
            model,
            "non-SDXL tile-detail refine",
            "image_detail is ported to MLX only for the SDXL/RealVisXL backbones (sc-3060); other models / third-party LyCORIS stay on the Python torch path.",
            Some("epic 3041"),
        )),

        // SenseNova-U1 VQA + Document-Studio interleave are ported to the Rust MLX worker
        // (sc-3905, via the concrete `T2iModel` — the `Generator` contract can't express
        // text / text+image output); eligible jobs early-return `Ok` above. This arm is
        // reached only for an understanding job on a model with no in-process path.
        JobType::ImageVqa | JobType::ImageInterleave => Err(UnsupportedReason::new(
            model,
            "image understanding / interleave on this model",
            "image VQA / interleaved generation runs on MLX for the SenseNova-U1 model (sensenova_u1_8b[_fast]); other models have no in-process understanding path and stay on the Python torch path.",
            Some("epic 3180"),
        )),

        JobType::VideoGenerate => Err(classify_video_gap(&job.payload)),

        // Reached only for ineligible extend/bridge jobs (the eligible LTX IC-LoRA path + the Wan
        // TI2V-5B boundary-keyframe path early-return `Ok` via `job_is_any_mlx_eligible`). The
        // remaining gap is an engine with no in-context / keyframe path: the 14B Wan MoE engines
        // and any non-MLX video model (sc-3522 / sc-3357).
        JobType::VideoExtend | JobType::VideoBridge => Err(UnsupportedReason::new(
            model,
            "extend / bridge on this engine",
            "extend_clip / video_bridge run on MLX on the LTX IC-LoRA path (ltx_2_3 / ltx_2_3_eros) \
             and Wan TI2V-5B (wan_2_2, single-frame boundary keyframe conditioning); other engines \
             (the 14B Wan MoE) have no keyframe path, so they stay on the Python torch path.",
            Some("epic 3040"),
        )),

        // replace_person → native Wan-VACE (the replace-capable models) or native SCAIL-2
        // (scail2_14b, sc-5452) is MLX-eligible (handled by the early `job_is_any_mlx_eligible` Ok
        // above). This arm is only reached for a replace_person job on a model with no MLX video
        // engine — that stays torch.
        JobType::PersonReplace => Err(UnsupportedReason::new(
            model,
            "replace_person",
            "person replacement runs on native Wan-VACE (the replace-capable MLX video models) or native SCAIL-2 (scail2_14b); this model has no MLX video engine, so it stays on the Python torch path.",
            Some("epic 3040"),
        )),

        // Person detection + tracking are now ported to the Rust worker (epic 3482,
        // sc-3488): native-MLX YOLO11 detection (sc-3633), SORT/ByteTrack track assembly
        // (sc-3634), and SAM2 per-frame segmentation (sc-3709) all run in-process on the
        // macOS MLX worker, so the Replace-Person detect → track → mask flow is
        // Python-free. (replace_person end-to-end still needs the video-gen/inpaint half,
        // a tracked torch gap on `PersonReplace` below — epic 3040.)
        JobType::PersonDetect | JobType::PersonTrack => Ok(()),

        // DWPose pose detection is now ported to the Rust worker (sc-3487): RTMW
        // whole-body via `ort`/CoreML on the macOS MLX worker, so the Pose Library
        // "create from photo" flow + InstantID pose conditioning run Python-free.
        JobType::PoseDetect => Ok(()),

        // SCRFD 5-point landmark extraction is native-MLX on the Rust worker (sc-4433,
        // epic 4422): the same SCRFD detector the InstantID face stack already runs
        // in-process, so the Key Point Library "extract kps from this image" flow is
        // Python-free on Mac.
        JobType::KpsExtract => Ok(()),

        // Real-ESRGAN image upscaling is ported to the Rust worker (sc-3489) and SeedVR2 (the
        // native-MLX one-step diffusion upscaler, epic 4811 / sc-4815) runs in-process via
        // `mlx-gen-seedvr2`, so the upscale tool runs Python-free. The AuraSR engine (`aura-sr`,
        // a 617M-param torch-only GigaGAN) was DROPPED on Mac after the sc-3668 port-or-drop
        // spike (no viable Rust path; only a marginal, ~35-50x-slower quality difference vs
        // Real-ESRGAN x4). The Mac UI hides the AuraSR engine option, so this Err is now a
        // defensive submit-time guard; AuraSR stays available on Windows/Linux.
        JobType::ImageUpscale => {
            if upscale_job_is_mlx_eligible(job) {
                Ok(())
            } else {
                Err(UnsupportedReason::new(
                    model,
                    "image_upscale (AuraSR)",
                    "the Rust upscaler runs Real-ESRGAN; the AuraSR engine is dropped on Mac (available on Windows/Linux).",
                    Some("sc-3668"),
                ))
            }
        }

        // Video upscaling is net-new on Mac (epic 4811 / sc-4816): the native-MLX SeedVR2
        // engine is the only path (there is no torch video upscaler), so a SeedVR2 job is
        // supported and anything else has no in-process engine. Eligible jobs early-return
        // `Ok` above via `job_is_any_mlx_eligible`; this arm is the defensive guard.
        JobType::VideoUpscale => {
            if video_upscale_job_is_mlx_eligible(job) {
                Ok(())
            } else {
                Err(UnsupportedReason::new(
                    model,
                    "video_upscale (non-SeedVR2 engine)",
                    "video upscaling runs on the native-MLX SeedVR2 engine (seedvr2); no other engine is available.",
                    Some("epic 4811"),
                ))
            }
        }

        JobType::ModelConvert => classify_convert_gap(&job.payload),

        JobType::LoraTrain => Err(classify_training_gap(&job.payload)),

        JobType::TrainingCaption => Err(UnsupportedReason::new(
            None,
            "dataset captioning",
            "this dataset captioning job is not in the Rust/MLX JoyCaption flow.",
            Some("sc-3556"),
        )),
    }
}

/// The user-facing affordance prefix the Mac UI shows in place of a torch-only control
/// (sc-3486). Centralised so the API, the web client, and the gap docs read identically.
pub const MAC_NOT_AVAILABLE_LABEL: &str = "Not available on Mac (Rust/MLX only)";

/// UI-facing per-model macOS support (sc-3486), derived from the same `*_mlx_eligible` routing
/// predicates as the [`mac_rust_supported`] job oracle — one source of truth, so what the UI
/// hides can never drift from what routing refuses. `supported` = at least one generation config
/// for this model routes to the in-process Rust/MLX flow on macOS, so the model stays in the
/// picker; `false` = a torch-only model the Mac UI hides/disables once gating is active (its
/// `reason` names the porting epic). The per-feature flags use "available in *some* MLX config"
/// semantics (they never over-gate a valid combination) so a control is disabled only when the
/// model can't use it on MLX at all; residual config-specific dead ends are caught by the
/// `mlx_unsupported` affordance at submit.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelMacSupport {
    pub supported: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<UnsupportedReason>,
    pub features: ModelMacFeatures,
}

/// Per-feature macOS support for a model (sc-3486). Each flag mirrors the routing predicate for
/// that feature with "eligible in at least one config" semantics; `false` → disable that control
/// on Mac when gating is active. `video_modes` is populated only for video models.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelMacFeatures {
    /// Pose conditioning (the pose picker): a non-empty `advanced.poses`, alone or with a
    /// reference. Base `qwen_image` strict-pose uses the MLX ControlNet path (epic 3401).
    pub pose: bool,
    /// Reference / IP-Adapter / `character_image` identity conditioning (`referenceAssetId`).
    pub reference: bool,
    /// img2img `edit_image` (`mode=edit_image` + a source/reference image).
    pub edit: bool,
    /// Third-party LyCORIS (LoHa / non-peft LoKr) adapters — now applied on every MLX provider
    /// (epic 3641: core loader sc-3642/3643, SDXL/Wan/LTX sc-3671), so `true` for MLX-routed models.
    pub lycoris: bool,
    /// Video-only: which `video_generate` modes route to MLX. Empty for non-video models.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub video_modes: BTreeMap<String, bool>,
}

/// Build a synthetic generation payload (`{ "model": ..., <entries> }`) for probing the routing
/// predicates without a full [`JobSnapshot`] — the UI-gating sibling of how the oracle reads a
/// real job's payload.
fn probe_payload(model: &str, entries: &[(&str, Value)]) -> Map<String, Value> {
    let mut payload = Map::new();
    payload.insert("model".to_owned(), Value::String(model.to_owned()));
    for (key, value) in entries {
        payload.insert((*key).to_owned(), value.clone());
    }
    payload
}

/// UI gating support for a model id of the given catalog `model_type` ("image" / "video" / other).
/// Non-image/video types (utility/infra: upscalers, captioners) are reported `supported` — their
/// Python-only *actions* are gated by [`mac_capabilities`] at the job-type level, not by hiding
/// the model from a picker. Same source of truth as [`mac_rust_supported`].
pub fn model_mac_support(model_id: &str, model_type: &str) -> ModelMacSupport {
    match model_type {
        "image" => image_model_mac_support(model_id),
        "video" => video_model_mac_support(model_id),
        _ => ModelMacSupport {
            supported: true,
            reason: None,
            features: ModelMacFeatures::default(),
        },
    }
}

fn image_model_mac_support(model: &str) -> ModelMacSupport {
    if !MLX_ROUTED_MODELS.contains(&model) {
        return ModelMacSupport {
            supported: false,
            reason: Some(classify_image_gap(&probe_payload(model, &[]))),
            features: ModelMacFeatures::default(),
        };
    }
    // "Available in some MLX config" probes — bias toward not-disabling so a valid combination
    // (e.g. a Z-Image reference, with or without a pose set — sc-3619) is never blocked. Any
    // residual config-only dead ends surface as the `mlx_unsupported` submit affordance.
    let pose = image_request_mlx_eligible(
        model,
        &probe_payload(model, &[("advanced", json!({ "poses": [{}] }))]),
    ) || image_request_mlx_eligible(
        model,
        &probe_payload(
            model,
            &[
                ("mode", json!("character_image")),
                ("referenceAssetId", json!("probe")),
                ("advanced", json!({ "poses": [{}] })),
            ],
        ),
    );
    let reference = image_request_mlx_eligible(
        model,
        &probe_payload(model, &[("referenceAssetId", json!("probe"))]),
    ) || image_request_mlx_eligible(
        model,
        &probe_payload(
            model,
            &[
                ("mode", json!("character_image")),
                ("referenceAssetId", json!("probe")),
            ],
        ),
    ) || image_request_mlx_eligible(
        model,
        &probe_payload(
            model,
            &[
                ("referenceAssetId", json!("probe")),
                ("advanced", json!({ "poses": [{}] })),
            ],
        ),
    );
    let edit = image_request_mlx_eligible(
        model,
        &probe_payload(
            model,
            &[
                ("mode", json!("edit_image")),
                ("sourceAssetId", json!("probe")),
            ],
        ),
    );
    ModelMacSupport {
        supported: true,
        reason: None,
        features: ModelMacFeatures {
            pose,
            reference,
            edit,
            // Third-party LyCORIS applies on every MLX provider now (epic 3641).
            lycoris: true,
            video_modes: BTreeMap::new(),
        },
    }
}

/// The `video_generate` modes the UI offers, in display order, so the gating mirrors
/// [`video_mode_is_mlx_eligible`] for every mode a Mac user could pick. The clip-conditioning
/// modes `extend_clip` / `video_bridge` are included (sc-3773) so the Mac UI gates them
/// per-model — MLX on the LTX IC-LoRA path, torch on Wan — rather than via a coarse global flag.
const VIDEO_UI_MODES: &[&str] = &[
    "text_to_video",
    "image_to_video",
    "first_last_frame",
    "extend_clip",
    "video_bridge",
    "replace_person",
    // Bernini editing / reference-driven video modes (sc-4703) + multi-source modes
    // (sc-5425: `multi_video_to_video` / `ads2v`): only `bernini` is eligible (see
    // `video_mode_is_mlx_eligible`); they surface disabled on the other models, the same
    // per-model gating as `replace_person` / the LTX clip modes.
    "video_to_video",
    "reference_to_video",
    "reference_video_to_video",
    "multi_video_to_video",
    "ads2v",
    // SCAIL-2 standalone character animation (epic 5439 / sc-5448): only `scail2_14b` is
    // eligible; surfaces disabled on the other models. Reference character + driving video
    // → animated clip. (Cross-identity replacement reuses `replace_person`, wired in sc-5452.)
    "animate_character",
];

fn video_model_mac_support(model: &str) -> ModelMacSupport {
    if !VIDEO_MLX_ROUTED_MODELS.contains(&model) {
        return ModelMacSupport {
            supported: false,
            reason: Some(classify_video_gap(&probe_payload(model, &[]))),
            features: ModelMacFeatures::default(),
        };
    }
    let video_modes = VIDEO_UI_MODES
        .iter()
        .map(|mode| ((*mode).to_owned(), video_mode_is_mlx_eligible(model, mode)))
        .collect();
    ModelMacSupport {
        supported: true,
        reason: None,
        features: ModelMacFeatures {
            video_modes,
            ..ModelMacFeatures::default()
        },
    }
}

/// macOS support for a non-model feature/sub-system (sc-3486): the infra job types that have no
/// in-process Rust path. `supported=false` carries the `reason` (the same `UnsupportedReason` the
/// `mlx_unsupported` event uses); when one of these is ported its flag flips to `true`.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MacFeatureSupport {
    pub supported: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<UnsupportedReason>,
}

impl MacFeatureSupport {
    // Declares a Mac feature gap with the reason + suggested port epic. Currently no
    // feature is gated (poseFromPhoto was the last, ported in sc-3487/flipped in
    // sc-4206) — kept as the gating vocabulary for the next torch-only surface that
    // appears before its Rust port lands, so a gap is declared the same way every time.
    #[allow(dead_code)]
    fn unsupported(feature: &str, detail: &str, suggested_epic: &str) -> Self {
        Self {
            supported: false,
            reason: Some(UnsupportedReason::new(
                None,
                feature,
                detail,
                Some(suggested_epic),
            )),
        }
    }
}

/// macOS training support (sc-3486): the kernels with a native mlx-gen Rust trainer, so the
/// Training studio can disable a base model whose kernel only runs on the Python torch trainer.
/// `lokr_on_wan_supported=false` mirrors the LoKr-on-Wan routing caveat.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MacTrainingSupport {
    pub supported_kernels: Vec<String>,
    pub lokr_on_wan_supported: bool,
}

/// What the Mac UI needs to gate every non-model Python surface plus the master switch
/// (sc-3486). `mac_gating_active` is the rollout flag (`SCENEWORKS_MLX_REQUIRED`): when `false`
/// (Windows/Linux, or a Mac still in observe mode) the client applies no gating at all, so
/// non-Mac pickers are untouched. The per-feature entries are facts about the Rust flow
/// independent of the flag; the client only acts on them when `mac_gating_active`.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MacCapabilities {
    pub platform: String,
    pub mac_gating_active: bool,
    pub not_available_label: String,
    pub features: BTreeMap<String, MacFeatureSupport>,
    pub training: MacTrainingSupport,
}

/// Build the [`MacCapabilities`] surface for the given platform + gating flag. The feature set is
/// the non-model half of `docs/mac-rust-gaps.md` §5 (infra) plus the global feature gaps; keep it
/// in sync with the oracle's job-type arms.
pub fn mac_capabilities(platform: &str, mac_gating_active: bool) -> MacCapabilities {
    // `std::env::consts::OS` is `"macos"` (the API host's OS, passed by the capabilities handler);
    // accept the legacy `"darwin"` alias defensively. Drives the platform-intrinsic engine flags
    // (e.g. `imageUpscaleSeedvr2`, which is Mac-only) rather than the gating-rollout flag.
    let is_mac = matches!(platform, "macos" | "darwin");
    let mut features = BTreeMap::new();
    // Third-party LyCORIS (LoHa / non-peft LoKr) now applies on every MLX provider (epic 3641:
    // core loader sc-3642/3643 + SDXL/Wan/LTX sc-3671), so it is no longer a Mac feature gap — the
    // per-model `features.lycoris` flag is `true` and the web LyCORIS upload control is un-gated.
    features.insert(
        // Real-ESRGAN image upscaling is ported to the Rust worker (sc-3489), so the
        // Image Editor upscale tool works on a Python-free Mac. The tool stays available;
        // only the second engine (AuraSR) is dropped, gated per-engine below.
        "imageUpscale".to_owned(),
        MacFeatureSupport {
            supported: true,
            reason: None,
        },
    );
    features.insert(
        // The AuraSR upscale engine (`engine=aura-sr`) is dropped on Mac (sc-3668,
        // port-or-drop spike): it is a 617M-param torch-only GigaGAN with no viable Rust
        // path and only a marginal, ~35-50x-slower quality difference vs the already-ported
        // Real-ESRGAN x4. The Mac UI hides this engine option so a user never reaches the
        // `mlx_unsupported` error; it stays available on Windows/Linux. This must agree with
        // the AuraSR arm of `mac_rust_supported` (what the UI hides == what routing refuses).
        "imageUpscaleAuraSr".to_owned(),
        MacFeatureSupport {
            supported: false,
            reason: Some(UnsupportedReason::new(
                None,
                "image_upscale (AuraSR)",
                "AuraSR is a torch-only GAN upscaler, dropped on Mac; Real-ESRGAN x4 is the Mac upscaler (it stays available on Windows/Linux).",
                Some("sc-3668"),
            )),
        },
    );
    features.insert(
        // SeedVR2 (`engine=seedvr2`) is the native-MLX one-step diffusion upscaler (epic 4811 /
        // sc-4815) — the INVERSE of AuraSR: it is supported on Mac (in-process `mlx-gen-seedvr2`)
        // and NOT yet available on Windows/Linux, where the backend is a separate Candle port
        // (sc-5157). This flag is platform-intrinsic (true only on Mac, regardless of the gating
        // rollout flag) so the web upscale picker can offer SeedVR2 on Mac and hide it elsewhere —
        // contrast the other entries here, which describe Mac torch-only gaps the UI hides only
        // under active gating.
        "imageUpscaleSeedvr2".to_owned(),
        MacFeatureSupport {
            supported: is_mac,
            reason: if is_mac {
                None
            } else {
                Some(UnsupportedReason::new(
                    None,
                    "image_upscale (SeedVR2)",
                    "SeedVR2 is a Mac-only native-MLX upscaler; Windows/Linux support is a separate Candle backend port.",
                    Some("sc-5157"),
                ))
            },
        },
    );
    features.insert(
        // DWPose pose detection is ported to the Rust worker (sc-3487): RTMW whole-body
        // via `ort`/CoreML on the macOS MLX worker, so the Pose Library "create from
        // photo" flow runs Python-free. This must agree with the PoseDetect arm of
        // `mac_rust_supported` — what the UI hides can never drift from what routing
        // refuses (sc-4206 / F-CORE-2).
        "poseFromPhoto".to_owned(),
        MacFeatureSupport {
            supported: true,
            reason: None,
        },
    );
    features.insert(
        // Person detection + tracking are ported to the Rust worker (sc-3488 /
        // sc-3633/3634/3709): native-MLX YOLO11 detection, SORT/ByteTrack track assembly,
        // and SAM2 per-frame segmentation all run in-process, so the Replace-Person
        // detect → track → mask flow works on a Python-free Mac. (The replace_person
        // video-gen half is gated per-model via each video model's `videoModes`.)
        "personDetect".to_owned(),
        MacFeatureSupport {
            supported: true,
            reason: None,
        },
    );
    features.insert(
        "datasetCaptioning".to_owned(),
        MacFeatureSupport {
            supported: true,
            reason: None,
        },
    );
    features.insert(
        // Video upscaling is net-new on Mac (epic 4811 / sc-4816): the native-MLX SeedVR2
        // engine gives SceneWorks its first video upscaler, running in-process on the macOS
        // MLX worker (zero-Python). There is no torch fallback (mac-only), so this feature is
        // the gate for the Video Studio "Upscale" action. Must agree with the VideoUpscale arm
        // of `mac_rust_supported` (what the UI shows == what routing accepts).
        "videoUpscale".to_owned(),
        MacFeatureSupport {
            supported: true,
            reason: None,
        },
    );
    // The former global `advancedVideoModes` flag is gone (sc-3773): every video mode — including
    // the LTX IC-LoRA clip-conditioning modes extend_clip / video_bridge — is now gated per-model
    // via each model's `macSupport.features.videoModes`, so a Mac user on LTX is no longer blocked
    // from a mode the in-process Rust worker can run.
    MacCapabilities {
        platform: platform.to_owned(),
        mac_gating_active,
        not_available_label: MAC_NOT_AVAILABLE_LABEL.to_owned(),
        features,
        training: MacTrainingSupport {
            supported_kernels: MLX_ROUTED_TRAINING_KERNELS
                .iter()
                .map(|kernel| (*kernel).to_owned())
                .collect(),
            lokr_on_wan_supported: false,
        },
    }
}

/// The dedicated MLX-porting epic for a torch-only image model (epic 3482 policy: every
/// unported model gets its own port epic + is dropped on Mac until it lands). `None` = a
/// model we don't have a port epic for yet, which the oracle reports as "needs an epic".
/// Keep in sync with `docs/mac-rust-gaps.md` §1.
///
/// **No whole-model torch-only image families remain.** Each was ported to MLX and moved into
/// `MLX_ROUTED_MODELS`, so it never reaches this classifier: Kolors (epic 3090 / sc-3875), InstantID
/// (epic 3109 / sc-3345), PuLID-FLUX (epic 3069 / sc-3344), z_image_edit (epic 3529 / sc-3923),
/// Chroma (epic 3531 / sc-3843), SenseNova-U1 (epic 3180 / sc-3900), and finally Lens / Lens-Turbo
/// (epic 3164 / sc-5105 — the LAST one). Models with a partial surface (e.g. InstantID pose-library,
/// PuLID reference-less) are named per-feature in `classify_image_gap`, not here. This function is
/// retained for the generic "unported model → needs a port epic" path and as the seam for any future
/// torch-only image model: add a `match _model { "<id>" => Some("epic NNNN"), _ => None }` arm here.
fn torch_only_image_model_epic(_model: &str) -> Option<&'static str> {
    None
}

/// Name the precise gap for an ineligible `image_generate` / `image_edit` job: a torch-only
/// model, or a torch-only feature on an otherwise-MLX family. Mirrors the per-family
/// `*_mlx_eligible` gates so the reason matches why routing refused it.
fn classify_image_gap(payload: &Map<String, Value>) -> UnsupportedReason {
    let Some(model) = payload.get("model").and_then(Value::as_str) else {
        return UnsupportedReason::new(None, "image generation", "no model specified.", None);
    };
    if !MLX_ROUTED_MODELS.contains(&model) {
        let epic = torch_only_image_model_epic(model);
        let detail = if epic.is_some() {
            "this model has no Rust/MLX engine yet; it runs on the Python torch path and is dropped on Mac until its port epic lands."
        } else {
            "this model has no Rust/MLX engine and no port epic yet — file a porting epic and drop it on Mac (epic 3482 policy)."
        };
        return UnsupportedReason::new(Some(model), "torch-only image model", detail, epic);
    }
    // Third-party LyCORIS (LoHa / non-peft LoKr) now applies on every MLX provider (epic 3641,
    // sc-3642/3643/3671), so it is no longer an image gap.
    match model {
        "qwen_image" => UnsupportedReason::new(
            Some(model),
            "reference / edit conditioning",
            "base Qwen-Image reference / edit_image conditioning stays on the Python torch path unless it is the strict-pose ControlNet tier.",
            Some("epic 3401"),
        ),
        "flux_schnell" | "flux_dev" => UnsupportedReason::new(
            Some(model),
            "reference (XLabs IP-Adapter)",
            "FLUX.1 reference is the XLabs IP-Adapter (not img2img-init); it stays on the Python torch path until the MLX port lands. (FLUX.1 edit_image has no torch path on any platform — a future Kontext capability, not an eradication gap; see sc-3535.)",
            Some("epic 3621"),
        ),
        "qwen_image_edit"
        | "qwen_image_edit_2509"
        | "qwen_image_edit_2511"
        | "qwen_image_edit_2511_lightning" => UnsupportedReason::new(
            Some(model),
            "edit without a reference/source image",
            "the Qwen-Image-Edit model needs edit_image+sourceAssetId or character_image+referenceAssetId to route to MLX.",
            None,
        ),
        "sensenova_u1_8b" | "sensenova_u1_8b_fast" => {
            let has_poses = payload
                .get("advanced")
                .and_then(Value::as_object)
                .and_then(|advanced| advanced.get("poses"))
                .and_then(Value::as_array)
                .is_some_and(|poses| !poses.is_empty());
            if has_poses {
                UnsupportedReason::new(
                    Some(model),
                    "strict pose (ControlNet)",
                    "SenseNova-U1 has no ControlNet/skeleton conditioning — the strict-pose tier is not an MLX path; it stays on the Python torch path (dropped on Mac).",
                    Some("epic 3180"),
                )
            } else {
                UnsupportedReason::new(
                    Some(model),
                    "edit/character without a reference",
                    "SenseNova-U1 edit needs edit_image+sourceAssetId, and Character Studio needs character_image+referenceAssetId, to route to MLX.",
                    None,
                )
            }
        }
        // InstantID (sc-3345 identity + angle set; sc-3381 pose mode + face-restore): the full
        // surface runs on MLX for `character_image` + `referenceAssetId`. Only a non-character /
        // reference-less job has no InstantID path. Mirrors `instantid_mlx_eligible`.
        "instantid_realvisxl" => UnsupportedReason::new(
            Some(model),
            "InstantID without a character reference",
            "InstantID runs on MLX for character_image with a referenceAssetId (single identity, the 11-view angle set, pose-library mode, and face-restore); a non-character / reference-less job has no InstantID path.",
            None,
        ),
        // PuLID-FLUX (sc-3344): runs on MLX only for character_image with a referenceAssetId (the
        // face it injects). A non-character / reference-less job has no PuLID path. Mirrors
        // `pulid_flux_mlx_eligible`.
        "pulid_flux_dev" => UnsupportedReason::new(
            Some(model),
            "PuLID-FLUX without a character reference",
            "PuLID-FLUX runs on MLX for character_image with a referenceAssetId (the reference face drives the identity injection); a non-character / reference-less job has no PuLID-FLUX path.",
            None,
        ),
        // Kolors (epic 3090) runs its full surface on MLX now — T2I (sc-3875), img2img (sc-4765),
        // the IP-Adapter-Plus reference (sc-4767) and the strict-pose tier (sc-4766 / engine sc-5012)
        // — so a kolors job is never gap-classified; any residual falls to the defensive arm below.
        // flux2 / sdxl / realvisxl only fall out via LyCORIS (handled above) — defensive.
        _ => UnsupportedReason::new(
            Some(model),
            "unsupported configuration",
            "this model/feature combination is not in the Rust/MLX flow.",
            None,
        ),
    }
}

/// Name the precise gap for an ineligible `video_generate` job: a torch-only model (incl. SVD) or
/// an advanced mode. Mirrors `video_job_is_mlx_eligible`. (Third-party LyCORIS and LoKr-on-Wan now
/// apply on the MLX Wan/LTX paths — epic 3641 sc-3671 — so neither is a video gap anymore.)
fn classify_video_gap(payload: &Map<String, Value>) -> UnsupportedReason {
    let Some(model) = payload.get("model").and_then(Value::as_str) else {
        return UnsupportedReason::new(None, "video generation", "no model specified.", None);
    };
    if !VIDEO_MLX_ROUTED_MODELS.contains(&model) {
        return UnsupportedReason::new(
            Some(model),
            "torch-only video model",
            "this video model has no Rust/MLX engine; it runs on the Python torch path.",
            Some("epic 3040"),
        );
    }
    let mode = payload
        .get("mode")
        .and_then(Value::as_str)
        .unwrap_or("image_to_video");
    if !video_mode_is_mlx_eligible(model, mode) {
        return UnsupportedReason::new(
            Some(model),
            "advanced video mode",
            "this video_generate mode is not MLX-eligible on this model (first_last_frame / \
             extend_clip / video_bridge / replace_person route to MLX only on the capable engines — \
             LTX + Wan TI2V-5B for the keyframe/clip modes, Wan-VACE for replace_person).",
            Some("epic 3040"),
        );
    }
    UnsupportedReason::new(
        Some(model),
        "unsupported video configuration",
        "this video configuration is not in the Rust/MLX flow.",
        None,
    )
}

/// Name the precise gap for an ineligible `lora_train` job. Mirrors `training_job_is_mlx_eligible`:
/// a kernel with no native mlx-gen Rust trainer, or LoKr-on-Wan.
fn classify_training_gap(payload: &Map<String, Value>) -> UnsupportedReason {
    let kernel = payload
        .get("plan")
        .and_then(Value::as_object)
        .and_then(|plan| plan.get("target"))
        .and_then(Value::as_object)
        .and_then(|target| target.get("kernel"))
        .and_then(Value::as_str);
    match kernel {
        // `kolors_lora` (sc-4568/sc-4732) and `lens_lora` (sc-5148/sc-5180) are no longer gaps —
        // both have native mlx-gen Rust trainers and route to the mlx worker, so they never reach
        // this classifier.
        Some("wan_lora") | Some("wan_moe_lora") => UnsupportedReason::new(
            None,
            "LoKr-on-Wan training",
            "Wan LoKr training stays on torch (no Kronecker merge in the mlx Wan path).",
            Some("epic 3039"),
        ),
        _ => UnsupportedReason::new(
            None,
            "LoRA/LoKr training",
            "this training kernel has no native mlx-gen Rust trainer.",
            Some("epic 3039"),
        ),
    }
}

/// `model_convert` is supported only for the in-process Rust FLUX.2-klein converter
/// (`flux2_klein_diffusers`, sc-3136). The default/absent converter is the Python mlx-video
/// Wan/LTX path (sc-3491 / sc-3224).
fn classify_convert_gap(payload: &Map<String, Value>) -> Result<(), UnsupportedReason> {
    if payload.get("converter").and_then(Value::as_str) == Some("flux2_klein_diffusers") {
        return Ok(());
    }
    Err(UnsupportedReason::new(
        payload.get("model").and_then(Value::as_str),
        "Wan/LTX model conversion (mlx_video)",
        "installing a non-turnkey Wan/LTX checkpoint converts via the Python mlx_video path.",
        Some("sc-3491 / sc-3224"),
    ))
}

/// Classify a *successful* claim for routing observability. `None` means the claim was
/// routing-neutral (the job is not MLX-eligible, so no `mlx` worker would have wanted
/// it). When a non-`mlx` worker claims an MLX-eligible job, the reason distinguishes a
/// user pinning a specific GPU (`explicit_gpu`) from the case the team cares about — no
/// idle `mlx` worker was available to take it (`fell_back_to_torch`/`no_idle_mlx_worker`).
/// The deferral path (a non-mlx worker yielding to an idle mlx worker) is reported
/// separately inside `claim_next_job_routed` as `deferred_to_mlx`.
fn route_decision_for_claim(
    job: &JobSnapshot,
    gpu_id: &str,
    worker_id: &str,
) -> Option<RouteDecision> {
    if !job_is_any_mlx_eligible(job) {
        return None;
    }
    if gpu_id.eq_ignore_ascii_case("mlx") {
        return Some(RouteDecision::new(
            job,
            gpu_id,
            worker_id,
            "claimed_by_mlx",
            "mlx_worker",
        ));
    }
    if job.requested_gpu == "auto" {
        Some(RouteDecision::new(
            job,
            gpu_id,
            worker_id,
            "fell_back_to_torch",
            "no_idle_mlx_worker",
        ))
    } else {
        Some(RouteDecision::new(
            job,
            gpu_id,
            worker_id,
            "explicit_gpu",
            "explicit_gpu",
        ))
    }
}

fn should_defer_auto_gpu_claim(
    connection: &Connection,
    job: &JobSnapshot,
    worker: &WorkerSnapshot,
) -> JobsStoreResult<bool> {
    if job.requested_gpu != "auto"
        || is_non_gpu_job_type(job.job_type.as_str())
        || worker.gpu_id == "cpu"
    {
        return Ok(false);
    }
    // The in-process `mlx` worker is the designated home for the jobs it claims
    // (a non-mlx worker defers MLX-eligible jobs to it via
    // `should_defer_image_to_mlx_worker` & siblings). It must never hand one of
    // those jobs to a "healthier" non-mlx GPU through this health-based dispatch:
    // on Apple Silicon the `mlx` and `mps` workers share the same physical GPU,
    // and that worker would only defer the job straight back, deadlocking it in
    // the queue. Keeping the mlx worker out of the auto-GPU health comparison
    // breaks that cycle regardless of whether it reports utilization.
    if worker.gpu_id.eq_ignore_ascii_case("mlx") {
        return Ok(false);
    }
    let current_score = dispatch_score(job, worker);
    if !current_score.has_utilization {
        return Ok(false);
    }

    let mut statement = connection.prepare(
        "
        select * from workers
         where id != ?1
           and gpu_id != 'cpu'
           and status = 'idle'
         order by gpu_id, id
        ",
    )?;
    let candidates = collect_workers(statement.query_map(params![worker.id], row_to_worker)?)?;
    // Cache the active-GPU-job fact per gpu_id so two idle workers sharing a GPU
    // don't each re-run the same `active_gpu_job_exists` query (sc-4273).
    let mut active_by_gpu: std::collections::HashMap<String, bool> =
        std::collections::HashMap::new();
    for candidate in candidates {
        if !worker_supports_job(&candidate, job) {
            continue;
        }
        let gpu_busy = match active_by_gpu.get(&candidate.gpu_id) {
            Some(&busy) => busy,
            None => {
                let busy = active_gpu_job_exists(connection, &candidate.gpu_id)?;
                active_by_gpu.insert(candidate.gpu_id.clone(), busy);
                busy
            }
        };
        if gpu_busy {
            continue;
        }
        let candidate_score = dispatch_score(job, &candidate);
        if dispatch_score_is_better(candidate_score, current_score) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Epic 3018 routing — prefer the in-process MLX worker for MLX-eligible image
/// jobs. A non-mlx GPU worker defers an `auto` `image_generate` job the mlx
/// worker can run when an idle `mlx` worker exists, so the fast NAX path claims
/// it. When no mlx worker is registered (Windows/Linux, or the mlx worker is
/// down), nothing defers and the torch worker is the fallback — a job is never
/// stuck. An explicit (non-`auto`) GPU choice is always honoured, never deferred.
fn should_defer_image_to_mlx_worker(
    connection: &Connection,
    job: &JobSnapshot,
    worker: &WorkerSnapshot,
    mlx_required: bool,
) -> JobsStoreResult<bool> {
    if worker.gpu_id.eq_ignore_ascii_case("mlx") || !job_is_mlx_eligible(job) {
        return Ok(false);
    }
    // macOS "MLX-required" (epic 3482 / sc-3483): the non-mlx (MPS) worker NEVER claims
    // an MLX-eligible job — it yields unconditionally, even when no idle `mlx` worker is
    // ready *right now*. The job waits for the `mlx` worker and, if none takes it within
    // the grace window, `fail_stranded_mlx_jobs` fails it terminal with `mlx_unavailable`
    // rather than letting MPS silently run it. This covers explicit-GPU pins too: "never
    // MPS" is absolute on Mac.
    if mlx_required {
        return Ok(true);
    }
    // Off (Windows/Linux/Docker, and Mac pre-cutover): unchanged — defer only an `auto`
    // job to an actually-idle `mlx` worker; otherwise the torch worker is the fallback and
    // an explicit (non-`auto`) GPU choice is always honoured.
    if job.requested_gpu != "auto" {
        return Ok(false);
    }
    idle_mlx_worker_can_claim(connection, job, worker)
}

/// Video sibling of [`should_defer_image_to_mlx_worker`] (sc-3036): a non-mlx GPU
/// worker defers an `auto` MLX-eligible `video_generate` job when an idle `mlx`
/// worker can run it. Same fallback guarantees — no mlx worker / explicit GPU →
/// never deferred.
fn should_defer_video_to_mlx_worker(
    connection: &Connection,
    job: &JobSnapshot,
    worker: &WorkerSnapshot,
    mlx_required: bool,
) -> JobsStoreResult<bool> {
    if worker.gpu_id.eq_ignore_ascii_case("mlx") || !video_job_is_mlx_eligible(job) {
        return Ok(false);
    }
    // macOS MLX-required (sc-3483): yield unconditionally, same as the image sibling.
    if mlx_required {
        return Ok(true);
    }
    if job.requested_gpu != "auto" {
        return Ok(false);
    }
    idle_mlx_worker_can_claim(connection, job, worker)
}

/// Training sibling of [`should_defer_image_to_mlx_worker`] (epic 3039): a non-mlx
/// GPU worker defers an `auto` MLX-eligible `lora_train` job when an idle `mlx`
/// worker can run it, so the native Rust trainer (`mlx_gen::load_trainer`) claims
/// it. Same fallback guarantees — no mlx worker registered (Windows/Linux, or the
/// mlx worker is down) → nothing defers and the Python torch trainer runs it; an
/// explicit (non-`auto`) GPU choice is always honoured. The torch trainers stay
/// the cross-platform path + the Mac fallback (sc-3049), so a job is never stuck.
fn should_defer_training_to_mlx_worker(
    connection: &Connection,
    job: &JobSnapshot,
    worker: &WorkerSnapshot,
    mlx_required: bool,
) -> JobsStoreResult<bool> {
    if worker.gpu_id.eq_ignore_ascii_case("mlx") || !training_job_is_mlx_eligible(job) {
        return Ok(false);
    }
    // macOS MLX-required (sc-3483): yield unconditionally, same as the image sibling.
    if mlx_required {
        return Ok(true);
    }
    if job.requested_gpu != "auto" {
        return Ok(false);
    }
    idle_mlx_worker_can_claim(connection, job, worker)
}

/// Captioning sibling of [`should_defer_image_to_mlx_worker`] (sc-3556): a non-mlx
/// GPU worker defers JoyCaption dataset-caption jobs to an idle mlx worker, so the
/// native Rust captioner (`mlx_gen::load_captioner`) can run them. Windows/Linux and
/// explicit non-auto GPU requests keep the existing Python torch captioner fallback.
fn should_defer_caption_to_mlx_worker(
    connection: &Connection,
    job: &JobSnapshot,
    worker: &WorkerSnapshot,
    mlx_required: bool,
) -> JobsStoreResult<bool> {
    if worker.gpu_id.eq_ignore_ascii_case("mlx") || !caption_job_is_mlx_eligible(job) {
        return Ok(false);
    }
    if mlx_required {
        return Ok(true);
    }
    if job.requested_gpu != "auto" {
        return Ok(false);
    }
    idle_mlx_worker_can_claim(connection, job, worker)
}

/// Understanding sibling of [`should_defer_image_to_mlx_worker`] (sc-3905): a non-mlx GPU worker
/// defers an `auto` MLX-eligible SenseNova-U1 `image_vqa` / `image_interleave` job to an idle mlx
/// worker, so the in-process `T2iModel` (`vqa` / `interleave_gen`) claims it. Windows/Linux and
/// explicit non-auto GPU requests keep the Python torch SenseNova path.
fn should_defer_understanding_to_mlx_worker(
    connection: &Connection,
    job: &JobSnapshot,
    worker: &WorkerSnapshot,
    mlx_required: bool,
) -> JobsStoreResult<bool> {
    if worker.gpu_id.eq_ignore_ascii_case("mlx") || !understanding_job_is_mlx_eligible(job) {
        return Ok(false);
    }
    // macOS MLX-required (sc-3483): yield unconditionally, same as the image sibling.
    if mlx_required {
        return Ok(true);
    }
    if job.requested_gpu != "auto" {
        return Ok(false);
    }
    idle_mlx_worker_can_claim(connection, job, worker)
}

/// Whether an idle `mlx` worker (other than `worker`) exists that supports `job`
/// and has no active GPU job — the shared tail of the image/video MLX deferral.
fn idle_mlx_worker_can_claim(
    connection: &Connection,
    job: &JobSnapshot,
    worker: &WorkerSnapshot,
) -> JobsStoreResult<bool> {
    let mut statement = connection.prepare(
        "
        select * from workers
         where id != ?1
           and gpu_id = 'mlx'
           and status = 'idle'
         order by id
        ",
    )?;
    let candidates = collect_workers(statement.query_map(params![worker.id], row_to_worker)?)?;
    // Every candidate here has `gpu_id = 'mlx'`, so the active-GPU-job fact is
    // identical for all of them — resolve a supporting candidate first, then run
    // `active_gpu_job_exists` once instead of once per candidate (sc-4273).
    let Some(candidate) = candidates.iter().find(|c| worker_supports_job(c, job)) else {
        return Ok(false);
    };
    Ok(!active_gpu_job_exists(connection, &candidate.gpu_id)?)
}

fn active_gpu_job_exists(connection: &Connection, gpu_id: &str) -> JobsStoreResult<bool> {
    if is_apple_unified_gpu_id(gpu_id) {
        return Ok(connection
            .query_row(
                &format!(
                    "
            select id from jobs
             where lower(assigned_gpu) in ('mlx', 'mps')
               and status in ({active})
               and type not in ({})
             limit 1
            ",
                    non_gpu_job_types_sql(),
                    active = active_statuses_sql()
                ),
                [],
                |_row| Ok(()),
            )
            .optional()?
            .is_some());
    }
    Ok(connection
        .query_row(
            &format!(
                "
            select id from jobs
             where assigned_gpu = ?1
               and status in ({active})
               and type not in ({})
             limit 1
            ",
                non_gpu_job_types_sql(),
                active = active_statuses_sql()
            ),
            params![gpu_id],
            |_row| Ok(()),
        )
        .optional()?
        .is_some())
}

fn is_apple_unified_gpu_id(gpu_id: &str) -> bool {
    gpu_id.eq_ignore_ascii_case("mlx") || gpu_id.eq_ignore_ascii_case("mps")
}

/// Models the in-process Rust MLX worker generates today, by id. This set grows
/// one family story at a time as each lands real generation in
/// `sceneworks-worker::image_jobs` — sc-3022 Z-Image, sc-3023 FLUX.1, sc-3024 Qwen,
/// sc-3025 FLUX.2, sc-3026 SDXL (live). A model id absent here is never routed to the
/// mlx worker, so the Python torch path stays authoritative for it.
const MLX_ROUTED_MODELS: &[&str] = &[
    "z_image_turbo",
    "z_image_edit",
    "flux_schnell",
    "flux_dev",
    "qwen_image",
    "qwen_image_edit",
    "qwen_image_edit_2509",
    "qwen_image_edit_2511",
    "qwen_image_edit_2511_lightning",
    "flux2_klein_9b",
    "flux2_klein_9b_kv",
    "flux2_klein_9b_true_v2",
    "sdxl",
    "realvisxl",
    // InstantID on RealVisXL (sc-3345): single-identity + the 11-view angle set route to the
    // native `mlx-gen-instantid` provider. Pose-library + face-restore InstantID jobs are gated
    // OUT by `instantid_mlx_eligible` and stay on the torch `InstantIDAdapter` (engine sc-3117 /
    // sc-3380 not ported).
    "instantid_realvisxl",
    // PuLID-FLUX on FLUX.1-dev (sc-3344): the native `mlx-gen-pulid` registry generator serves
    // `character_image` with a reference face. Mirrors `pulid_flux_mlx_eligible`.
    "pulid_flux_dev",
    "chroma1_hd",
    "chroma1_base",
    "chroma1_flash",
    "sensenova_u1_8b",
    "sensenova_u1_8b_fast",
    // Kolors (epic 3090): the full surface runs on the Rust `kolors` engine model — T2I (sc-3875),
    // img2img (sc-4765), the IP-Adapter-Plus reference (sc-4767) and the strict-pose tier (sc-4766 /
    // engine sc-5012, the combined pose-ControlNet + IP-Adapter-identity + img2img pass).
    "kolors",
    // Microsoft Lens / Lens-Turbo (epic 3164 engine / sc-5105 cutover): pure T2I on the native
    // `mlx-gen-lens` engine (gpt-oss-20b MoE encoder + dual-stream MMDiT + Flux.2 VAE), retiring the
    // Python `/opt/lens-venv` transformers-5 sidecar on Mac. Both ids are always MLX-eligible
    // (`lens_mlx_eligible` — no conditioning surface to gate). Lens was the LAST whole-model
    // torch-only image family; with it routed, every image model here is MLX (`torch_only_image_model_epic`
    // now matches nothing).
    "lens",
    "lens_turbo",
    // Bernini still-image companion (epic 4699 / sc-5424): the image-typed catalog id
    // (`bernini_image`) routes its t2i / i2i (`edit_image`) jobs to the in-process Rust
    // worker, where the same `engine_id:"bernini"` planner+renderer runs with `frames:1`.
    // The video `bernini` id lives in `VIDEO_MLX_ROUTED_MODELS`, not here. Mirrors
    // `bernini_image_mlx_eligible`.
    "bernini_image",
];

/// Epic 3018 routing — does this image job belong on the in-process Rust MLX
/// worker (vs the Python torch worker)? This lifts the per-family Python
/// `_should_route_*_to_mlx` decision (apps/worker/scene_worker/image_adapters.py)
/// up to the API claim layer, minus the worker-local gates (platform / disable
/// env / sidecar presence) — those are now expressed by whether an `mlx` worker
/// is registered and idle (see `should_defer_image_to_mlx_worker`).
///
/// Routing-layer caveat: LyCORIS detection uses only the LoRA's *recorded*
/// `networkType`. The Python predicate also sniffs the safetensors header, but
/// the API has no access to the LoRA files; the mlx worker's own adapter
/// classifier (`image_jobs::classify_adapter`, sc-3022) is the backstop for an
/// unstamped third-party LyCORIS file that slips through.
fn image_job_is_mlx_eligible(job: &JobSnapshot) -> bool {
    // Both `image_generate` (text-to-image / character_image / reference) and the
    // distinct `image_edit` job type (Image Studio/Editor "plain Image Edit":
    // `mode=edit_image` + `sourceAssetId`, epic 2427) route through the same
    // per-model predicates. The engine dispatches on payload model+mode, not job
    // type (`run_image_generate_job`), and the per-model arms below already gate
    // `edit_image` (qwen/flux2/sdxl edit → eligible; torch-only edit models aren't
    // in `MLX_ROUTED_MODELS` → torch). Without `image_edit` in this gate, plain
    // Image Edit fell through to torch silently with no `mlx_route_decision`
    // (sc-3513).
    if !matches!(job.job_type, JobType::ImageGenerate | JobType::ImageEdit) {
        return false;
    }
    let Some(model) = job.payload.get("model").and_then(Value::as_str) else {
        return false;
    };
    image_request_mlx_eligible(model, &job.payload)
}

/// Per-model image MLX-eligibility dispatch, factored out of [`image_job_is_mlx_eligible`] so the
/// UI gating oracle ([`model_mac_support`], sc-3486) can probe the same per-family predicates with
/// synthetic payloads — one dispatch table, no divergence between routing and what the UI hides.
fn image_request_mlx_eligible(model: &str, payload: &Map<String, Value>) -> bool {
    if !MLX_ROUTED_MODELS.contains(&model) {
        return false;
    }
    match model {
        "z_image_turbo" | "z_image_edit" => z_image_mlx_eligible(payload),
        "flux_schnell" | "flux_dev" => flux_mlx_eligible(payload),
        "qwen_image" => qwen_mlx_eligible(payload),
        "qwen_image_edit"
        | "qwen_image_edit_2509"
        | "qwen_image_edit_2511"
        | "qwen_image_edit_2511_lightning" => qwen_edit_mlx_eligible(payload),
        "flux2_klein_9b" | "flux2_klein_9b_kv" | "flux2_klein_9b_true_v2" => {
            flux2_mlx_eligible(payload)
        }
        "sdxl" | "realvisxl" => sdxl_mlx_eligible(payload),
        "instantid_realvisxl" => instantid_mlx_eligible(payload),
        "pulid_flux_dev" => pulid_flux_mlx_eligible(payload),
        "chroma1_hd" | "chroma1_base" | "chroma1_flash" => chroma_mlx_eligible(payload),
        "sensenova_u1_8b" | "sensenova_u1_8b_fast" => sensenova_mlx_eligible(payload),
        "kolors" => kolors_mlx_eligible(payload),
        "lens" | "lens_turbo" => lens_mlx_eligible(payload),
        "bernini_image" => bernini_image_mlx_eligible(payload),
        // Every model in MLX_ROUTED_MODELS must have an arm.
        _ => false,
    }
}

/// Does this `image_detail` job belong on the in-process Rust MLX worker? sc-3060 (epic 3041)
/// ports the tile-ControlNet detail refine onto the engine. Detail is SDXL-family only
/// (`sdxl` / `realvisxl`, the detail-capable backbones; the payload defaults to `realvisxl`).
/// Third-party LyCORIS (LoHa / non-peft LoKr) now applies on the SDXL merge path too (epic 3641,
/// sc-3671), so it no longer forces torch. On Windows/Linux no `mlx` worker exists, so detail stays
/// on the Python torch path.
fn image_detail_mlx_eligible(job: &JobSnapshot) -> bool {
    if !matches!(job.job_type, JobType::ImageDetail) {
        return false;
    }
    let model = job
        .payload
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("realvisxl");
    matches!(model, "sdxl" | "realvisxl")
}

/// Whether the in-process MLX worker can serve this GPU job (image_generate or image_detail).
fn job_is_mlx_eligible(job: &JobSnapshot) -> bool {
    image_job_is_mlx_eligible(job) || image_detail_mlx_eligible(job)
}

/// Epic 3180 / sc-3905 routing — does this understanding job (`image_vqa` / `image_interleave`)
/// belong on the in-process Rust MLX worker on macOS? These two modes are SenseNova-U1's
/// understanding/interleave surface, served via the concrete `T2iModel` (`vqa` / `interleave_gen`)
/// because the `Generator` contract emits Images/Video only. SenseNova-U1 is the only model with an
/// in-process understanding path, so eligibility = a SenseNova-U1 id (the worker handler validates
/// the per-mode request: VQA needs a source image + question; interleave needs a prompt). Other
/// models on these job types have no MLX path and stay on the Python torch worker.
fn understanding_job_is_mlx_eligible(job: &JobSnapshot) -> bool {
    if !matches!(job.job_type, JobType::ImageVqa | JobType::ImageInterleave) {
        return false;
    }
    // The understanding job types are SenseNova-specific; a missing model defaults to the base id.
    let model = job
        .payload
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("sensenova_u1_8b");
    matches!(model, "sensenova_u1_8b" | "sensenova_u1_8b_fast")
}

/// SDXL MLX-routing conditions. sc-3026 brought txt2img + LoRA; sc-3060 (epic 3041) adds the
/// advanced shapes the Rust `mlx-gen-sdxl` engine now handles — reference/IP-Adapter, img2img
/// `edit_image`, masked inpaint, and outpaint — so they route to the in-process MLX worker on
/// Mac instead of the Python torch `SdxlDiffusersAdapter`. The torch path stays authoritative
/// on Windows/Linux (no `mlx` worker registered → nothing defers) and as the Mac fallback.
/// Third-party LyCORIS (LoHa / non-peft LoKr) now applies on the SDXL merge path (epic 3641,
/// sc-3671), so every SDXL shape — including a LyCORIS-tagged job — is MLX-eligible.
/// `image_detail` is a separate job type with its own routing (see `image_detail_mlx_eligible`).
fn sdxl_mlx_eligible(_payload: &Map<String, Value>) -> bool {
    true
}

/// The models the candle (Windows/CUDA) lane can serve (epic 3672 sc-3678 for SDXL; epic 5095
/// sc-5096 adds the four image families; sc-5126 adds Lens / Lens-Turbo; sc-5484 + sc-5576 add Chroma,
/// Kolors, and SenseNova-U1). Mirrors the worker's
/// `image_jobs::is_candle_engine`: SDXL/RealVisXL (`realvisxl` shares the candle `"sdxl"` engine via a
/// weights swap), plus z-image-turbo, FLUX.1 schnell/dev, FLUX.2-klein-9B, Qwen-Image,
/// `lens`/`lens_turbo`, `chroma1_hd`/`_base`/`_flash`, `kolors`, and `sensenova_u1_8b`/`_fast` —
/// the base **txt2img** ids only. Deliberately narrow: candle is a gated
/// txt2img-only lane, so every conditioning shape AND every non-base weight variant (e.g.
/// `flux2_klein_9b_kv`, `qwen_image_edit`) falls back to the Python torch worker. Lens is pure T2I
/// (no conditioning at all) but — unlike the others — DOES advertise quant + LoRA/LoKr, so it is also
/// listed in [`CANDLE_QUANT_LORA_MODELS`] below to exempt it from the quant/LoRA → torch fallbacks.
const CANDLE_ROUTED_MODELS: &[&str] = &[
    "sdxl",
    "realvisxl",
    "z_image_turbo",
    "flux_schnell",
    "flux_dev",
    "flux2_klein_9b",
    "qwen_image",
    "lens",
    "lens_turbo",
    // epic 3692 candle image families. Chroma's worker lane (#658) shipped without this router half, so
    // chroma jobs never reached the candle worker — added here with Kolors + SenseNova-U1 (sc-5576). All
    // pure **txt2img** on candle: their edit / IP-reference / pose-control / VQA shapes are rejected
    // below (`image_request_candle_eligible`) and fall back to the Python torch worker.
    "chroma1_hd",
    "chroma1_base",
    "chroma1_flash",
    "kolors",
    "sensenova_u1_8b",
    "sensenova_u1_8b_fast",
];

/// The candle image families that advertise on-the-fly Q4/Q8 quant AND LoRA/LoKr adapters — Lens /
/// Lens-Turbo (sc-5126), the first such candle family. For these a LoRA or an explicit quant request
/// does NOT force the job to the Python torch worker: the candle `generate_candle_stream` maps both
/// into the `LoadSpec` (descriptor-gated, see `ResolvedModel::supports_quant`/`supports_adapters`).
/// Every other candle family advertises neither, so a LoRA/quant request there still defers to torch.
const CANDLE_QUANT_LORA_MODELS: &[&str] = &["lens", "lens_turbo"];

/// Whether `worker` is the candle (Windows/CUDA) SDXL worker — identified by the `candle` marker
/// capability it self-advertises (`gpu::with_candle_capabilities`), mirroring the `nvidia` marker
/// the Rust GPU worker already emits. The candle worker runs on a real CUDA gpu index, not the
/// `mlx` sentinel, so it can't be recognized by `gpu_id`; the marker is the seam. When candle is
/// disabled the worker never advertises the marker, so this is always `false` and routing is
/// unchanged.
fn worker_is_candle(worker: &WorkerSnapshot) -> bool {
    worker
        .capabilities
        .iter()
        .any(|capability| capability.as_str() == "candle")
}

/// Does this image job belong on the candle SDXL **txt2img-only** lane (epic 3672, sc-3678)? The
/// candle generator (`image_jobs::generate_candle_stream`) drives plain text-to-image only — no
/// img2img/edit (`mode == "edit_image"` + `sourceAssetId`), no reference/IP-Adapter
/// (`referenceAssetId`), no masked inpaint/outpaint (`maskAssetId`), no strict-pose ControlNet
/// (`advanced.poses`), and no LoRAs. Every one of those shapes must fall back to the Python torch
/// worker, so the candle worker refuses them here. The conditioning signals mirror the worker's
/// `sdxl_sub_mode` / `pose_entries` exactly, so the router and worker agree on the lane boundary.
fn image_job_is_candle_eligible(job: &JobSnapshot) -> bool {
    if !matches!(job.job_type, JobType::ImageGenerate) {
        return false;
    }
    let Some(model) = job.payload.get("model").and_then(Value::as_str) else {
        return false;
    };
    // InstantID (sc-5491, epic 5480): the candle `candle-gen-instantid` provider serves the SAME
    // identity-preserving surface as the MLX path (single-identity character_image, the angle set,
    // pose-library mode, face-restore) — a bespoke `generate_instantid_stream` lane, NOT the
    // txt2img-only `image_request_candle_eligible` gate (which rejects `referenceAssetId`, which
    // InstantID requires). Branch it out before that gate. Retires the Python `_vendor/instantid`
    // off-Mac; the candle worker only advertises the `candle` marker when the backend is enabled, so a
    // candle-disabled box still falls these jobs back to the Python torch worker unchanged.
    if model == "instantid_realvisxl" {
        return instantid_candle_eligible(&job.payload);
    }
    // SDXL IP-Adapter-Plus reference conditioning (sc-5488, epic 5480): an sdxl-family model with a
    // reference image is a bespoke candle lane (`generate_candle_sdxl_ipadapter_stream`), NOT txt2img —
    // the `image_request_candle_eligible` gate below rejects `referenceAssetId`. Branch it out first
    // (pure IP only; img2img/inpaint/edit shapes stay on torch — those are sc-5487). Mirrors the
    // worker's `sdxl_ipadapter_available` gate.
    if matches!(model, "sdxl" | "realvisxl") && sdxl_ipadapter_candle_eligible(&job.payload) {
        return true;
    }
    // Kolors IP-Adapter-Plus reference conditioning (sc-5488, epic 5480): the `kolors` family with a
    // reference image is the same bespoke candle lane (`generate_candle_kolors_ipadapter_stream`), NOT
    // txt2img — branch it out before the gate (which rejects `referenceAssetId`). Pure IP only;
    // img2img/edit shapes stay on torch (sc-5487). Mirrors the worker's `kolors_ipadapter_available`.
    if model == "kolors" && kolors_ipadapter_candle_eligible(&job.payload) {
        return true;
    }
    // FLUX XLabs IP-Adapter reference conditioning (sc-5872, epic 5480): a `flux_dev`/`flux_schnell`
    // model with a reference image is the same bespoke candle lane (`generate_candle_flux_ipadapter_\
    // stream`), NOT txt2img — branch it out before the gate (which rejects `referenceAssetId`). Pure IP
    // only; img2img/edit shapes stay on torch (sc-5487). Mirrors the worker's `flux_ipadapter_available`.
    if matches!(model, "flux_dev" | "flux_schnell") && flux_ipadapter_candle_eligible(&job.payload)
    {
        return true;
    }
    // Qwen-Image strict-pose ControlNet (sc-5489, epic 5480): `qwen_image` + `advanced.poses` is a
    // bespoke candle lane (`generate_candle_qwen_control_stream`), NOT txt2img — the
    // `image_request_candle_eligible` gate below DEFERS any `advanced.poses` job to torch. Branch it out
    // first so `qwen_image` pose jobs reach candle (the kolors / z_image families follow below — all three
    // strict-pose families are now wired; plain-sdxl pose has no product route). Mirrors the worker's
    // `qwen_control_available`.
    if model == "qwen_image" && qwen_control_candle_eligible(&job.payload) {
        return true;
    }
    // Kolors strict-pose ControlNet (sc-5489, epic 5480): `kolors` + `advanced.poses` is the bespoke
    // candle lane (`generate_candle_kolors_control_stream`), NOT txt2img — the `image_request_candle_\
    // eligible` gate below DEFERS any `advanced.poses` job to torch. Branch it out first (the Qwen-control
    // reasoning, for the Kolors family). A pure-pose `kolors` job (no `referenceAssetId`) does NOT match
    // the `kolors_ipadapter_candle_eligible` branch above, so it reaches here. Mirrors the worker's
    // `kolors_control_available`.
    if model == "kolors" && kolors_control_candle_eligible(&job.payload) {
        return true;
    }
    // Z-Image strict-pose Fun-ControlNet (sc-5489, epic 5480): `z_image_turbo` + `advanced.poses` is the
    // bespoke candle lane (`generate_candle_zimage_control_stream`), NOT txt2img — the `image_request_\
    // candle_eligible` gate below DEFERS any `advanced.poses` job to torch. Branch it out first (the
    // Qwen/Kolors-control reasoning, for the last strict-pose family). Mirrors the worker's
    // `zimage_control_available`. With this all three control families (qwen / kolors / z_image) are wired.
    if model == "z_image_turbo" && zimage_control_candle_eligible(&job.payload) {
        return true;
    }
    image_request_candle_eligible(model, &job.payload)
}

/// Per-model candle txt2img-eligibility, factored out of [`image_job_is_candle_eligible`] so the
/// routing tests can probe it with synthetic payloads (parity with `image_request_mlx_eligible`).
fn image_request_candle_eligible(model: &str, payload: &Map<String, Value>) -> bool {
    if !CANDLE_ROUTED_MODELS.contains(&model) {
        return false;
    }
    // img2img / inpaint / outpaint all arrive as `mode == "edit_image"` (+ a source); reject the
    // whole edit family up front (the worker's `sdxl_sub_mode` keys off the same mode).
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    let has_nonempty_id = |key: &str| {
        payload
            .get(key)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
    };
    // Any conditioning asset (img2img source, IP-Adapter reference, or inpaint mask) → torch. Applies
    // to EVERY candle family including Lens (pure T2I — no conditioning shapes in the Lens port).
    if has_nonempty_id("sourceAssetId")
        || has_nonempty_id("referenceAssetId")
        || has_nonempty_id("maskAssetId")
    {
        return false;
    }
    // Lens / Lens-Turbo advertise Q4/Q8 + LoRA/LoKr, so a quant request or a LoRA stays on the candle
    // lane for them; every other candle family advertises neither and defers those to torch.
    let supports_quant_lora = CANDLE_QUANT_LORA_MODELS.contains(&model);
    // LoRAs: not in the candle lane unless the family advertises adapters (Lens).
    if !supports_quant_lora
        && payload
            .get("loras")
            .and_then(Value::as_array)
            .is_some_and(|loras| !loras.is_empty())
    {
        return false;
    }
    // Strict-pose ControlNet (`advanced.poses`, object-shaped entries) → torch.
    let has_poses = payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("poses"))
        .and_then(Value::as_array)
        .is_some_and(|poses| !poses.is_empty());
    if has_poses {
        return false;
    }
    // On-the-fly quantization (`advanced.mlxQuantize` > 0) → torch UNLESS the family advertises quant.
    // The sc-3675/sc-5096 candle providers advertise `supported_quants: &[]` (dense bf16/fp16 only), so
    // an explicit quant request can't be honored — route to Python rather than silently running dense
    // (sc-5099). Lens advertises Q4/Q8, so its quant request stays here (sc-5126).
    if !supports_quant_lora && candle_request_wants_quant(payload) {
        return false;
    }
    true
}

/// Whether the request explicitly asks for on-the-fly quantization the candle backend can't do.
/// `advanced.mlxQuantize` is an optional advanced override (the web UI doesn't send it; the MLX path
/// otherwise defaults quant from the manifest) — so a payload-level value `> 0` is a deliberate quant
/// request. `<= 0` (dense) and absent both leave candle on its native dense path (sc-5099).
fn candle_request_wants_quant(payload: &Map<String, Value>) -> bool {
    payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("mlxQuantize"))
        .and_then(|value| {
            value
                .as_i64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .is_some_and(|bits| bits > 0)
}

/// The video models the candle (Windows/CUDA) lane serves: the base txt2video engines `wan_2_2`
/// (→ candle `wan2_2_ti2v_5b`) and `ltx_2_3` (→ candle `ltx_2_3_distilled`) (epic 5095, sc-5097),
/// plus the Wan2.2 **14B** MoE pair `wan_2_2_t2v_14b` (text-only) and `wan_2_2_i2v_14b` (image→video)
/// (sc-5175), plus `svd` (→ candle `svd_xt`, image→video, sc-5493 / epic 5481). Mirrors the worker's
/// `video_jobs::candle_video_engine_id`. `ltx_2_3_eros` (sc-5495) now routes to candle for plain
/// text-to-video too — it's a full dense LTX-2.3 fine-tune → the same `ltx_2_3_distilled` engine, just
/// its own weights repo; every conditioned mode (first_last_frame / extend / bridge / replace) + LoRA
/// still stays on the Python torch worker. Note the 14B I2V and SVD are image→video, NOT txt2video —
/// see [`CANDLE_VIDEO_I2V_ROUTED_MODELS`].
const CANDLE_VIDEO_ROUTED_MODELS: &[&str] = &[
    "wan_2_2",
    "ltx_2_3",
    "ltx_2_3_eros",
    "wan_2_2_t2v_14b",
    "wan_2_2_i2v_14b",
    "svd",
];

/// The candle video models that run **image→video** (a source image is required), not txt2video: the
/// Wan2.2 14B I2V MoE (sc-5175) and SVD (`svd` → `svd_xt`, sc-5493). Their candle providers condition on
/// a source frame, so their eligibility gate requires `mode=image_to_video` + a non-empty
/// `sourceAssetId` — the inverse of the txt2video-only gate the 5B / T2V-14B / ltx ids use.
const CANDLE_VIDEO_I2V_ROUTED_MODELS: &[&str] = &["wan_2_2_i2v_14b", "svd"];

/// Does this video job belong on the candle video lane? The candle wan/ltx providers drive plain
/// text-to-video, the 14B I2V's single source-image conditioning (sc-5175), SVD image→video (sc-5493),
/// **and** the Wan-VACE advanced modes — replace_person / extend / bridge (sc-5494, the `PersonReplace`
/// / `VideoExtend` / `VideoBridge` job types → the candle `wan_vace` engine). Every other shape
/// (reference/mask/first-last-frame conditioning, LoRAs, SCAIL-2 replace) must fall back to the Python
/// torch worker, so the candle worker refuses it here. The per-model shape gates are
/// [`video_request_candle_eligible`] (base) and [`video_request_candle_vace_eligible`] (VACE modes).
fn video_job_is_candle_eligible(job: &JobSnapshot) -> bool {
    let Some(model) = job.payload.get("model").and_then(Value::as_str) else {
        return false;
    };
    match job.job_type {
        // The base txt2video / image→video lane (sc-5097 / sc-5175 / sc-5493).
        JobType::VideoGenerate => video_request_candle_eligible(model, &job.payload),
        // The Wan-VACE advanced modes (sc-5494): replace_person / extend_clip / video_bridge.
        JobType::PersonReplace | JobType::VideoExtend | JobType::VideoBridge => {
            video_request_candle_vace_eligible(model, &job.payload, &job.job_type)
        }
        _ => false,
    }
}

/// Per-model candle txt2video-eligibility, factored out so the routing tests can probe it with
/// synthetic payloads (parity with `image_request_candle_eligible`).
fn video_request_candle_eligible(model: &str, payload: &Map<String, Value>) -> bool {
    if !CANDLE_VIDEO_ROUTED_MODELS.contains(&model) {
        return false;
    }
    let has_nonempty_id = |key: &str| {
        payload
            .get(key)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
    };
    if CANDLE_VIDEO_I2V_ROUTED_MODELS.contains(&model) {
        // Wan 14B I2V is image→video ONLY (sc-5175): require the `image_to_video` mode + a source
        // image. A txt2video shape (no source) is rejected so a mis-picked text job stays on torch.
        if payload.get("mode").and_then(Value::as_str) != Some("image_to_video") {
            return false;
        }
        if !has_nonempty_id("sourceAssetId") {
            return false;
        }
    } else {
        // txt2video only: the base `video_generate` mode defaults to `image_to_video`, so require an
        // explicit `text_to_video`. Every conditioned mode (i2v / first_last_frame / extend / bridge /
        // replace) is thereby excluded, as is a stray source image.
        if payload.get("mode").and_then(Value::as_str) != Some("text_to_video") {
            return false;
        }
        if has_nonempty_id("sourceAssetId") {
            return false;
        }
    }
    // Reference / inpaint-mask conditioning is never in the candle video lane (i2v needs only the
    // single source image; reference + mask are the character / inpaint shapes that stay on torch).
    if has_nonempty_id("referenceAssetId") || has_nonempty_id("maskAssetId") {
        return false;
    }
    // LoRAs are not in the candle video lane (the providers advertise none).
    if payload
        .get("loras")
        .and_then(Value::as_array)
        .is_some_and(|loras| !loras.is_empty())
    {
        return false;
    }
    // On-the-fly quantization → torch (the candle video providers are dense; sc-5099).
    if candle_request_wants_quant(payload) {
        return false;
    }
    true
}

/// The candle video models eligible for the Wan-VACE advanced modes (sc-5494). These route to the
/// single candle `wan_vace` engine regardless of the user's wan pick. The SCAIL-2 person-replace
/// backend is MLX-only, so `scail2_*` is deliberately absent (those stay on the torch / mac worker).
const CANDLE_VIDEO_VACE_MODELS: &[&str] = &["wan_2_2", "wan_2_2_t2v_14b", "wan_2_2_i2v_14b"];

/// Candle Wan-VACE eligibility for the advanced video job types (sc-5494): `PersonReplace`
/// (replace_person), `VideoExtend` (extend_clip), `VideoBridge` (video_bridge). Routes to the candle
/// `wan_vace` engine when the model is VACE-capable and the per-mode source assets are present. LoRA /
/// on-the-fly quant are not in the candle video lane (the VACE provider rejects them). Factored out so
/// the routing tests can probe it with synthetic payloads (parity with [`video_request_candle_eligible`]).
fn video_request_candle_vace_eligible(
    model: &str,
    payload: &Map<String, Value>,
    job_type: &JobType,
) -> bool {
    if !CANDLE_VIDEO_VACE_MODELS.contains(&model) {
        return false;
    }
    let has_nonempty_id = |key: &str| {
        payload
            .get(key)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
    };
    match job_type {
        // replace_person: the source control clip + the tracked person + the character references.
        JobType::PersonReplace => {
            if !has_nonempty_id("sourceClipAssetId")
                || !has_nonempty_id("personTrackId")
                || !has_nonempty_id("characterId")
            {
                return false;
            }
        }
        // extend_clip: the source clip whose tail anchors the continuation.
        JobType::VideoExtend => {
            if !has_nonempty_id("sourceClipAssetId") {
                return false;
            }
        }
        // video_bridge: both clips (the left tail + the right head) are pinned around the gap.
        JobType::VideoBridge => {
            if !has_nonempty_id("sourceClipAssetId") || !has_nonempty_id("bridgeRightClipAssetId") {
                return false;
            }
        }
        _ => return false,
    }
    // LoRAs / on-the-fly quant are not in the candle video lane (the VACE provider rejects them).
    if payload
        .get("loras")
        .and_then(Value::as_array)
        .is_some_and(|loras| !loras.is_empty())
    {
        return false;
    }
    if candle_request_wants_quant(payload) {
        return false;
    }
    true
}

/// InstantID (`instantid_realvisxl`) MLX-routing conditions. The native `mlx-gen-instantid`
/// provider now serves the FULL surface on Mac: single-identity `character_image`, the 11-view
/// Character-Studio angle set (sc-3345), AND pose-library mode + face-restore (sc-3381, on the
/// #193 engine — `generate_pose` MultiControlNet IdentityNet+OpenPose / `restore_face`). So every
/// `character_image` job with a reference face routes to MLX; only a non-character / reference-less
/// job stays off. Mirrors the worker's `instantid_available` gate so the router and worker agree.
fn instantid_mlx_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) != Some("character_image") {
        return false;
    }
    payload
        .get("referenceAssetId")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
}

/// InstantID candle-routing conditions (sc-5491, epic 5480). The candle `candle-gen-instantid`
/// provider is the off-Mac sibling of `mlx-gen-instantid` and serves the IDENTICAL surface (single
/// identity, the angle set, pose-library mode, face-restore via `generate_pose` / `restore_face`), so
/// the gate is the same as [`instantid_mlx_eligible`]: a `character_image` job with a reference face.
/// Mirrors the candle worker's `instantid_available` gate so the router and worker agree.
fn instantid_candle_eligible(payload: &Map<String, Value>) -> bool {
    instantid_mlx_eligible(payload)
}

/// SDXL IP-Adapter-Plus candle-routing conditions (sc-5488, epic 5480). The candle `IpAdapterSdxl`
/// provider serves PURE reference (image-prompt) conditioning on the sdxl family: a `referenceAssetId`
/// with NO img2img source / inpaint mask and NOT an `edit_image` (those advanced SDXL shapes are
/// sc-5487, still torch). Mirrors the worker's `sdxl_ipadapter_available` gate (minus the local
/// weight-resolve check) so the router and worker agree on the lane boundary. Candle-only — there is no
/// MLX `IpAdapterSdxl` (the MLX SDXL IP path is the registry `SdxlSubMode::Ip`), so this has no
/// `*_mlx_eligible` sibling.
fn sdxl_ipadapter_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    let non_empty = |key: &str| {
        payload
            .get(key)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
    };
    non_empty("referenceAssetId") && !non_empty("sourceAssetId") && !non_empty("maskAssetId")
}

/// Kolors IP-Adapter-Plus candle-routing conditions (sc-5488, epic 5480). The candle `IpAdapterKolors`
/// provider serves PURE reference (image-prompt) conditioning on the `kolors` family — the same payload
/// shape as the SDXL IP lane: a `referenceAssetId` with NO img2img source / inpaint mask and NOT an
/// `edit_image` (those advanced Kolors shapes are sc-5487, still torch). Mirrors the worker's
/// `kolors_ipadapter_available` gate (minus the local weight-resolve check) so the router and worker
/// agree on the lane boundary. Candle-only — the macOS Kolors IP path is the registry `Reference` route,
/// not a separate candle-eligible gate.
fn kolors_ipadapter_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    let non_empty = |key: &str| {
        payload
            .get(key)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
    };
    non_empty("referenceAssetId") && !non_empty("sourceAssetId") && !non_empty("maskAssetId")
}

/// FLUX XLabs IP-Adapter candle-routing conditions (sc-5872, epic 5480). The candle `IpAdapterFlux`
/// provider serves PURE reference (image-prompt) conditioning on the `flux_dev`/`flux_schnell` families
/// — the same payload shape as the SDXL/Kolors IP lanes: a `referenceAssetId` with NO img2img source /
/// inpaint mask and NOT an `edit_image` (those advanced FLUX shapes are sc-5487, still torch). Mirrors
/// the worker's `flux_ipadapter_available` gate (minus the local weight-resolve check) so the router and
/// worker agree on the lane boundary. Candle-only — the macOS FLUX IP path is the registry `Reference`
/// route (epic 3621), not a separate candle-eligible gate.
fn flux_ipadapter_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    let non_empty = |key: &str| {
        payload
            .get(key)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
    };
    non_empty("referenceAssetId") && !non_empty("sourceAssetId") && !non_empty("maskAssetId")
}

/// Qwen-Image strict-pose ControlNet candle-routing conditions (sc-5489, epic 5480). The candle
/// `QwenControl` provider serves `qwen_image` + a non-empty object `advanced.poses` (one image per pose,
/// each conditioned on a DWPose skeleton), NOT an `edit_image`. A `referenceAssetId`, if present, is
/// ignored (identity comes from a character LoRA on the base, mirroring the MLX/torch
/// `QwenImageControlNetPipeline`). Mirrors the worker's `qwen_control_available` gate (minus the local
/// weight-resolve check) so the router and worker agree. Candle-only — the macOS path is the registry
/// `qwen_image_control` generator, not a separate candle-eligible gate.
fn qwen_control_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("poses"))
        .and_then(Value::as_array)
        .is_some_and(|poses| !poses.is_empty())
}

/// Kolors strict-pose ControlNet candle-routing conditions (sc-5489, epic 5480). The candle
/// `KolorsControl` provider serves `kolors` + a non-empty `advanced.poses` (one image per pose, each
/// conditioned on a DWPose skeleton via the `Kwai-Kolors/Kolors-ControlNet-Pose` branch), NOT an
/// `edit_image`. Same shape as `qwen_control_candle_eligible` — the model gate (`kolors`) is applied at
/// the call site. Mirrors the worker's `kolors_control_available` gate (minus the local weight-resolve
/// check) so the router and worker agree. Candle-only — the macOS path is the MLX Kolors ControlNet.
fn kolors_control_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("poses"))
        .and_then(Value::as_array)
        .is_some_and(|poses| !poses.is_empty())
}

/// Z-Image strict-pose Fun-ControlNet candle-routing conditions (sc-5489, epic 5480). The candle
/// `ZImageControl` provider serves `z_image_turbo` + a non-empty `advanced.poses` (one image per pose,
/// each conditioned on a DWPose skeleton via the VACE-style `Z-Image-Turbo-Fun-Controlnet-Union-2.1`
/// branch), NOT an `edit_image`. Same shape as the qwen/kolors gates — the model gate (`z_image_turbo`)
/// is applied at the call site. Mirrors the worker's `zimage_control_available`. Candle-only — the macOS
/// path is the MLX `z_image_turbo_control` registry generator.
fn zimage_control_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("poses"))
        .and_then(Value::as_array)
        .is_some_and(|poses| !poses.is_empty())
}

/// Candle-routed image models that HAVE a candle strict-pose lane (sc-5489). A `advanced.poses` job on
/// any OTHER candle-routed model has no pose path on candle (plain-SDXL pose ships via InstantID,
/// `instantid_realvisxl`, not `sdxl`).
fn model_has_candle_pose_lane(model: &str) -> bool {
    matches!(model, "qwen_image" | "kolors" | "z_image_turbo")
}

/// A strict-pose (`advanced.poses`) job on a **candle-routed model with no candle pose lane** —
/// `sdxl` / `realvisxl` / `chroma*` / `flux*` / `lens*` / `sensenova*` (everything but the three wired
/// pose families), not `edit_image` (sc-5968, epic 5483). Neither candle nor the co-resident torch
/// worker has a pose path for these models off-Mac (the torch `sdxl` adapter's OpenPose lives only in
/// the `instantid_realvisxl` adapter), so torch would silently drop the poses → an unconditioned T2I
/// image. The candle worker therefore CLAIMS these (`worker_supports_job`) to REJECT them with a typed
/// error in the handler, and the co-resident torch worker DECLINES them (below) so candle reliably wins
/// and nothing silently mis-serves them. **Mac is unaffected:** `sdxl + poses` is MLX-served there
/// (`model_mac_support("sdxl").features.pose`), so the MLX worker claims it and only the torch/`mps`
/// worker declines. Pairs with the worker's `candle_unsupported_pose_reject` dispatch guard.
fn image_request_candle_pose_reject(model: &str, payload: &Map<String, Value>) -> bool {
    if !CANDLE_ROUTED_MODELS.contains(&model) || model_has_candle_pose_lane(model) {
        return false;
    }
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("poses"))
        .and_then(Value::as_array)
        .is_some_and(|poses| !poses.is_empty())
}

/// [`image_request_candle_pose_reject`] on a [`JobSnapshot`].
fn image_job_candle_pose_reject(job: &JobSnapshot) -> bool {
    if !matches!(job.job_type, JobType::ImageGenerate) {
        return false;
    }
    let Some(model) = job.payload.get("model").and_then(Value::as_str) else {
        return false;
    };
    image_request_candle_pose_reject(model, &job.payload)
}

/// PuLID-FLUX (`pulid_flux_dev`) MLX-routing conditions (sc-3344). The native `mlx-gen-pulid`
/// registry generator serves the single surface PuLID-FLUX has: a `character_image` job with a
/// reference face (no plain text-to-image, no `edit_image` — the engine requires the face it
/// injects). Mirrors the worker's `pulid_flux_available` gate so the router and worker agree, and
/// mirrors `instantid_mlx_eligible` (its face-identity sibling). The "person-type vs non-face"
/// split is the upstream model-id choice — a person character selects `pulid_flux_dev`; a
/// non-person reference selects `flux_dev` + the native XLabs IP-Adapter (epic 3621) — so no
/// separate fall-through gate is needed here. PuLID has no user-LoRA path (`supports_lora=false`),
/// and the torch path ignored LoRAs too, so a LoRA never changes eligibility.
fn pulid_flux_mlx_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) != Some("character_image") {
        return false;
    }
    payload
        .get("referenceAssetId")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
}

/// FLUX.2-klein MLX-routing conditions. FLUX.2-klein is an **MLX-only** family (no torch backend),
/// so everything it does runs on MLX: txt2img (sc-3025), edit/reference + KV-cache + multi-reference
/// (sc-3029), and — since epic 3641 (sc-3642/3643) — third-party LyCORIS via the core loader.
fn flux2_mlx_eligible(_payload: &Map<String, Value>) -> bool {
    true
}

/// Qwen-Image (sc-3024 / strict pose sc-3575) MLX-routing conditions: text-to-image,
/// plus the base-Qwen strict pose tier (`advanced.poses`) handled by the `qwen_image_control`
/// engine variant. A reference without poses (character/edit flow) and `edit_image` stay on
/// the Python torch path. Third-party LyCORIS (LoHa / non-peft LoKr) now applies on the core MLX
/// loader (epic 3641, sc-3642/3643), so it no longer forces torch.
fn qwen_mlx_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    let has_poses = payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("poses"))
        .and_then(Value::as_array)
        .is_some_and(|poses| !poses.is_empty());
    if has_poses {
        return true;
    }
    let has_reference = payload
        .get("referenceAssetId")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty());
    if has_reference {
        return false;
    }
    true
}

/// Qwen-Image-Edit (sc-3397/sc-3398) MLX-routing conditions. The `qwen_image_edit` /
/// `_2509` / `_2511` / `_2511_lightning` ids run the engine's `qwen_image_edit` model on
/// the Rust worker (the edit sibling of `qwen_mlx_eligible`). Eligible when the job carries
/// the reference the edit model requires: `edit_image` with a `sourceAssetId` (or a
/// `referenceAssetId`), or `character_image` with a `referenceAssetId` (the subject-variation
/// / best-effort-pose / angle-set flows — all reference-conditioned). The lightning distill
/// (sc-3398) shares the same gate (its sampler + distill-LoRA are worker-local). Third-party
/// LyCORIS now applies on the core MLX loader (epic 3641), so it no longer forces torch.
fn qwen_edit_mlx_eligible(payload: &Map<String, Value>) -> bool {
    let has_reference = payload
        .get("referenceAssetId")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty());
    let has_source = payload
        .get("sourceAssetId")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty());
    match payload.get("mode").and_then(Value::as_str) {
        Some("edit_image") => has_source || has_reference,
        Some("character_image") => has_reference,
        _ => false,
    }
}

/// FLUX.1 (sc-3023) MLX-routing conditions, ported from `_should_route_flux_to_mlx`:
/// text-to-image only — FLUX.1 reference/IP-Adapter and `edit_image` stay on the
/// Python torch path (`FluxDiffusersAdapter`). A third-party LyCORIS LoRA also falls
/// back to torch: the engine + the worker's `classify_adapter` apply LoRA and peft
/// LoKr natively, but not arbitrary LyCORIS (which the worker would reject).
/// FLUX.1 (`flux_schnell` / `flux_dev`) MLX-routing conditions. Text-to-image and
/// **reference-image** (the XLabs IP-Adapter, epic 3621 — `referenceAssetId`, both
/// variants: the Rust engine has no diffusers `load_ip_adapter` schnell limitation,
/// so reference is native on schnell too). `edit_image` stays off — FLUX.1 has no
/// edit path on any platform (a future Kontext epic, NOT a Python-eradication gap).
/// Third-party LyCORIS now applies on the core MLX loader (epic 3641), so only `edit_image`
/// keeps a FLUX.1 job off MLX.
fn flux_mlx_eligible(payload: &Map<String, Value>) -> bool {
    payload.get("mode").and_then(Value::as_str) != Some("edit_image")
}

/// Z-Image (sc-3022) MLX-routing conditions, ported from
/// `_should_route_z_image_to_mlx`: text-to-image, reference-identity img2img-init
/// (sc-3619 — `referenceAssetId` without a pose set, the plain img2img path the
/// base engine already supports), reference+pose (the Fun-ControlNet pose tier
/// lives only on MLX — sc-2257/sc-2328, so a reference+pose job must NOT divert to
/// torch, which would honour count while dropping the poses), and `edit_image`
/// img2img-edit (epic 3529 — the engine's `Conditioning::Reference` img2img path with a
/// `sourceAssetId` init, shared by `z_image_turbo` edit_image mode and the `z_image_edit`
/// model, both on Turbo weights). An `edit_image` without a source asset has nothing to
/// edit, so it stays off MLX. Third-party LyCORIS now applies on the core MLX loader
/// (epic 3641), so a LoRA never forces torch.
fn z_image_mlx_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return payload
            .get("sourceAssetId")
            .and_then(Value::as_str)
            .is_some_and(|id| !id.trim().is_empty());
    }
    true
}

/// Chroma (epic 3531, sc-3843) MLX-routing conditions. Chroma is **text-to-image only**
/// (`text_to_image` + `style_variations`; no edit / reference / ControlNet — those would be
/// later engine ports), so every non-edit `image_generate` job routes to the in-process Rust
/// `mlx-gen-chroma` worker on Mac. An `edit_image` mode — which Chroma has no path for on any
/// platform — stays off MLX (defensive; the UI never offers edit for Chroma). All three variants
/// (`chroma1_hd` / `chroma1_base` / `chroma1_flash`) share this gate. Third-party LyCORIS and peft
/// LoKr apply on the core MLX loader (epic 3641 / sc-3842), so a LoRA never forces torch.
fn chroma_mlx_eligible(payload: &Map<String, Value>) -> bool {
    payload.get("mode").and_then(Value::as_str) != Some("edit_image")
}

/// SenseNova-U1 (sc-3900, epic 3180) MLX-routing conditions. The unified NEO-Unify model serves
/// three image modes on the single `sensenova_u1_8b` / `sensenova_u1_8b_fast` ids: plain T2I
/// (base path), instruction edit (`edit_image` → `Conditioning::Reference`), and Character Studio
/// (`character_image` → `Conditioning::MultiReference`, incl. the angle set) — all via the Rust
/// worker. It has NO ControlNet, so the strict-pose tier (`advanced.poses`) is unsupported and
/// drops to torch on non-Mac (it has no Mac path — epic 3482). Edit/character require the
/// reference the it2i path needs; plain T2I is always eligible. User LoRAs are not supported
/// (`supports_lora=false`) and the manifest surfaces no LoRA slot, so no LoRA gate is needed.
fn sensenova_mlx_eligible(payload: &Map<String, Value>) -> bool {
    let has_poses = payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("poses"))
        .and_then(Value::as_array)
        .is_some_and(|poses| !poses.is_empty());
    if has_poses {
        // No skeleton/ControlNet conditioning — strict pose is not an MLX SenseNova path.
        return false;
    }
    let has_reference = payload
        .get("referenceAssetId")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty());
    let has_source = payload
        .get("sourceAssetId")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty());
    match payload.get("mode").and_then(Value::as_str) {
        Some("edit_image") => has_source || has_reference,
        Some("character_image") => has_reference,
        // Plain T2I (text_to_image / no mode) — eligible with or without an inert reference.
        _ => true,
    }
}

/// Kolors (epic 3090) MLX-routing conditions. The engine `kolors` model (an SDXL-family U-Net under
/// a ChatGLM3-6B encoder) now runs the **full surface** on the in-process Rust worker: plain T2I
/// (sc-3875), img2img (`edit_image` + `sourceAssetId`, sc-4765), the IP-Adapter-Plus reference
/// (`referenceAssetId`, sc-4767) — all via the base `Reference` path — and the strict-pose tier
/// (`advanced.poses` + a reference, the combined pose-ControlNet + IP-Adapter-identity + img2img pass:
/// engine sc-5012 + the worker `generate_kolors_control_stream`, sc-4766). A pose set without a
/// reference is not the pose tier (torch `_pose_entries` ignores it) and falls through to the base
/// path as plain T2I — same as torch — so every Kolors job is MLX-eligible. Third-party LyCORIS / peft
/// LoKr apply on the SDXL-family loader (epic 3641), so a LoRA never forces torch.
fn kolors_mlx_eligible(_payload: &Map<String, Value>) -> bool {
    true
}

/// Lens / Lens-Turbo (epic 3164 / sc-5105) is a pure T2I family — the `mlx-gen-lens` descriptor
/// advertises no conditioning (no img2img / ControlNet / IP), and the base + turbo ids share the
/// architecture/weights tree, differing only in their step/guidance defaults. Every non-edit
/// `image_generate` job routes to the in-process Rust `mlx-gen-lens` worker on Mac. An `edit_image`
/// mode — which Lens has no path for on any platform (`supportsEdit=false`) — stays off MLX so it is
/// never silently run as plain T2I against a dropped source image (defensive; the UI never offers
/// edit for Lens). Mirrors [`chroma_mlx_eligible`]. (LoRA/LoKr apply at load on the DiT — sc-3174 —
/// so a LoRA never forces torch; LoRA/LoKr *training* is also native MLX now — the `lens_lora` kernel
/// routes to the `mlx-gen-lens` Rust trainer via [`MLX_ROUTED_TRAINING_KERNELS`], sc-5148/sc-5180.)
fn lens_mlx_eligible(payload: &Map<String, Value>) -> bool {
    payload.get("mode").and_then(Value::as_str) != Some("edit_image")
}

/// Bernini still-image companion (epic 4699 / sc-5424) MLX-routing conditions. The image-typed
/// `bernini_image` id serves two still tasks on the same `engine_id:"bernini"` planner+renderer the
/// video `bernini` id uses: plain text-to-image (t2i, the base path) and `edit_image` img2img (i2i —
/// the source image is VAE/ViT-encoded as the engine's `Conditioning::Reference`, with the worker
/// forcing `frames:1` + `video_mode:"t2i"|"i2i"` so the engine returns a single still). An
/// `edit_image` mode without a `sourceAssetId` has nothing to edit, so it stays off MLX (mirrors
/// [`z_image_mlx_eligible`]); plain t2i is always eligible. There is no reference/character/pose
/// still surface (the renderer's reference path is video-only — `reference_to_video`), and the
/// engine reports `supports_lora: false`, so no LoRA gate is needed. macOS-only (the engine is
/// `mac_only`); on Windows/Linux no `mlx` worker is registered, so nothing defers.
fn bernini_image_mlx_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return payload
            .get("sourceAssetId")
            .and_then(Value::as_str)
            .is_some_and(|id| !id.trim().is_empty());
    }
    true
}

/// Video models the in-process Rust MLX worker generates today (sc-3034 Wan2.2,
/// sc-3035 LTX-2.3 + audio, sc-3523 SVD-XT image→video). Mirrors
/// `MlxVideoAdapter._supported_models`. A model id absent here is never routed to the
/// mlx worker — the Python torch path stays authoritative for it.
const VIDEO_MLX_ROUTED_MODELS: &[&str] = &[
    "ltx_2_3",
    "ltx_2_3_eros",
    "wan_2_2",
    "wan_2_2_t2v_14b",
    "wan_2_2_i2v_14b",
    "svd",
    // Bernini (epic 4699 / sc-4707): full Qwen2.5-VL planner + Wan2.2-T2V-A14B
    // renderer, native MLX (engine id "bernini"). Slice A serves text_to_video
    // only; the editing/reference video modes (v2v/mv2v/r2v/rv2v/ads2v) are
    // net-new UI vocabulary tracked under sc-4703.
    "bernini",
    // SCAIL-2 (epic 5439 / sc-5448): Wan2.1-14B I2V end-to-end character animation,
    // native MLX (engine id "scail2_14b"). Serves the standalone `animate_character`
    // mode; cross-identity `replace_person` reuses the same engine, wired in sc-5452.
    "scail2_14b",
];

/// Epic 3018 routing (sc-3036, the video sibling of [`image_job_is_mlx_eligible`]):
/// does this video job belong on the in-process Rust MLX worker? Encodes today's
/// Python `create_video_adapter` MLX-eligibility (video_adapters.py) at the claim
/// layer, minus the worker-local gates (MPS presence / sidecar) — those are now
/// expressed by whether an `mlx` worker is registered and idle (see
/// [`should_defer_video_to_mlx_worker`]).
///
/// MLX covers `text_to_video` + `image_to_video` on Wan/LTX, `image_to_video` on SVD
/// (`svd`→`svd_xt`, image-conditioned only — sc-3523), `first_last_frame` on the FLF-capable
/// engines (LTX + Wan TI2V-5B `wan_2_2`; sc-3520), the clip-conditioning modes `extend_clip` /
/// `video_bridge` on the LTX IC-LoRA path **and Wan TI2V-5B** (sc-3522 / sc-3357, the `VideoExtend`
/// / `VideoBridge` job types — Wan via single-frame boundary keyframe conditioning), and
/// `replace_person` → native Wan-VACE (the `PersonReplace` job type, sc-3521 — see
/// [`video_mode_is_mlx_eligible`]). Still on the Python torch path: a non-MLX model, and
/// extend/bridge on the 14B Wan MoE engines (no `Keyframe` path).
/// **Third-party LyCORIS (LoHa / non-peft LoKr) and LoKr-on-Wan now run on MLX**
/// (epic 3641, sc-3671 + sc-3644): the Wan/LTX engine paths reconstruct + merge/residual the delta —
/// the peft-LoKr-on-Wan merge has existed since sc-2393, and the old `create_video_adapter` torch
/// gate was a routing caution, never an engine limit.
fn video_job_is_mlx_eligible(job: &JobSnapshot) -> bool {
    // The base `video_generate` job type plus the advanced job types: the clip-conditioning
    // `video_extend` / `video_bridge` (sc-3522, LTX IC-LoRA) and `person_replace` (sc-3521 →
    // Wan-VACE). The per-model/per-mode gate below keeps each mode to its capable engines.
    if !matches!(
        job.job_type,
        JobType::VideoGenerate
            | JobType::VideoExtend
            | JobType::VideoBridge
            | JobType::PersonReplace
    ) {
        return false;
    }
    let Some(model) = job.payload.get("model").and_then(Value::as_str) else {
        return false;
    };
    if !VIDEO_MLX_ROUTED_MODELS.contains(&model) {
        return false;
    }
    // The advanced job types carry their mode by construction (the API maps
    // `extend_clip`→`VideoExtend` / `video_bridge`→`VideoBridge` / `replace_person`→
    // `PersonReplace`), so derive it from the job type rather than trusting the payload
    // `mode` — a missing/stale `mode` on those types must not fall through to the
    // `image_to_video` default and route incorrectly. The base `video_generate` type reads
    // the payload `mode` (default `image_to_video`, mirroring `video_request_from_job`).
    let mode = match job.job_type {
        JobType::VideoExtend => "extend_clip",
        JobType::VideoBridge => "video_bridge",
        JobType::PersonReplace => "replace_person",
        _ => job
            .payload
            .get("mode")
            .and_then(Value::as_str)
            .unwrap_or("image_to_video"),
    };
    if !video_mode_is_mlx_eligible(model, mode) {
        return false;
    }
    true
}

/// Which `video_generate` modes the in-process Rust MLX worker serves for `model`. The Wan/LTX
/// engines serve `text_to_video` + `image_to_video` (sc-3034/3035); `first_last_frame` is
/// additionally MLX on the FLF-capable engines — LTX (`ltx_2_3`/`ltx_2_3_eros`, the
/// reference-grounded `Keyframe` path, sc-3052) and Wan TI2V-5B (`wan_2_2`, the mask-blend
/// multi-keyframe path, sc-3357). The 14B Wan MoE engines have no `Keyframe` path, so FLF on
/// them stays torch. **SVD (`svd`) is image-conditioned only** — it serves `image_to_video`
/// exclusively (no text→video, sc-3523). The clip-conditioning modes `extend_clip` /
/// `video_bridge` are MLX on the **LTX** engines (`ltx_2_3`/`ltx_2_3_eros`, the IC-LoRA
/// multi-frame keyframe-append path — sc-3522, engine `build_clips` sc-3052/3053) **and Wan
/// TI2V-5B** (`wan_2_2`, single-frame boundary `Keyframe` conditioning — sc-3357: extend pins the
/// source clip's last frame, bridge pins the two boundary frames, the same mask-blend primitive as
/// Wan FLF, matching the torch Wan reference which routed these to plain i2v). The 14B Wan MoE
/// engines have no `Keyframe` path so they stay torch. `replace_person` is MLX on the
/// replace-capable models (→ native Wan-VACE, sc-3521).
fn video_mode_is_mlx_eligible(model: &str, mode: &str) -> bool {
    if model == "svd" {
        return mode == "image_to_video";
    }
    // Bernini's renderer is Wan2.2-T2V (text-conditioned) — it has no classic
    // still-image-to-video. Beyond `text_to_video` (sc-4707) it serves the planner's
    // editing + reference-driven video tasks (sc-4703): `video_to_video` (v2v — a
    // source-clip edit, `Conditioning::VideoClip`), `reference_to_video` (r2v —
    // subject reference images, `MultiReference`), and `reference_video_to_video`
    // (rv2v — source clip + reference images); plus the multi-source modes (sc-5425):
    // `multi_video_to_video` (mv2v — several source clips) and `ads2v` (source video +
    // reference video + reference images). The engine selects the matching guidance
    // mode from `video_mode` + the supplied conditioning.
    if model == "bernini" {
        return matches!(
            mode,
            "text_to_video"
                | "video_to_video"
                | "reference_to_video"
                | "reference_video_to_video"
                | "multi_video_to_video"
                | "ads2v"
        );
    }
    // SCAIL-2 (epic 5439) is a Wan2.1-14B I2V character-animation engine: a reference character
    // image + a driving video → an animated clip. It serves the standalone `animate_character` mode
    // (sc-5448, the worker paints the color-coded masks from native SAM3) AND cross-identity
    // `replace_person` (sc-5452, the integrated backend behind the YOLO11 → ByteTrack → SAM3
    // person-track pipeline). Both run the same engine; `replace_person` flips the engine
    // `replace_flag`. It has no classic text/image-to-video.
    if model == "scail2_14b" {
        return matches!(mode, "animate_character" | "replace_person");
    }
    match mode {
        "text_to_video" | "image_to_video" => true,
        "first_last_frame" => matches!(model, "ltx_2_3" | "ltx_2_3_eros" | "wan_2_2"),
        // extend_clip / video_bridge: LTX via the IC-LoRA multi-frame keyframe-append (sc-3522),
        // and Wan (`wan_2_2`) — the worker prefers native Wan-VACE ControlClip for genuine motion
        // continuity (sc-3812, tier C: real source frames pinned at the kept positions + a
        // generated-span mask) and falls back to the TI2V-5B single-frame boundary keyframe path
        // (sc-3357) when the VACE snapshot is unprovisioned. Both run MLX-native, so `wan_2_2` is
        // eligible regardless of which the worker picks. The 14B Wan MoE engines have neither
        // path, so extend/bridge on them stay torch.
        "extend_clip" | "video_bridge" => matches!(model, "ltx_2_3" | "ltx_2_3_eros" | "wan_2_2"),
        // replace_person → native Wan-VACE (sc-3521): the engine `wan_vace` provider serves it
        // regardless of the user-picked replace-capable model (ltx_2_3 / ltx_2_3_eros / wan_2_2,
        // the models that advertise the capability), so admit those.
        "replace_person" => matches!(model, "ltx_2_3" | "ltx_2_3_eros" | "wan_2_2"),
        _ => false,
    }
}

/// SceneWorks training kernels with a native mlx-gen Rust trainer (epic 3039):
/// the engine registers `z_image_turbo`/`sdxl`/`kolors`/`ltx_2_3`/`wan2_2_*` trainers,
/// which the worker reaches via these SceneWorks kernel ids (the mlx worker maps the
/// kernel and base model onto an engine trainer id). `kolors_lora` (SDXL U-Net plus
/// ChatGLM3) gained a native trainer in sc-4568, cut over here in sc-4732. `lens_lora`
/// gained a native mlx-gen-lens trainer in sc-5148, cut over here in sc-5180 (off-Mac
/// keeps the Python sidecar trainer). A kernel absent here is never routed to the mlx worker.
const MLX_ROUTED_TRAINING_KERNELS: &[&str] = &[
    "z_image_lora",
    "sdxl_lora",
    "kolors_lora",
    "lens_lora",
    "wan_lora",
    "wan_moe_lora",
    "ltx_mlx_lora",
];

/// Epic 3039 routing — does this `lora_train` job belong on the in-process Rust MLX
/// worker (vs the Python torch worker)? The training sibling of
/// [`image_job_is_mlx_eligible`]/[`video_job_is_mlx_eligible`]: the engine has a
/// native trainer for the family. Both dry-run and real runs are eligible (the
/// dry-run validates the same resolved plan). LoKr-on-Wan stays torch — the mlx Wan
/// inference path can't load a Kronecker adapter, mirroring [`video_job_is_mlx_eligible`];
/// LoKr on Z-Image/SDXL/LTX is fine (the Rust engine applies it natively).
///
/// The resolved plan is stamped into the job payload at submit (apps/rust-api
/// training.rs) for both dry-run and real runs, so the kernel + network type are
/// readable here without touching the dataset or weights.
fn training_job_is_mlx_eligible(job: &JobSnapshot) -> bool {
    if !matches!(job.job_type, JobType::LoraTrain) {
        return false;
    }
    let Some(plan) = job.payload.get("plan").and_then(Value::as_object) else {
        return false;
    };
    let kernel = plan
        .get("target")
        .and_then(Value::as_object)
        .and_then(|target| target.get("kernel"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    if !MLX_ROUTED_TRAINING_KERNELS.contains(&kernel) {
        return false;
    }
    // LoKr-on-Wan stays on the torch path (no Kronecker merge in the mlx Wan path).
    if matches!(kernel, "wan_lora" | "wan_moe_lora") && training_plan_is_lokr(plan) {
        return false;
    }
    true
}

/// sc-3556 routing: SceneWorks training caption jobs keep their public
/// `captioner=joy_caption` contract while the macOS mlx worker serves them through
/// mlx-gen's JoyCaption provider. Other/unknown captioners stay off the mlx worker.
fn caption_job_is_mlx_eligible(job: &JobSnapshot) -> bool {
    matches!(job.job_type, JobType::TrainingCaption)
        && job
            .payload
            .get("captioner")
            .and_then(Value::as_str)
            .is_some_and(|value| value.trim() == "joy_caption")
}

/// Whether an `image_upscale` job runs on the Rust/MLX path (epic 3482, sc-3489): the
/// Real-ESRGAN (RRDBNet) engine — the default — is ported to the Rust worker, and `seedvr2`
/// (the native-MLX one-step diffusion upscaler, epic 4811 / sc-4815) runs in-process via
/// `mlx-gen-seedvr2`. `aura-sr` (a 617M-param torch-only GigaGAN) was dropped on Mac after the
/// sc-3668 port-or-drop spike, so the mlx worker refuses it (it runs on the Python worker on
/// Windows/Linux). Engine defaults to `real-esrgan` when absent (mirrors `run_image_upscale`).
/// SeedVR2 is Mac-only here (a Windows/Linux Candle backend is the separate sc-5157); the Mac UI
/// gating + `imageUpscaleSeedvr2` capability keep it off non-Mac pickers.
fn upscale_job_is_mlx_eligible(job: &JobSnapshot) -> bool {
    if !matches!(job.job_type, JobType::ImageUpscale) {
        return false;
    }
    let engine = job
        .payload
        .get("engine")
        .and_then(Value::as_str)
        .map(|value| value.trim().to_ascii_lowercase())
        .unwrap_or_else(|| "real-esrgan".to_owned());
    matches!(
        engine.as_str(),
        "" | "real-esrgan" | "realesrgan" | "real_esrgan" | "seedvr2"
    )
}

/// Whether a `video_upscale` job is MLX-eligible (epic 4811 / sc-4816). The only Mac engine is the
/// native-MLX SeedVR2 upscaler (`mlx-gen-seedvr2`); there is no torch fallback (mac-only). A job with
/// any other engine is refused by the mlx worker — though no other backend advertises `video_upscale`
/// today, so an unsupported engine simply has nowhere to run (surfaced as unsupported, not silently
/// dropped). Defaults to `seedvr2` when the payload omits the engine.
fn video_upscale_job_is_mlx_eligible(job: &JobSnapshot) -> bool {
    if !matches!(job.job_type, JobType::VideoUpscale) {
        return false;
    }
    let engine = job
        .payload
        .get("engine")
        .and_then(Value::as_str)
        .map(|value| value.trim().to_ascii_lowercase())
        .unwrap_or_else(|| "seedvr2".to_owned());
    matches!(engine.as_str(), "" | "seedvr2" | "seedvr2_3b")
}

/// Training kernels with NO non-Rust fallback — only the in-process Rust mlx worker
/// can run them. `ltx_mlx_lora` was Apple-Silicon-only MLX-Python; epic 3039 (sc-3049)
/// retired that Python trainer, leaving the native Rust LTX trainer as the sole path,
/// so a non-mlx worker must refuse the job (leaving it queued for the mlx worker)
/// rather than claim it and fail with "no training kernel". The torch families
/// (z-image/sdxl/wan) keep their Python trainer as the Windows path + Mac fallback, so
/// they are deliberately NOT listed here.
const MLX_ONLY_TRAINING_KERNELS: &[&str] = &["ltx_mlx_lora"];

/// Whether this `lora_train` job targets a kernel with no non-Rust fallback (see
/// [`MLX_ONLY_TRAINING_KERNELS`]). Such a job can only run on the mlx worker.
fn training_kernel_is_mlx_only(job: &JobSnapshot) -> bool {
    if !matches!(job.job_type, JobType::LoraTrain) {
        return false;
    }
    job.payload
        .get("plan")
        .and_then(Value::as_object)
        .and_then(|plan| plan.get("target"))
        .and_then(Value::as_object)
        .and_then(|target| target.get("kernel"))
        .and_then(Value::as_str)
        .is_some_and(|kernel| MLX_ONLY_TRAINING_KERNELS.contains(&kernel))
}

/// Whether a resolved training plan requests a LoKr (Kronecker) adapter. The network
/// type lives in the plan's `config.advanced.networkType` (SceneWorks training
/// contract), distinct from a generation request's per-LoRA `networkType`.
fn training_plan_is_lokr(plan: &Map<String, Value>) -> bool {
    plan.get("config")
        .and_then(Value::as_object)
        .and_then(|config| config.get("advanced"))
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("networkType"))
        .and_then(Value::as_str)
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("lokr"))
}

fn worker_supports_job(worker: &WorkerSnapshot, job: &JobSnapshot) -> bool {
    if job_requires_gpu(&job.job_type) && worker.gpu_id.eq_ignore_ascii_case("cpu") {
        return false;
    }
    // Epic 3039 (sc-3049): a training kernel with no torch fallback (the retired Python
    // MLX LTX trainer) runs only on the mlx worker — a non-mlx worker must refuse it
    // (leaving it queued for the mlx worker) instead of claiming it and failing.
    if !worker.gpu_id.eq_ignore_ascii_case("mlx") && training_kernel_is_mlx_only(job) {
        return false;
    }
    // Epic 3018/3041 + sc-3036: the in-process MLX worker (gpu_id "mlx") serves a fixed
    // set of model families. It must not claim a job that needs the torch path — a family
    // not yet ported, an unsupported shape, or a third-party LyCORIS LoRA — those stay on
    // the Python worker. Non-mlx workers are unaffected here; the *preference* to route
    // eligible jobs to an idle mlx worker is a soft deferral in the claim path.
    if worker.gpu_id.eq_ignore_ascii_case("mlx") {
        // Image: sc-3026 txt2img/LoRA + sc-3060 reference/edit/inpaint/outpaint +
        // image_detail + sc-3513 the `image_edit` job type (plain Image Edit). A
        // torch-only edit model (kolors/lens/pulid) is not MLX-eligible, so the mlx
        // worker refuses it and it stays on torch. (z_image_edit was ported to MLX,
        // epic 3529 / sc-3923; instantid + sensenova are MLX-routed too.)
        if matches!(
            job.job_type,
            JobType::ImageGenerate | JobType::ImageEdit | JobType::ImageDetail
        ) && !job_is_mlx_eligible(job)
        {
            return false;
        }
        // Video (sc-3036 + the epic-3040 cutover): the mlx worker claims MLX-eligible
        // `video_generate` jobs (Wan/LTX text_to_video / image_to_video + SVD
        // image_to_video) plus the advanced job types now ported to the Rust engine —
        // `first_last_frame` (LTX + Wan TI2V-5B, sc-3520), `extend_clip` / `video_bridge`
        // (LTX IC-LoRA, sc-3522), and `person_replace` → native Wan-VACE (sc-3521). The
        // per-(model, mode) gate in `video_job_is_mlx_eligible` keeps each mode to its
        // capable engines; everything it rejects — a non-MLX model, Wan extend/bridge
        // (no IC-LoRA keyframe-append path), LoKr-on-Wan — stays on the Python worker.
        if matches!(
            job.job_type,
            JobType::VideoGenerate
                | JobType::VideoExtend
                | JobType::VideoBridge
                | JobType::PersonReplace
        ) && !video_job_is_mlx_eligible(job)
        {
            return false;
        }
        // Training (epic 3039): the mlx worker trains only the MLX-native families
        // (z_image / sdxl / kolors / wan / ltx) via `mlx_gen::load_trainer`. `lens_lora`
        // (sidecar, no mlx-gen crate) and LoKr-on-Wan stay on the Python torch worker.
        // Applies to both dry-run and real runs.
        if matches!(job.job_type, JobType::LoraTrain) && !training_job_is_mlx_eligible(job) {
            return false;
        }
        // Dataset captioning (sc-3556): the mlx worker claims only JoyCaption jobs
        // backed by the mlx-gen provider. Any future non-JoyCaption captioner stays
        // on the worker that advertises that capability.
        if matches!(job.job_type, JobType::TrainingCaption) && !caption_job_is_mlx_eligible(job) {
            return false;
        }
        // Image upscale (sc-3489): the mlx worker runs Real-ESRGAN (the default engine) via
        // `ort`/CoreML and SeedVR2 via in-process `mlx-gen-seedvr2` (sc-4815). `aura-sr` has no
        // Rust path, so the mlx worker refuses it and it stays on the Python torch worker.
        if matches!(job.job_type, JobType::ImageUpscale) && !upscale_job_is_mlx_eligible(job) {
            return false;
        }
        // Video upscale (epic 4811 / sc-4816): the mlx worker runs the native SeedVR2 engine
        // (`mlx-gen-seedvr2`). Any non-SeedVR2 engine is refused; since there is no torch
        // video-upscale backend, this is mac-only by construction.
        if matches!(job.job_type, JobType::VideoUpscale) && !video_upscale_job_is_mlx_eligible(job)
        {
            return false;
        }
        // SenseNova-U1 understanding (sc-3905): the mlx worker serves `image_vqa` /
        // `image_interleave` only for the SenseNova-U1 ids (the sole in-process understanding
        // path). A non-SenseNova understanding job is not MLX-eligible, so the mlx worker
        // refuses it and it stays on the Python torch worker.
        if matches!(job.job_type, JobType::ImageVqa | JobType::ImageInterleave)
            && !understanding_job_is_mlx_eligible(job)
        {
            return false;
        }
    }
    // No-silent-T2I / no-torch-fallback (sc-5968, epic 5483): the co-resident Python torch worker (a
    // non-candle, non-mlx GPU worker) must DECLINE the unsupported-pose shapes the candle worker
    // owns-to-reject (a `advanced.poses` job on a candle model with no pose lane, e.g. sdxl) — so torch
    // can't claim + silently render an unconditioned T2I image, and the candle worker reliably wins
    // them (then rejects with a typed error). Mac is unaffected: those shapes are MLX-served there
    // (model_mac_support pose), so the `mlx` worker still claims them and only torch/`mps` declines.
    if !worker_is_candle(worker)
        && !worker.gpu_id.eq_ignore_ascii_case("mlx")
        && image_job_candle_pose_reject(job)
    {
        return false;
    }
    // Candle (Windows/CUDA) lane (epic 3672 image sc-3678; epic 5095 image families sc-5096 + video
    // sc-5097): the candle worker advertises `image_generate` (+ `video_generate` once video engines
    // are wired) and serves gated, narrow **txt2img / txt2video-only** lanes. It must refuse every
    // other shape — a non-candle family, or a conditioned (img2img/edit/reference/inpaint/pose/
    // i2v/extend/bridge/replace) / LoRA request — so those transparently fall back to the Python torch
    // worker that co-resides on the box. Identified by the `candle` marker capability (not `gpu_id`,
    // which is a real CUDA index here). When candle is disabled the marker is absent and this is inert,
    // so production routing is unchanged until the lane is turned on.
    if worker_is_candle(worker) {
        // ImageGenerate: claim the candle-served shapes AND the unsupported-pose shapes the candle
        // worker must OWN to reject (a `advanced.poses` job on a candle model with no pose lane, e.g.
        // sdxl) — so those fail loudly on candle instead of falling back to torch + silently rendering
        // an unconditioned T2I image (sc-5968, the no-torch-fallback / no-silent-T2I directive). Every
        // other shape candle declines, staying on the co-resident torch worker.
        if matches!(job.job_type, JobType::ImageGenerate)
            && !(image_job_is_candle_eligible(job) || image_job_candle_pose_reject(job))
        {
            return false;
        }
        // The candle worker advertises only the base `video_generate` (txt2video); refuse the
        // advanced video job types and every non-eligible `video_generate` shape.
        if matches!(
            job.job_type,
            JobType::VideoGenerate
                | JobType::VideoExtend
                | JobType::VideoBridge
                | JobType::PersonReplace
        ) && !video_job_is_candle_eligible(job)
        {
            return false;
        }
        // Dataset captioning (sc-5098): the candle worker serves only JoyCaption (the candle
        // captioner provider). A non-`joy_caption` caption job stays on the Python torch worker.
        // Eligibility is backend-neutral (captioner == joy_caption), so reuse the mlx gate.
        if matches!(job.job_type, JobType::TrainingCaption) && !caption_job_is_mlx_eligible(job) {
            return false;
        }
        // SenseNova-U1 understanding (sc-5501): the candle worker serves `image_vqa` /
        // `image_interleave` only for the SenseNova-U1 ids (via the concrete candle `T2iModel::{vqa,
        // interleave_gen}` — the off-Mac sibling of the MLX understanding path). Eligibility is
        // backend-neutral (the model is SenseNova-U1), so reuse the understanding gate; a
        // non-SenseNova understanding job stays on the Python torch worker.
        if matches!(job.job_type, JobType::ImageVqa | JobType::ImageInterleave)
            && !understanding_job_is_mlx_eligible(job)
        {
            return false;
        }
    }
    let advertises = |capability: &str| {
        worker
            .capabilities
            .iter()
            .any(|owned| owned.as_str() == capability)
    };
    if !advertises(required_capability(job)) {
        return false;
    }
    // A real (non-dry-run) LoRA training job additionally needs the execute
    // capability, which a worker advertises only when its inference backend is
    // available. Dry-run plan validation needs just the base `lora_train`
    // capability. This keeps a real run queued for a capable worker instead of
    // failing terminally after a torch-less worker claims it.
    if is_real_training_job(job) {
        return advertises(WorkerCapability::LoraTrainExecute.as_str());
    }
    true
}

/// True when a job is a real (non-dry-run) LoRA training run. The training
/// payload defaults to dry-run; only an explicit `dryRun: false` is a real run.
fn is_real_training_job(job: &JobSnapshot) -> bool {
    matches!(job.job_type, JobType::LoraTrain)
        && job.payload.get("dryRun").and_then(Value::as_bool) == Some(false)
}

/// The worker capability a job requires. Person detection/tracking default to
/// the real, model-backed capability served by the Python GPU worker; an
/// explicit `preview: true` payload requests the Rust utility worker's
/// procedural preview capability instead — so a real job never routes to the
/// placeholder. Mirrors the dry-run training capability split.
fn required_capability(job: &JobSnapshot) -> &str {
    match job.job_type {
        JobType::PersonDetect if person_job_is_preview(job) => {
            WorkerCapability::PersonDetectPreview.as_str()
        }
        JobType::PersonTrack if person_job_is_preview(job) => {
            WorkerCapability::PersonTrackPreview.as_str()
        }
        _ => job.job_type.as_str(),
    }
}

/// True when a person detection/tracking job explicitly opts into the procedural
/// preview path (`preview: true`); real model-backed runs are the default.
fn person_job_is_preview(job: &JobSnapshot) -> bool {
    matches!(job.job_type, JobType::PersonDetect | JobType::PersonTrack)
        && job.payload.get("preview").and_then(Value::as_bool) == Some(true)
}

#[derive(Debug, Clone, Copy)]
struct DispatchScore {
    has_utilization: bool,
    free_memory_mb: f64,
    memory_usage_percent: f64,
    gpu_load_percent: f64,
    warm_model: bool,
}

fn dispatch_score(job: &JobSnapshot, worker: &WorkerSnapshot) -> DispatchScore {
    let utilization = worker.utilization.as_ref();
    let total = utilization.and_then(|item| item.memory_total_mb);
    let used = utilization.and_then(|item| item.memory_used_mb);
    let gpu_load = utilization.and_then(|item| item.gpu_load_percent);
    // Derive free memory only from data the worker actually reported: an explicit
    // free reading, or total-minus-used when both are present. A worker that
    // reports no utilization at all must stay `has_utilization = false` so the
    // auto-GPU dispatcher leaves it alone — the earlier `total.checked_sub(used)`
    // with total/used defaulted to 0 yielded `Some(0)`, which scored a
    // no-utilization worker as a real GPU with 0 MB free. That made the
    // Apple-Silicon `mlx` worker (whose nvidia-smi probe finds nothing, so it
    // never reports utilization) always look "worse" than the idle Python `mps`
    // worker, so it deferred every MLX-eligible job to `mps` — which deferred the
    // same job right back to `mlx` (`should_defer_image_to_mlx_worker`), leaving
    // it queued on "Waiting for an available worker" forever (sc-3289 regression).
    let free = utilization
        .and_then(|item| item.memory_free_mb)
        .or_else(|| match (total, used) {
            (Some(total), Some(used)) => total.checked_sub(used),
            _ => None,
        });
    let memory_usage_percent = match (total, used) {
        (Some(total), Some(used)) if total > 0 => used as f64 / total as f64 * 100.0,
        _ => 0.0,
    };
    DispatchScore {
        has_utilization: free.is_some() || gpu_load.is_some() || total.is_some(),
        free_memory_mb: free.unwrap_or(0) as f64,
        memory_usage_percent,
        gpu_load_percent: gpu_load.unwrap_or(0.0),
        warm_model: job_matches_loaded_model(job, worker),
    }
}

fn dispatch_score_is_better(candidate: DispatchScore, current: DispatchScore) -> bool {
    if !candidate.has_utilization || !current.has_utilization {
        return false;
    }

    let free_delta = candidate.free_memory_mb - current.free_memory_mb;
    let load_delta = current.gpu_load_percent - candidate.gpu_load_percent;
    let usage_delta = current.memory_usage_percent - candidate.memory_usage_percent;
    // Prefer a meaningfully freer/lower-load GPU, with tolerance bands so two
    // similarly healthy GPUs do not trade claims back and forth on tiny deltas.
    let candidate_is_not_worse = candidate.free_memory_mb + DISPATCH_MEMORY_NOT_WORSE_TOLERANCE_MB
        >= current.free_memory_mb
        && candidate.gpu_load_percent
            <= current.gpu_load_percent + DISPATCH_LOAD_NOT_WORSE_TOLERANCE_PERCENT
        && candidate.memory_usage_percent
            <= current.memory_usage_percent + DISPATCH_MEMORY_USAGE_NOT_WORSE_TOLERANCE_PERCENT;
    let candidate_relief = free_delta >= DISPATCH_MEMORY_RELIEF_THRESHOLD_MB
        || load_delta >= DISPATCH_LOAD_RELIEF_THRESHOLD_PERCENT
        || usage_delta >= DISPATCH_MEMORY_USAGE_RELIEF_THRESHOLD_PERCENT;

    if candidate_is_not_worse && candidate_relief {
        return true;
    }
    if candidate_is_not_worse && candidate.warm_model && !current.warm_model {
        return true;
    }
    (current.free_memory_mb < DISPATCH_LOW_MEMORY_THRESHOLD_MB
        && candidate.free_memory_mb >= DISPATCH_HEALTHY_MEMORY_THRESHOLD_MB)
        || (current.gpu_load_percent >= DISPATCH_HIGH_LOAD_THRESHOLD_PERCENT
            && candidate.gpu_load_percent <= DISPATCH_RECOVERED_LOAD_THRESHOLD_PERCENT)
        || (current.memory_usage_percent >= DISPATCH_HIGH_MEMORY_USAGE_THRESHOLD_PERCENT
            && candidate.memory_usage_percent <= DISPATCH_RECOVERED_MEMORY_USAGE_THRESHOLD_PERCENT)
}

fn choose_claimable_job(rows: Vec<JobSnapshot>, worker: &WorkerSnapshot) -> Option<JobSnapshot> {
    let compatible = rows
        .into_iter()
        .filter(|job| worker_supports_job(worker, job))
        .collect::<Vec<_>>();
    let first = compatible.first()?;
    if is_non_gpu_job_type(first.job_type.as_str()) || first.requested_gpu != "auto" {
        return compatible.into_iter().next();
    }
    if let Some(explicit_gpu_job) = compatible
        .iter()
        .find(|job| !is_non_gpu_job_type(job.job_type.as_str()) && job.requested_gpu != "auto")
        .cloned()
    {
        return Some(explicit_gpu_job);
    }
    compatible
        .iter()
        .find(|job| job_matches_loaded_model(job, worker))
        .cloned()
        .or_else(|| compatible.into_iter().next())
}

fn job_matches_loaded_model(job: &JobSnapshot, worker: &WorkerSnapshot) -> bool {
    if job.requested_gpu != "auto"
        || is_non_gpu_job_type(job.job_type.as_str())
        || worker.loaded_models.is_empty()
    {
        return false;
    }
    let keys = desired_model_keys(&job.payload);
    worker
        .loaded_models
        .iter()
        .any(|loaded_model| keys.iter().any(|key| key == loaded_model))
}

fn desired_model_keys(payload: &Map<String, Value>) -> Vec<String> {
    let mut keys = Vec::new();
    push_string_value(&mut keys, payload.get("model"));
    push_string_value(&mut keys, payload.get("repo"));
    if let Some(advanced) = payload.get("advanced").and_then(Value::as_object) {
        push_string_value(&mut keys, advanced.get("modelRepo"));
        push_string_value(&mut keys, advanced.get("repo"));
    }
    keys.sort();
    keys.dedup();
    keys
}

fn push_string_value(output: &mut Vec<String>, value: Option<&Value>) {
    if let Some(value) = value
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        output.push(value.to_owned());
    }
}

fn normalize_requested_gpu(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        "auto".to_owned()
    } else if trimmed.eq_ignore_ascii_case("auto") || trimmed.eq_ignore_ascii_case("cpu") {
        trimmed.to_ascii_lowercase()
    } else {
        trimmed.to_owned()
    }
}

// Keep GPU-required job types in sync with
// apps/worker/scene_worker/runtime.py (SUPPORTED_JOB_TYPES + TRAINING_JOB_TYPES +
// CAPTION_JOB_TYPES) and apps/web/src/screens/QueueScreen.jsx::gpuRequiredJobTypes.
// `lora_train` is GPU-required like generation, but its worker capability is
// advertised separately (the dry-run plan validation needs no inference backend;
// real execution is gated per platform in story 1417).
fn job_requires_gpu(job_type: &JobType) -> bool {
    matches!(
        job_type,
        JobType::ImageGenerate
            | JobType::ImageEdit
            | JobType::ImageVqa
            | JobType::ImageInterleave
            | JobType::ImageUpscale
            | JobType::ImageDetail
            | JobType::VideoGenerate
            | JobType::VideoExtend
            | JobType::VideoBridge
            | JobType::VideoUpscale
            | JobType::PersonReplace
            | JobType::LoraTrain
            | JobType::TrainingCaption
    )
}

fn placeholders_from(start: usize, count: usize) -> String {
    (start..start + count)
        .map(|index| format!("?{index}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn sort_json_value(value: &mut Value) {
    match value {
        Value::Object(map) => {
            let mut entries = map
                .iter_mut()
                .map(|(key, value)| {
                    sort_json_value(value);
                    (key.clone(), value.clone())
                })
                .collect::<Vec<_>>();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            map.clear();
            map.extend(entries);
        }
        Value::Array(items) => {
            for item in items {
                sort_json_value(item);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
}

#[cfg(test)]
mod active_statuses_sql_tests {
    use super::{active_statuses_sql, ACTIVE_STATUSES};

    /// Anti-drift guard for sc-4207 / F-CORE-3: the five `status in (...)` SQL
    /// statements now interpolate [`active_statuses_sql`] instead of a
    /// copy-pasted literal, so the generated list must stay exactly in sync with
    /// [`ACTIVE_STATUSES`] — every status quoted, comma-separated, none dropped.
    #[test]
    fn sql_list_matches_active_statuses_const() {
        let expected = ACTIVE_STATUSES
            .iter()
            .map(|status| format!("'{status}'"))
            .collect::<Vec<_>>()
            .join(", ");
        assert_eq!(active_statuses_sql(), expected);

        // Each status appears as a quoted token, guarding against a future const
        // edit that silently fails to reach the SQL filters.
        for status in ACTIVE_STATUSES {
            assert!(
                active_statuses_sql().contains(&format!("'{status}'")),
                "active status {status:?} missing from SQL list"
            );
        }
    }
}

#[cfg(test)]
mod signal_failure_error_tests {
    //! sc-4881 signal attribution + sc-5567 job-kind-aware OOM remediation: a signal-9
    //! (SIGKILL/OOM) death must give guidance that fits the dead job — count/resolution
    //! for an image batch, frames for video, gradient checkpointing only for training —
    //! and non-OOM uncatchable deaths must keep naming their real cause.
    use super::{signal_failure_error, JobType};

    #[test]
    fn signal_9_image_batch_points_at_count_not_gradient_checkpointing() {
        let msg = signal_failure_error(9, Some(&JobType::ImageGenerate));
        assert!(msg.contains("signal 9 (SIGKILL)"), "{msg}");
        assert!(msg.contains("out-of-memory"), "{msg}");
        assert!(msg.contains("image count or resolution"), "{msg}");
        // The old training-only hint must NOT leak onto an image batch (the sc-5567 bug).
        assert!(!msg.contains("Gradient Checkpointing"), "{msg}");
        assert!(!msg.contains("training step"), "{msg}");
    }

    #[test]
    fn signal_9_training_keeps_gradient_checkpointing_hint() {
        let msg = signal_failure_error(9, Some(&JobType::LoraTrain));
        assert!(msg.contains("Gradient Checkpointing"), "{msg}");
        assert!(msg.contains("training step"), "{msg}");
    }

    #[test]
    fn signal_9_video_points_at_frame_count() {
        let msg = signal_failure_error(9, Some(&JobType::VideoGenerate));
        assert!(msg.contains("out-of-memory"), "{msg}");
        assert!(msg.contains("frame count"), "{msg}");
        assert!(!msg.contains("Gradient Checkpointing"), "{msg}");
    }

    #[test]
    fn signal_9_unknown_and_idle_fall_back_to_generic_oom() {
        // No active job (worker died idle) and an unmapped job kind both get the generic
        // OOM hint rather than a misleading training/image/video-specific one.
        for job_type in [None, Some(&JobType::Unknown("future".to_owned()))] {
            let msg = signal_failure_error(9, job_type);
            assert!(msg.contains("out-of-memory"), "{msg}");
            assert!(!msg.contains("Gradient Checkpointing"), "{msg}");
            assert!(!msg.contains("image count"), "{msg}");
            assert!(!msg.contains("frame count"), "{msg}");
        }
    }

    #[test]
    fn non_oom_signals_keep_their_own_cause_regardless_of_job_kind() {
        // SIGABRT / SIGSEGV are not OOM, so the job kind must not turn them into one.
        let abort = signal_failure_error(6, Some(&JobType::ImageGenerate));
        assert!(abort.contains("signal 6 (SIGABRT)"), "{abort}");
        assert!(abort.contains("GPU/Metal command-buffer abort"), "{abort}");
        assert!(!abort.contains("out-of-memory"), "{abort}");

        let segv = signal_failure_error(11, Some(&JobType::LoraTrain));
        assert!(segv.contains("signal 11 (SIGSEGV)"), "{segv}");
        assert!(segv.contains("segmentation fault"), "{segv}");
        assert!(!segv.contains("Gradient Checkpointing"), "{segv}");
    }
}

#[cfg(test)]
mod candle_routing_tests {
    //! Candle (Windows/CUDA) SDXL lane routing (epic 3672, sc-3678): the candle worker serves a
    //! gated, narrow SDXL/RealVisXL **txt2img-only** lane and must defer every other shape to the
    //! Python torch worker. These tests pin the lane boundary (`image_request_candle_eligible`) and
    //! the full claim gate (`worker_supports_job` via the `candle` marker capability).
    use super::*;
    use serde_json::{json, Value};

    fn object(value: Value) -> Map<String, Value> {
        value.as_object().expect("test value is an object").clone()
    }

    /// A queued `image_generate` job carrying `payload`, built via serde so the test never has to
    /// spell out the full `JobSnapshot` field set.
    fn image_generate_job(payload: Value) -> JobSnapshot {
        serde_json::from_value(json!({
            "id": "job_1",
            "type": "image_generate",
            "status": "queued",
            "payload": payload,
            "result": {},
            "requestedGpu": "auto",
            "progress": 0,
            "stage": "queued",
            "message": "",
            "attempts": 1,
            "cancelRequested": false,
            "createdAt": "2026-06-12T00:00:00Z",
            "updatedAt": "2026-06-12T00:00:00Z",
        }))
        .expect("valid JobSnapshot")
    }

    /// A worker on a real CUDA gpu index advertising `capabilities` (string ids). The candle worker
    /// carries the `candle` marker; the torch worker on the same box does not.
    fn gpu_worker(capabilities: &[&str]) -> WorkerSnapshot {
        serde_json::from_value(json!({
            "id": "worker_1",
            "gpuId": "0",
            "status": "idle",
            "capabilities": capabilities,
            "loadedModels": [],
            "registeredAt": "2026-06-12T00:00:00Z",
            "lastSeenAt": "2026-06-12T00:00:00Z",
        }))
        .expect("valid WorkerSnapshot")
    }

    const CANDLE_CAPS: &[&str] = &["gpu", "image_generate", "candle"];
    // The Python torch worker advertises the broad image surface but no `candle` marker.
    const TORCH_CAPS: &[&str] = &["gpu", "image_generate", "image_edit", "image_detail"];

    #[test]
    fn candle_routed_models_plain_txt2img_are_eligible() {
        // SDXL/RealVisXL (sc-3678) + the four image families wired in sc-5096 — every base txt2img id.
        for model in CANDLE_ROUTED_MODELS {
            assert!(
                image_request_candle_eligible(model, &object(json!({ "prompt": "a red fox" }))),
                "{model} plain txt2img should be candle-eligible"
            );
        }
    }

    #[test]
    fn non_candle_families_and_variants_are_never_candle_eligible() {
        // A family with no candle provider at all (`bernini_image`) AND the non-base weight/shape
        // variants of wired families (edit ids, the kv distill) all stay on the Python torch worker.
        // (chroma / kolors / sensenova ARE candle-routed now — sc-5484 / sc-5576 — for txt2img.)
        for model in [
            "bernini_image",
            "z_image_edit",
            "qwen_image_edit",
            "flux2_klein_9b_kv",
        ] {
            assert!(
                !image_request_candle_eligible(model, &object(json!({ "prompt": "p" }))),
                "{model} must fall back to the Python worker"
            );
        }
    }

    #[test]
    fn new_candle_families_conditioning_shapes_fall_back_to_torch() {
        // Every candle image family is txt2img-only on candle: any conditioning shape defers to torch
        // (the worker advertises none of these, so this is the no-silently-dropped-control boundary).
        let cases = [
            (
                "z_image_turbo",
                json!({ "mode": "edit_image", "sourceAssetId": "a" }),
            ),
            ("flux_dev", json!({ "referenceAssetId": "a" })),
            ("flux_schnell", json!({ "loras": [{ "name": "x" }] })),
            (
                "qwen_image",
                json!({ "advanced": { "poses": [{ "id": "pose_1" }] } }),
            ),
            (
                "flux2_klein_9b",
                json!({ "mode": "edit_image", "sourceAssetId": "a" }),
            ),
            // sc-5484 / sc-5576: Chroma / Kolors / SenseNova-U1 are pure T2I on candle. Their MLX-only
            // conditioning shapes (Kolors edit / IP-reference / pose-control; SenseNova edit) defer.
            (
                "chroma1_hd",
                json!({ "mode": "edit_image", "sourceAssetId": "a" }),
            ),
            (
                "kolors",
                json!({ "mode": "edit_image", "sourceAssetId": "a" }),
            ),
            ("kolors", json!({ "referenceAssetId": "a" })),
            (
                "kolors",
                json!({ "advanced": { "poses": [{ "id": "pose_1" }] } }),
            ),
            (
                "sensenova_u1_8b",
                json!({ "mode": "edit_image", "sourceAssetId": "a" }),
            ),
            ("sensenova_u1_8b_fast", json!({ "referenceAssetId": "a" })),
        ];
        for (model, payload) in cases {
            assert!(
                !image_request_candle_eligible(model, &object(payload.clone())),
                "{model} conditioning shape must fall back to torch: {payload}"
            );
        }
    }

    #[test]
    fn explicit_quantization_falls_back_to_torch_image_and_video() {
        // sc-5099: the candle providers are dense (supported_quants: &[]); an explicit
        // `advanced.mlxQuantize > 0` must route to Python rather than silently running dense.
        assert!(!image_request_candle_eligible(
            "sdxl",
            &object(json!({ "advanced": { "mlxQuantize": 8 } }))
        ));
        assert!(!image_request_candle_eligible(
            "qwen_image",
            &object(json!({ "advanced": { "mlxQuantize": 4 } }))
        ));
        assert!(!video_request_candle_eligible(
            "wan_2_2",
            &object(json!({ "mode": "text_to_video", "advanced": { "mlxQuantize": 8 } }))
        ));
        // Dense (<= 0) or absent quant leaves candle on its native dense path → still eligible.
        assert!(image_request_candle_eligible(
            "sdxl",
            &object(json!({ "advanced": { "mlxQuantize": 0 } }))
        ));
        assert!(image_request_candle_eligible(
            "sdxl",
            &object(json!({ "advanced": { "steps": 30 } }))
        ));
    }

    #[test]
    fn lens_quant_and_lora_stay_on_the_candle_lane() {
        // sc-5126: Lens / Lens-Turbo advertise Q4/Q8 + LoRA/LoKr, so — UNLIKE the sc-3675/sc-5096
        // families — a quant request or a LoRA does NOT defer to torch; the candle lane maps both into
        // the LoadSpec.
        for model in ["lens", "lens_turbo"] {
            assert!(
                image_request_candle_eligible(
                    model,
                    &object(json!({ "advanced": { "mlxQuantize": 8 } }))
                ),
                "{model} Q8 request should stay on candle"
            );
            assert!(
                image_request_candle_eligible(
                    model,
                    &object(json!({ "advanced": { "mlxQuantize": 4 } }))
                ),
                "{model} Q4 request should stay on candle"
            );
            assert!(
                image_request_candle_eligible(
                    model,
                    &object(json!({ "loras": [{ "name": "x", "path": "/x.safetensors" }] }))
                ),
                "{model} with a LoRA should stay on candle"
            );
        }
    }

    #[test]
    fn lens_conditioning_shapes_fall_back_to_torch() {
        // Lens is pure T2I (the port has no img2img/edit/reference/ControlNet), so every conditioning
        // shape still defers to the Python worker — quant/LoRA being allowed does not widen this.
        let cases = [
            json!({ "mode": "edit_image", "sourceAssetId": "a" }),
            json!({ "referenceAssetId": "a" }),
            json!({ "maskAssetId": "m" }),
            json!({ "advanced": { "poses": [{ "id": "pose_1" }] } }),
        ];
        for model in ["lens", "lens_turbo"] {
            for case in &cases {
                assert!(
                    !image_request_candle_eligible(model, &object(case.clone())),
                    "{model} conditioning shape must fall back to torch: {case}"
                );
            }
        }
    }

    #[test]
    fn sdxl_advanced_shapes_fall_back_to_torch() {
        // Every conditioning shape the txt2img candle lane can't honor must be ineligible.
        let cases = [
            json!({ "mode": "edit_image", "sourceAssetId": "asset_1" }), // img2img / inpaint / outpaint
            json!({ "referenceAssetId": "asset_1" }),                    // IP-Adapter reference
            json!({ "mode": "edit_image", "sourceAssetId": "a", "maskAssetId": "m" }), // inpaint
            json!({ "loras": [{ "name": "x" }] }),                       // LoRA
            json!({ "advanced": { "poses": [{ "id": "pose_1" }] } }),    // strict-pose ControlNet
        ];
        for case in cases {
            assert!(
                !image_request_candle_eligible("sdxl", &object(case.clone())),
                "sdxl shape must fall back to torch: {case}"
            );
        }
    }

    #[test]
    fn blank_conditioning_ids_are_treated_as_absent() {
        // Whitespace/empty ids are not real conditioning → still plain txt2img → eligible.
        assert!(image_request_candle_eligible(
            "sdxl",
            &object(
                json!({ "referenceAssetId": "  ", "sourceAssetId": "", "advanced": { "poses": [] } })
            )
        ));
    }

    #[test]
    fn candle_worker_claims_txt2img_but_refuses_unsupported_shapes() {
        let candle = gpu_worker(CANDLE_CAPS);
        // Claims the lane — SDXL plus every wired candle image family, all plain txt2img.
        for model in [
            "sdxl",
            "realvisxl",
            "z_image_turbo",
            "flux_dev",
            "qwen_image",
            "chroma1_hd",
            "kolors",
            "sensenova_u1_8b",
            "sensenova_u1_8b_fast",
        ] {
            assert!(
                worker_supports_job(
                    &candle,
                    &image_generate_job(json!({ "model": model, "prompt": "a red fox" }))
                ),
                "candle worker should claim {model} plain txt2img"
            );
        }
        // Refuses a family with no candle provider, and a conditioning shape on a wired family —
        // both defer to torch.
        assert!(!worker_supports_job(
            &candle,
            &image_generate_job(json!({ "model": "bernini_image", "prompt": "p" }))
        ));
        assert!(!worker_supports_job(
            &candle,
            &image_generate_job(json!({
                "model": "kolors",
                "mode": "edit_image",
                "sourceAssetId": "asset_1"
            }))
        ));
        // sc-5489: `qwen_image` + `advanced.poses` IS now a candle lane (the bespoke strict-pose
        // ControlNet route), so the candle worker claims it (was deferred to torch before this slice).
        assert!(
            worker_supports_job(
                &candle,
                &image_generate_job(json!({
                    "model": "qwen_image",
                    "advanced": { "poses": [{ "id": "pose_1" }] }
                }))
            ),
            "candle worker should claim qwen_image strict-pose (sc-5489)"
        );
        // sc-5489: `kolors` + `advanced.poses` is also a candle lane now (the Kolors strict-pose
        // ControlNet route), so the candle worker claims it too.
        assert!(
            worker_supports_job(
                &candle,
                &image_generate_job(json!({
                    "model": "kolors",
                    "advanced": { "poses": [{ "id": "pose_1" }] }
                }))
            ),
            "candle worker should claim kolors strict-pose (sc-5489)"
        );
        // sc-5489: `z_image_turbo` + `advanced.poses` is the LAST strict-pose family wired (the VACE
        // Fun-ControlNet route) — all three (qwen / kolors / z_image) are candle lanes now.
        assert!(
            worker_supports_job(
                &candle,
                &image_generate_job(json!({
                    "model": "z_image_turbo",
                    "advanced": { "poses": [{ "id": "pose_1" }] }
                }))
            ),
            "candle worker should claim z_image_turbo strict-pose (sc-5489)"
        );
        // sc-5968: plain `sdxl` + poses has NO candle pose lane (SDXL pose ships via InstantID), and
        // the torch `sdxl` adapter has no pose path either — so the candle worker CLAIMS it (to reject
        // with a typed error in the handler) rather than declining → torch silently rendering an
        // unconditioned T2I image. `worker_supports_job` is therefore TRUE here (candle owns it to fail
        // it loudly); the handler's `candle_unsupported_pose_reject` guard does the rejecting.
        assert!(worker_supports_job(
            &candle,
            &image_generate_job(json!({
                "model": "sdxl",
                "advanced": { "poses": [{ "id": "pose_1" }] }
            }))
        ));
        // …but a plain SDXL edit (img2img) still declines on candle → torch (real edit route, sc-5487).
        assert!(!worker_supports_job(
            &candle,
            &image_generate_job(json!({
                "model": "sdxl",
                "mode": "edit_image",
                "sourceAssetId": "asset_1"
            }))
        ));
    }

    #[test]
    fn torch_worker_claims_everything_the_candle_worker_defers() {
        // The co-resident Python torch worker (no `candle` marker) is ungated here: it claims the
        // shapes the candle worker refused, so nothing is stranded — EXCEPT the unsupported-pose shapes
        // the candle worker now owns-to-reject (sc-5968, asserted at the end of this test): torch
        // declines those so it can't silently render an unconditioned T2I image.
        let torch = gpu_worker(TORCH_CAPS);
        // A family with no candle provider, and a conditioning shape on a wired family.
        assert!(worker_supports_job(
            &torch,
            &image_generate_job(json!({ "model": "bernini_image", "prompt": "p" }))
        ));
        assert!(worker_supports_job(
            &torch,
            &image_generate_job(json!({
                "model": "kolors",
                "mode": "edit_image",
                "sourceAssetId": "asset_1"
            }))
        ));
        assert!(worker_supports_job(
            &torch,
            &image_generate_job(json!({
                "model": "qwen_image",
                "advanced": { "poses": [{ "id": "pose_1" }] }
            }))
        ));
        assert!(worker_supports_job(
            &torch,
            &image_generate_job(json!({
                "model": "sdxl",
                "mode": "edit_image",
                "sourceAssetId": "asset_1"
            }))
        ));
        // sc-5968: but torch DECLINES the unsupported-pose shape the candle worker owns-to-reject
        // (sdxl + poses) — so it can't silently render an unconditioned T2I; only candle takes it (and
        // rejects). On Mac the same shape is MLX-served, so the `mlx` worker still claims it (asserted
        // in `unsupported_pose_is_owned_by_candle_declined_by_torch_served_by_mlx`).
        assert!(!worker_supports_job(
            &torch,
            &image_generate_job(json!({
                "model": "sdxl",
                "advanced": { "poses": [{ "id": "pose_1" }] }
            }))
        ));
    }

    /// sc-5968: the unsupported-pose routing across the three GPU workers — candle OWNS it (to reject),
    /// torch DECLINES it (no silent T2I), and the Mac `mlx` worker still SERVES it (no Mac regression,
    /// `sdxl_mlx_eligible` is unconditional). Plus: the wired candle pose families are unaffected, and
    /// `image_job_is_candle_eligible` still reports sdxl+poses as NOT candle-*served* (it's owned only
    /// to reject — the distinction the worker's dispatch guard keys on).
    #[test]
    fn unsupported_pose_is_owned_by_candle_declined_by_torch_served_by_mlx() {
        let candle = gpu_worker(CANDLE_CAPS);
        let torch = gpu_worker(TORCH_CAPS);
        let mlx: WorkerSnapshot = serde_json::from_value(json!({
            "id": "worker_mlx",
            "gpuId": "mlx",
            "status": "idle",
            "capabilities": ["gpu", "image_generate"],
            "loadedModels": [],
            "registeredAt": "2026-06-16T00:00:00Z",
            "lastSeenAt": "2026-06-16T00:00:00Z",
        }))
        .expect("valid WorkerSnapshot");
        let sdxl_pose = image_generate_job(
            json!({ "model": "sdxl", "advanced": { "poses": [{ "id": "p" }] } }),
        );

        assert!(image_request_candle_pose_reject(
            "sdxl",
            &object(json!({ "advanced": { "poses": [{ "id": "p" }] } }))
        ));
        assert!(worker_supports_job(&candle, &sdxl_pose), "candle owns it");
        assert!(
            !worker_supports_job(&torch, &sdxl_pose),
            "torch declines it"
        );
        assert!(worker_supports_job(&mlx, &sdxl_pose), "mlx still serves it");
        // It is NOT candle-*served* (only owned-to-reject); the worker's dispatch guard rejects it.
        assert!(!image_job_is_candle_eligible(&sdxl_pose));

        // A wired candle pose family is NOT a reject shape, and edit_image is never a reject shape.
        assert!(!image_request_candle_pose_reject(
            "qwen_image",
            &object(json!({ "advanced": { "poses": [{ "id": "p" }] } }))
        ));
        assert!(!image_request_candle_pose_reject(
            "sdxl",
            &object(json!({ "mode": "edit_image", "advanced": { "poses": [{ "id": "p" }] } }))
        ));
        // No poses → not a reject shape (plain txt2img stays candle-eligible).
        assert!(!image_request_candle_pose_reject(
            "sdxl",
            &object(json!({ "prompt": "a fox" }))
        ));
    }

    // ---- Candle video lane (sc-5097) ----

    /// A queued `video_generate` job carrying `payload`.
    fn video_generate_job(payload: Value) -> JobSnapshot {
        serde_json::from_value(json!({
            "id": "job_v",
            "type": "video_generate",
            "status": "queued",
            "payload": payload,
            "result": {},
            "requestedGpu": "auto",
            "progress": 0,
            "stage": "queued",
            "message": "",
            "attempts": 1,
            "cancelRequested": false,
            "createdAt": "2026-06-13T00:00:00Z",
            "updatedAt": "2026-06-13T00:00:00Z",
        }))
        .expect("valid JobSnapshot")
    }

    // The candle worker on the video lane advertises `video_generate` + the `candle` marker.
    const CANDLE_VIDEO_CAPS: &[&str] = &["gpu", "video_generate", "candle"];
    const TORCH_VIDEO_CAPS: &[&str] = &["gpu", "video_generate"];

    #[test]
    fn candle_routed_video_models_are_eligible_in_their_native_shape() {
        // txt2video lane: the 5B, ltx, and the 14B T2V (text-only) are eligible for text_to_video.
        for model in ["wan_2_2", "ltx_2_3", "wan_2_2_t2v_14b"] {
            assert!(
                video_request_candle_eligible(
                    model,
                    &object(json!({ "mode": "text_to_video", "prompt": "a river at dawn" }))
                ),
                "{model} text_to_video should be candle-eligible"
            );
        }
        // image→video lane: the 14B I2V + SVD are eligible only with the i2v mode + a source image
        // (sc-5175 / sc-5493).
        for model in ["wan_2_2_i2v_14b", "svd"] {
            assert!(
                video_request_candle_eligible(
                    model,
                    &object(
                        json!({ "mode": "image_to_video", "sourceAssetId": "asset_1", "prompt": "p" })
                    )
                ),
                "{model} image_to_video with a source should be candle-eligible"
            );
        }
    }

    #[test]
    fn non_candle_video_models_and_conditioned_shapes_fall_back() {
        // `ltx_2_3_eros` now routes to candle for plain text_to_video (sc-5495 — it's a full dense
        // LTX-2.3 fine-tune on the `ltx_2_3_distilled` engine), but any conditioned eros shape stays on
        // the Python torch worker (the candle LTX lane is txt2video-only).
        assert!(
            video_request_candle_eligible(
                "ltx_2_3_eros",
                &object(json!({ "mode": "text_to_video" }))
            ),
            "ltx_2_3_eros text_to_video must route to the candle lane"
        );
        assert!(
            !video_request_candle_eligible(
                "ltx_2_3_eros",
                &object(json!({ "mode": "first_last_frame" }))
            ),
            "a conditioned ltx_2_3_eros shape must fall back to the Python worker"
        );
        // A genuinely non-candle video model stays on torch.
        assert!(
            !video_request_candle_eligible(
                "some_unported_model",
                &object(json!({ "mode": "text_to_video" }))
            ),
            "an unported model must fall back to the Python worker"
        );
        // A txt2video model in any conditioned shape (default/i2v mode, a source, or a LoRA) → torch.
        let cases = [
            json!({ "prompt": "p" }), // no mode → defaults to i2v
            json!({ "mode": "image_to_video", "sourceAssetId": "a" }),
            json!({ "mode": "first_last_frame" }),
            json!({ "mode": "text_to_video", "sourceAssetId": "a" }), // txt mode but conditioned
            json!({ "mode": "text_to_video", "loras": [{ "name": "x" }] }),
        ];
        for case in cases {
            assert!(
                !video_request_candle_eligible("wan_2_2", &object(case.clone())),
                "wan_2_2 shape must fall back to torch: {case}"
            );
        }
        // The 14B T2V is text-only: any image_to_video / sourced shape falls back to torch (sc-5175).
        for case in [
            json!({ "mode": "image_to_video", "sourceAssetId": "a" }),
            json!({ "mode": "text_to_video", "sourceAssetId": "a" }),
        ] {
            assert!(
                !video_request_candle_eligible("wan_2_2_t2v_14b", &object(case.clone())),
                "wan_2_2_t2v_14b conditioned shape must fall back to torch: {case}"
            );
        }
        // The 14B I2V + SVD are image→video only: a txt2video shape, an i2v with no source, or a LoRA
        // → torch (sc-5175 / sc-5493).
        for model in ["wan_2_2_i2v_14b", "svd"] {
            for case in [
                json!({ "mode": "text_to_video", "prompt": "p" }),
                json!({ "mode": "image_to_video" }), // i2v but no source image
                json!({ "mode": "image_to_video", "sourceAssetId": "a", "loras": [{ "name": "x" }] }),
            ] {
                assert!(
                    !video_request_candle_eligible(model, &object(case.clone())),
                    "{model} non-i2v / LoRA shape must fall back to torch: {case}"
                );
            }
        }
    }

    #[test]
    fn candle_vace_modes_eligible_with_required_assets() {
        // replace_person (PersonReplace): needs the source clip + person track + character.
        assert!(video_request_candle_vace_eligible(
            "wan_2_2",
            &object(json!({
                "sourceClipAssetId": "clip_1",
                "personTrackId": "track_1",
                "characterId": "char_1"
            })),
            &JobType::PersonReplace
        ));
        // extend_clip (VideoExtend): needs a source clip.
        assert!(video_request_candle_vace_eligible(
            "wan_2_2_t2v_14b",
            &object(json!({ "sourceClipAssetId": "clip_1" })),
            &JobType::VideoExtend
        ));
        // video_bridge (VideoBridge): needs both clips.
        assert!(video_request_candle_vace_eligible(
            "wan_2_2_i2v_14b",
            &object(json!({ "sourceClipAssetId": "l", "bridgeRightClipAssetId": "r" })),
            &JobType::VideoBridge
        ));
    }

    #[test]
    fn candle_vace_modes_fall_back_without_assets_or_for_unsupported_models() {
        // Missing required assets → torch.
        assert!(!video_request_candle_vace_eligible(
            "wan_2_2",
            &object(json!({ "sourceClipAssetId": "clip_1" })), // no personTrackId / characterId
            &JobType::PersonReplace
        ));
        assert!(!video_request_candle_vace_eligible(
            "wan_2_2",
            &object(json!({ "sourceClipAssetId": "l" })), // bridge needs the right clip too
            &JobType::VideoBridge
        ));
        // SCAIL-2 (MLX-only) is not a candle VACE model → torch.
        assert!(!video_request_candle_vace_eligible(
            "scail2_14b",
            &object(json!({ "sourceClipAssetId": "c", "personTrackId": "t", "characterId": "ch" })),
            &JobType::PersonReplace
        ));
        // A LoRA shape → torch (the candle VACE provider advertises no adapters).
        assert!(!video_request_candle_vace_eligible(
            "wan_2_2",
            &object(json!({
                "sourceClipAssetId": "c",
                "personTrackId": "t",
                "characterId": "ch",
                "loras": [{ "name": "x" }]
            })),
            &JobType::PersonReplace
        ));
        // A non-VACE job type is never VACE-eligible (the base txt2video gate handles VideoGenerate).
        assert!(!video_request_candle_vace_eligible(
            "wan_2_2",
            &object(json!({ "sourceClipAssetId": "c", "personTrackId": "t", "characterId": "ch" })),
            &JobType::VideoGenerate
        ));
    }

    #[test]
    fn candle_worker_claims_txt2video_but_refuses_other_video_shapes() {
        let candle = gpu_worker(CANDLE_VIDEO_CAPS);
        // Claims wan + ltx + the 14B T2V plain txt2video.
        for model in ["wan_2_2", "ltx_2_3", "wan_2_2_t2v_14b"] {
            assert!(
                worker_supports_job(
                    &candle,
                    &video_generate_job(json!({ "model": model, "mode": "text_to_video" }))
                ),
                "candle worker should claim {model} txt2video"
            );
        }
        // Claims the 14B I2V + SVD in their image→video shape (with a source image) (sc-5175 / sc-5493).
        for model in ["wan_2_2_i2v_14b", "svd"] {
            assert!(
                worker_supports_job(
                    &candle,
                    &video_generate_job(json!({
                        "model": model,
                        "mode": "image_to_video",
                        "sourceAssetId": "a"
                    }))
                ),
                "candle worker should claim {model} image_to_video"
            );
        }
        // Claims `ltx_2_3_eros` text_to_video (sc-5495 — the candle LTX engine serves the eros fine-tune
        // too). Refuses an unported model, a conditioned (i2v) shape on a txt2video model, an image→video
        // model (svd) in a txt2video shape, and the 14B I2V in a txt2video shape (both image→video only).
        assert!(worker_supports_job(
            &candle,
            &video_generate_job(json!({ "model": "ltx_2_3_eros", "mode": "text_to_video" }))
        ));
        assert!(!worker_supports_job(
            &candle,
            &video_generate_job(json!({ "model": "some_unported_model", "mode": "text_to_video" }))
        ));
        assert!(!worker_supports_job(
            &candle,
            &video_generate_job(json!({ "model": "svd", "mode": "text_to_video" }))
        ));
        assert!(!worker_supports_job(
            &candle,
            &video_generate_job(
                json!({ "model": "wan_2_2", "mode": "image_to_video", "sourceAssetId": "a" })
            )
        ));
        assert!(!worker_supports_job(
            &candle,
            &video_generate_job(json!({ "model": "wan_2_2_i2v_14b", "mode": "text_to_video" }))
        ));
        // The co-resident torch worker claims everything the candle worker defers.
        let torch = gpu_worker(TORCH_VIDEO_CAPS);
        assert!(worker_supports_job(
            &torch,
            &video_generate_job(
                json!({ "model": "wan_2_2", "mode": "image_to_video", "sourceAssetId": "a" })
            )
        ));
    }

    // ---- SeedVR2 video upscale (epic 4811 / sc-4816) ----

    /// A queued `video_upscale` job carrying `payload`.
    fn video_upscale_job(payload: Value) -> JobSnapshot {
        serde_json::from_value(json!({
            "id": "job_vu",
            "type": "video_upscale",
            "status": "queued",
            "payload": payload,
            "result": {},
            "requestedGpu": "auto",
            "progress": 0,
            "stage": "queued",
            "message": "",
            "attempts": 1,
            "cancelRequested": false,
            "createdAt": "2026-06-13T00:00:00Z",
            "updatedAt": "2026-06-13T00:00:00Z",
        }))
        .expect("valid JobSnapshot")
    }

    /// An idle MLX (`gpu_id = "mlx"`) worker advertising `capabilities`.
    fn mlx_worker(capabilities: &[&str]) -> WorkerSnapshot {
        serde_json::from_value(json!({
            "id": "worker_mlx",
            "gpuId": "mlx",
            "status": "idle",
            "capabilities": capabilities,
            "loadedModels": [],
            "registeredAt": "2026-06-12T00:00:00Z",
            "lastSeenAt": "2026-06-12T00:00:00Z",
        }))
        .expect("valid WorkerSnapshot")
    }

    #[test]
    fn video_upscale_seedvr2_is_mlx_eligible_other_engines_are_not() {
        // seedvr2 (alias + 3b id) and the absent-engine default are eligible.
        for engine in [json!("seedvr2"), json!("seedvr2_3b"), Value::Null] {
            let payload = if engine.is_null() {
                json!({ "sourceAssetId": "a" })
            } else {
                json!({ "sourceAssetId": "a", "engine": engine })
            };
            assert!(
                video_upscale_job_is_mlx_eligible(&video_upscale_job(payload.clone())),
                "video_upscale should be MLX-eligible for {payload}"
            );
        }
        // An unknown engine is not eligible (no torch video upscaler exists).
        assert!(!video_upscale_job_is_mlx_eligible(&video_upscale_job(
            json!({ "sourceAssetId": "a", "engine": "aura-sr" })
        )));
        // The predicate is gated to the job type.
        assert!(!video_upscale_job_is_mlx_eligible(&video_generate_job(
            json!({ "model": "wan_2_2" })
        )));
    }

    #[test]
    fn mlx_worker_claims_seedvr2_video_upscale_and_refuses_other_engines() {
        let mlx = mlx_worker(&["gpu", "video_upscale"]);
        assert!(worker_supports_job(
            &mlx,
            &video_upscale_job(json!({ "sourceAssetId": "a", "engine": "seedvr2" }))
        ));
        // A non-SeedVR2 engine is refused by the mlx worker (mac-only; nowhere else to run).
        assert!(!worker_supports_job(
            &mlx,
            &video_upscale_job(json!({ "sourceAssetId": "a", "engine": "aura-sr" }))
        ));
    }

    #[test]
    fn video_upscale_requires_gpu() {
        assert!(job_requires_gpu(&JobType::VideoUpscale));
    }

    #[test]
    fn mac_capabilities_advertises_video_upscale() {
        let caps = mac_capabilities("darwin", true);
        let feature = caps
            .features
            .get("videoUpscale")
            .expect("videoUpscale feature present");
        assert!(feature.supported);
        assert!(feature.reason.is_none());
    }

    #[test]
    fn mac_rust_supports_seedvr2_video_upscale_only() {
        assert!(mac_rust_supported(&video_upscale_job(
            json!({ "sourceAssetId": "a", "engine": "seedvr2" })
        ))
        .is_ok());
        assert!(mac_rust_supported(&video_upscale_job(
            json!({ "sourceAssetId": "a", "engine": "aura-sr" })
        ))
        .is_err());
    }

    // ---- Candle caption lane (sc-5098) ----

    /// A queued `training_caption` job carrying `payload`.
    fn caption_job(payload: Value) -> JobSnapshot {
        serde_json::from_value(json!({
            "id": "job_c",
            "type": "training_caption",
            "status": "queued",
            "payload": payload,
            "result": {},
            "requestedGpu": "auto",
            "progress": 0,
            "stage": "queued",
            "message": "",
            "attempts": 1,
            "cancelRequested": false,
            "createdAt": "2026-06-13T00:00:00Z",
            "updatedAt": "2026-06-13T00:00:00Z",
        }))
        .expect("valid JobSnapshot")
    }

    #[test]
    fn candle_worker_claims_joycaption_but_refuses_other_captioners() {
        let candle = gpu_worker(&["gpu", "training_caption", "candle"]);
        // Claims a JoyCaption job.
        assert!(worker_supports_job(
            &candle,
            &caption_job(json!({ "captioner": "joy_caption", "datasetId": "ds_1" }))
        ));
        // Refuses a non-JoyCaption captioner → falls back to the Python torch worker.
        assert!(!worker_supports_job(
            &candle,
            &caption_job(json!({ "captioner": "blip2", "datasetId": "ds_1" }))
        ));
        let torch = gpu_worker(&["gpu", "training_caption"]);
        assert!(worker_supports_job(
            &torch,
            &caption_job(json!({ "captioner": "blip2", "datasetId": "ds_1" }))
        ));
    }

    /// sc-5501: the candle worker claims SenseNova-U1 `image_vqa` / `image_interleave` (served off-Mac
    /// by the concrete candle `T2iModel::{vqa, interleave_gen}`) but refuses other models, which stay
    /// on the Python torch worker.
    #[test]
    fn candle_worker_claims_sensenova_understanding_but_refuses_other_models() {
        let candle = gpu_worker(&["gpu", "image_vqa", "image_interleave", "candle"]);
        let understanding_job = |job_type: &str, payload: Value| -> JobSnapshot {
            serde_json::from_value(json!({
                "id": "job_u",
                "type": job_type,
                "status": "queued",
                "payload": payload,
                "result": {},
                "requestedGpu": "auto",
                "progress": 0,
                "stage": "queued",
                "message": "",
                "attempts": 1,
                "cancelRequested": false,
                "createdAt": "2026-06-14T00:00:00Z",
                "updatedAt": "2026-06-14T00:00:00Z",
            }))
            .expect("valid JobSnapshot")
        };
        // Claims SenseNova-U1 VQA + interleave (base + `_fast` ids).
        assert!(worker_supports_job(
            &candle,
            &understanding_job(
                "image_vqa",
                json!({ "model": "sensenova_u1_8b", "question": "what is this?", "sourceAssetId": "a1" })
            )
        ));
        assert!(worker_supports_job(
            &candle,
            &understanding_job(
                "image_interleave",
                json!({ "model": "sensenova_u1_8b_fast", "prompt": "a short illustrated story" })
            )
        ));
        // Refuses a non-SenseNova understanding job → falls back to the Python torch worker.
        assert!(!worker_supports_job(
            &candle,
            &understanding_job(
                "image_vqa",
                json!({ "model": "some_other_vlm", "question": "?", "sourceAssetId": "a1" })
            )
        ));
    }

    #[test]
    fn instantid_character_jobs_route_to_candle_off_mac() {
        // The candle InstantID provider (sc-5491) serves the SAME surface as the MLX path off-Mac, so
        // every character_image + referenceAssetId shape is candle-eligible — via the bespoke
        // `image_job_is_candle_eligible` branch, NOT the txt2img-only `image_request_candle_eligible`
        // gate (which rejects `referenceAssetId`, which InstantID requires).
        for advanced in [
            json!({}),
            json!({ "angleSet": true }),
            json!({ "poses": [{ "id": "a" }] }),
            json!({ "faceRestore": true }),
            json!({ "poses": [{ "id": "a" }], "faceRestore": true }),
        ] {
            let payload = json!({
                "model": "instantid_realvisxl",
                "mode": "character_image",
                "referenceAssetId": "asset_1",
                "advanced": advanced,
            });
            assert!(instantid_candle_eligible(&object(payload.clone())));
            assert!(image_job_is_candle_eligible(&image_generate_job(payload)));
        }

        // No reference face → not candle-eligible (mirrors the MLX gate).
        assert!(!image_job_is_candle_eligible(&image_generate_job(json!({
            "model": "instantid_realvisxl",
            "mode": "character_image"
        }))));
        // Non-character mode → not candle-eligible (InstantID is a character flow).
        assert!(!image_job_is_candle_eligible(&image_generate_job(json!({
            "model": "instantid_realvisxl",
            "mode": "text_to_image",
            "referenceAssetId": "asset_1"
        }))));
    }

    #[test]
    fn sdxl_ipadapter_reference_jobs_route_to_candle() {
        // A pure SDXL/RealVisXL reference (IP-Adapter) job routes to the candle lane (sc-5488) via the
        // bespoke branch, NOT the txt2img `image_request_candle_eligible` gate (which rejects
        // `referenceAssetId`).
        for model in ["sdxl", "realvisxl"] {
            let payload = json!({ "model": model, "referenceAssetId": "asset_1" });
            assert!(sdxl_ipadapter_candle_eligible(&object(payload.clone())));
            assert!(image_job_is_candle_eligible(&image_generate_job(payload)));
        }
        // No reference → not an IP-Adapter job (plain txt2img routes via the txt2img gate instead).
        assert!(!sdxl_ipadapter_candle_eligible(&object(
            json!({ "model": "sdxl" })
        )));
        // img2img / inpaint / edit shapes are NOT this lane (those are sc-5487, still torch).
        assert!(!sdxl_ipadapter_candle_eligible(&object(json!({
            "model": "sdxl", "mode": "edit_image", "referenceAssetId": "a", "sourceAssetId": "s"
        }))));
        assert!(!sdxl_ipadapter_candle_eligible(&object(json!({
            "model": "sdxl", "referenceAssetId": "a", "sourceAssetId": "s"
        }))));
        assert!(!sdxl_ipadapter_candle_eligible(&object(json!({
            "model": "sdxl", "referenceAssetId": "a", "maskAssetId": "m"
        }))));
    }

    #[test]
    fn kolors_ipadapter_reference_jobs_route_to_candle() {
        // A pure Kolors reference (IP-Adapter) job routes to the candle lane (sc-5488) via the bespoke
        // branch, NOT the txt2img `image_request_candle_eligible` gate (which rejects `referenceAssetId`).
        let payload = json!({ "model": "kolors", "referenceAssetId": "asset_1" });
        assert!(kolors_ipadapter_candle_eligible(&object(payload.clone())));
        assert!(image_job_is_candle_eligible(&image_generate_job(payload)));
        // No reference → plain txt2img routes via the txt2img gate instead.
        assert!(!kolors_ipadapter_candle_eligible(&object(
            json!({ "model": "kolors" })
        )));
        // img2img / inpaint / edit shapes are NOT this lane (those are sc-5487, still torch).
        assert!(!kolors_ipadapter_candle_eligible(&object(json!({
            "model": "kolors", "mode": "edit_image", "referenceAssetId": "a", "sourceAssetId": "s"
        }))));
        assert!(!kolors_ipadapter_candle_eligible(&object(json!({
            "model": "kolors", "referenceAssetId": "a", "sourceAssetId": "s"
        }))));
        assert!(!kolors_ipadapter_candle_eligible(&object(json!({
            "model": "kolors", "referenceAssetId": "a", "maskAssetId": "m"
        }))));
    }

    #[test]
    fn flux_ipadapter_reference_jobs_route_to_candle() {
        // A pure FLUX reference (XLabs IP-Adapter) job routes to the candle lane (sc-5872) via the
        // bespoke branch, NOT the txt2img `image_request_candle_eligible` gate (which rejects
        // `referenceAssetId`). Both variants.
        for model in ["flux_dev", "flux_schnell"] {
            let payload = json!({ "model": model, "referenceAssetId": "asset_1" });
            assert!(flux_ipadapter_candle_eligible(&object(payload.clone())));
            assert!(image_job_is_candle_eligible(&image_generate_job(payload)));
        }
        // No reference → plain txt2img routes via the txt2img gate instead.
        assert!(!flux_ipadapter_candle_eligible(&object(
            json!({ "model": "flux_dev" })
        )));
        // img2img / inpaint / edit shapes are NOT this lane (those are sc-5487, still torch).
        assert!(!flux_ipadapter_candle_eligible(&object(json!({
            "model": "flux_dev", "mode": "edit_image", "referenceAssetId": "a", "sourceAssetId": "s"
        }))));
        assert!(!flux_ipadapter_candle_eligible(&object(json!({
            "model": "flux_dev", "referenceAssetId": "a", "sourceAssetId": "s"
        }))));
        assert!(!flux_ipadapter_candle_eligible(&object(json!({
            "model": "flux_schnell", "referenceAssetId": "a", "maskAssetId": "m"
        }))));
    }

    #[test]
    fn qwen_control_pose_jobs_route_to_candle() {
        // qwen_image + advanced.poses routes to the candle strict-pose lane (sc-5489) via the bespoke
        // branch, NOT the txt2img gate (which DEFERS any advanced.poses job to torch).
        let payload =
            json!({ "model": "qwen_image", "advanced": { "poses": [{ "keypoints": [] }] } });
        assert!(qwen_control_candle_eligible(&object(payload.clone())));
        assert!(image_job_is_candle_eligible(&image_generate_job(payload)));
        // No poses (or empty) → plain txt2img routes via the txt2img gate instead.
        assert!(!qwen_control_candle_eligible(&object(
            json!({ "model": "qwen_image" })
        )));
        assert!(!qwen_control_candle_eligible(&object(json!({
            "model": "qwen_image", "advanced": { "poses": [] }
        }))));
        // edit_image with poses is NOT this lane.
        assert!(!qwen_control_candle_eligible(&object(json!({
            "model": "qwen_image", "mode": "edit_image", "advanced": { "poses": [{}] }
        }))));
        // Plain `sdxl` + poses is NOT candle-*served* (no plain-SDXL pose lane — SDXL pose ships via
        // InstantID): the qwen branch is specific and the txt2img gate's has_poses check rejects it, so
        // `image_job_is_candle_eligible` is false. (It is, however, candle-*owned-to-reject* at the
        // worker layer per sc-5968 — see `unsupported_pose_is_owned_by_candle_*`; that claim lives in
        // `worker_supports_job`, not here. z_image_turbo + poses IS a candle lane — `zimage_control_*`.)
        assert!(!image_job_is_candle_eligible(&image_generate_job(json!({
            "model": "sdxl", "advanced": { "poses": [{}] }
        }))));
    }

    #[test]
    fn kolors_control_pose_jobs_route_to_candle() {
        // kolors + advanced.poses routes to the candle strict-pose lane (sc-5489) via the bespoke
        // branch, NOT the txt2img gate (which DEFERS any advanced.poses job to torch).
        let payload = json!({ "model": "kolors", "advanced": { "poses": [{ "keypoints": [] }] } });
        assert!(kolors_control_candle_eligible(&object(payload.clone())));
        assert!(image_job_is_candle_eligible(&image_generate_job(payload)));
        // No poses (or empty) → plain txt2img routes via the txt2img gate instead.
        assert!(!kolors_control_candle_eligible(&object(
            json!({ "model": "kolors" })
        )));
        assert!(!kolors_control_candle_eligible(&object(json!({
            "model": "kolors", "advanced": { "poses": [] }
        }))));
        // edit_image with poses is NOT this lane.
        assert!(!kolors_control_candle_eligible(&object(json!({
            "model": "kolors", "mode": "edit_image", "advanced": { "poses": [{}] }
        }))));
        // A kolors reference job (no poses) still routes via the IP-Adapter branch, not this one.
        assert!(!kolors_control_candle_eligible(&object(json!({
            "model": "kolors", "referenceAssetId": "asset_1"
        }))));
    }

    #[test]
    fn zimage_control_pose_jobs_route_to_candle() {
        // z_image_turbo + advanced.poses routes to the candle VACE strict-pose lane (sc-5489, the last
        // family) via the bespoke branch, NOT the txt2img gate (which DEFERS any advanced.poses to torch).
        let payload =
            json!({ "model": "z_image_turbo", "advanced": { "poses": [{ "keypoints": [] }] } });
        assert!(zimage_control_candle_eligible(&object(payload.clone())));
        assert!(image_job_is_candle_eligible(&image_generate_job(payload)));
        // No poses (or empty) → plain txt2img routes via the txt2img gate instead.
        assert!(!zimage_control_candle_eligible(&object(
            json!({ "model": "z_image_turbo" })
        )));
        assert!(!zimage_control_candle_eligible(&object(json!({
            "model": "z_image_turbo", "advanced": { "poses": [] }
        }))));
        // edit_image with poses is NOT this lane.
        assert!(!zimage_control_candle_eligible(&object(json!({
            "model": "z_image_turbo", "mode": "edit_image", "advanced": { "poses": [{}] }
        }))));
    }
}

#[cfg(test)]
mod mlx_routing_tests {
    use super::{
        flux2_mlx_eligible, flux_mlx_eligible, image_request_mlx_eligible, instantid_mlx_eligible,
        qwen_edit_mlx_eligible, qwen_mlx_eligible, sdxl_mlx_eligible, video_mode_is_mlx_eligible,
        z_image_mlx_eligible, VIDEO_MLX_ROUTED_MODELS,
    };
    use serde_json::{json, Map, Value};

    fn object(value: Value) -> Map<String, Value> {
        value.as_object().expect("test value is an object").clone()
    }

    #[test]
    fn z_image_plain_txt2img_is_eligible() {
        assert!(z_image_mlx_eligible(&object(
            json!({ "prompt": "a misty fjord" })
        )));
        assert!(z_image_mlx_eligible(&Map::new()));
    }

    #[test]
    fn z_image_edit_mode_with_source_is_eligible() {
        // epic 3529: img2img-edit (sourceAssetId) now routes to MLX via the engine's
        // `Conditioning::Reference` img2img path.
        assert!(z_image_mlx_eligible(&object(json!({
            "mode": "edit_image",
            "sourceAssetId": "asset_1"
        }))));
    }

    #[test]
    fn z_image_edit_mode_without_source_is_not_eligible() {
        // An edit with nothing to edit (no/blank sourceAssetId) stays off MLX.
        assert!(!z_image_mlx_eligible(&object(
            json!({ "mode": "edit_image" })
        )));
        assert!(!z_image_mlx_eligible(&object(json!({
            "mode": "edit_image",
            "sourceAssetId": "   "
        }))));
    }

    #[test]
    fn z_image_reference_without_poses_is_eligible() {
        // sc-3619: reference-identity img2img-init (no pose) now routes to MLX — the
        // base engine already supports the plain img2img path, and torch dropped the
        // reference entirely (it was a no-op on the fallback).
        assert!(z_image_mlx_eligible(&object(
            json!({ "referenceAssetId": "asset_1" })
        )));
        // Empty/whitespace reference id is treated as absent → plain txt2img, eligible.
        assert!(z_image_mlx_eligible(&object(
            json!({ "referenceAssetId": "   " })
        )));
        // A reference with empty poses is still reference-only → eligible (not the
        // pose tier, which needs a non-empty pose set).
        assert!(z_image_mlx_eligible(&object(json!({
            "referenceAssetId": "asset_1",
            "advanced": { "poses": [] }
        }))));
    }

    #[test]
    fn z_image_reference_with_poses_stays_on_mlx() {
        // The strict pose ControlNet tier lives only on MLX, so a reference+pose
        // job must route to the mlx worker, not torch (which would drop the poses).
        assert!(z_image_mlx_eligible(&object(json!({
            "referenceAssetId": "asset_1",
            "advanced": { "poses": [{ "id": "pose_1" }] }
        }))));
    }

    #[test]
    fn z_image_peft_lokr_and_thirdparty_lycoris_both_route_mlx() {
        // SceneWorks peft LoKr applies natively on the MLX Z-Image path → eligible.
        assert!(z_image_mlx_eligible(&object(json!({
            "loras": [{ "path": "a.safetensors", "networkType": "lokr" }]
        }))));
        // Third-party LyCORIS now applies via the core MLX loader (epic 3641) → MLX too.
        assert!(z_image_mlx_eligible(&object(json!({
            "loras": [{ "path": "b.safetensors", "networkType": "lycoris" }]
        }))));
    }

    #[test]
    fn flux_plain_txt2img_is_eligible() {
        assert!(flux_mlx_eligible(&object(json!({ "prompt": "a red fox" }))));
        assert!(flux_mlx_eligible(&Map::new()));
        // A LoRA is fine on the MLX flux path (engine applies LoRA + peft LoKr).
        assert!(flux_mlx_eligible(&object(json!({
            "loras": [{ "path": "a.safetensors", "networkType": "lora" }]
        }))));
    }

    #[test]
    fn flux_reference_is_eligible() {
        // Reference (XLabs IP-Adapter, epic 3621) now routes to MLX on both variants —
        // the Rust engine has no diffusers schnell limitation.
        assert!(flux_mlx_eligible(&object(
            json!({ "referenceAssetId": "asset_1" })
        )));
        // A reference + LoRA is still fine.
        assert!(flux_mlx_eligible(&object(json!({
            "referenceAssetId": "asset_1",
            "loras": [{ "networkType": "lora" }]
        }))));
    }

    #[test]
    fn flux_only_edit_falls_back_lycoris_routes_mlx() {
        // edit_image (no FLUX.1 edit on any platform — future Kontext) is the only fall-back.
        assert!(!flux_mlx_eligible(&object(json!({ "mode": "edit_image" }))));
        // Third-party LyCORIS now applies via the core MLX loader (epic 3641) → MLX.
        assert!(flux_mlx_eligible(&object(json!({
            "loras": [{ "networkType": "lycoris" }]
        }))));
        // Reference + a LyCORIS LoRA also routes MLX now.
        assert!(flux_mlx_eligible(&object(json!({
            "referenceAssetId": "asset_1",
            "loras": [{ "networkType": "lycoris" }]
        }))));
    }

    #[test]
    fn qwen_plain_txt2img_is_eligible() {
        assert!(qwen_mlx_eligible(&object(json!({ "prompt": "a red fox" }))));
        // A negative prompt + LoRA are fine on the MLX qwen path (true CFG + LoRA wired).
        assert!(qwen_mlx_eligible(&object(json!({
            "negativePrompt": "blurry",
            "loras": [{ "networkType": "lokr" }]
        }))));
    }

    #[test]
    fn qwen_edit_reference_falls_back_but_pose_and_lycoris_route_mlx() {
        assert!(!qwen_mlx_eligible(&object(json!({ "mode": "edit_image" }))));
        assert!(!qwen_mlx_eligible(&object(
            json!({ "referenceAssetId": "asset_1" })
        )));
        // Strict pose ControlNet (sc-2291 / sc-3575) routes to MLX, even if a reference is
        // present; the strict-pose tier is pose-from-prompt and ignores the reference.
        assert!(qwen_mlx_eligible(&object(json!({
            "advanced": { "poses": [{ "id": "p1" }] }
        }))));
        assert!(qwen_mlx_eligible(&object(json!({
            "referenceAssetId": "asset_1",
            "advanced": { "poses": [{ "id": "p1" }] }
        }))));
        // Third-party LyCORIS on a plain txt2img qwen job now routes MLX (epic 3641).
        assert!(qwen_mlx_eligible(&object(json!({
            "loras": [{ "networkType": "lycoris" }]
        }))));
    }

    #[test]
    fn qwen_edit_routes_edit_and_reference_flows_to_mlx() {
        // sc-3397: the qwen_image_edit ids run the engine's `qwen_image_edit` model.
        // edit_image with a source → eligible.
        assert!(qwen_edit_mlx_eligible(&object(json!({
            "mode": "edit_image", "sourceAssetId": "src_1"
        }))));
        // character_image with a reference (subject variation) → eligible.
        assert!(qwen_edit_mlx_eligible(&object(json!({
            "mode": "character_image", "referenceAssetId": "ref_1"
        }))));
        // character_image + reference + best-effort poses → still eligible. Unlike the base
        // Qwen strict-pose ControlNet (torch until epic 3401), the edit best-effort pose tier
        // is native multi-image ([reference, skeleton]) → MLX.
        assert!(qwen_edit_mlx_eligible(&object(json!({
            "mode": "character_image", "referenceAssetId": "ref_1",
            "advanced": { "poses": [{ "id": "p1" }] }
        }))));
        // character_image + reference + angle set → eligible.
        assert!(qwen_edit_mlx_eligible(&object(json!({
            "mode": "character_image", "referenceAssetId": "ref_1",
            "advanced": { "angleSet": true }
        }))));
        // A peft LoKr is fine on the MLX edit path.
        assert!(qwen_edit_mlx_eligible(&object(json!({
            "mode": "edit_image", "sourceAssetId": "src_1",
            "loras": [{ "networkType": "lokr" }]
        }))));
    }

    #[test]
    fn qwen_edit_without_reference_falls_back_to_torch() {
        // edit_image with nothing to edit (no source, no reference) → torch.
        assert!(!qwen_edit_mlx_eligible(&object(
            json!({ "mode": "edit_image" })
        )));
        // character_image without a reference → torch (the edit model needs a reference).
        assert!(!qwen_edit_mlx_eligible(&object(
            json!({ "mode": "character_image" })
        )));
        // A plain txt2img mode is not an edit job (that's the base qwen_image MLX path).
        assert!(!qwen_edit_mlx_eligible(&object(json!({
            "mode": "text_to_image", "sourceAssetId": "src_1"
        }))));
        // Whitespace-only ids are treated as absent.
        assert!(!qwen_edit_mlx_eligible(&object(json!({
            "mode": "edit_image", "sourceAssetId": "   "
        }))));
        // A third-party LyCORIS LoRA on an otherwise-eligible edit job now routes MLX (epic 3641).
        assert!(qwen_edit_mlx_eligible(&object(json!({
            "mode": "edit_image", "sourceAssetId": "src_1",
            "loras": [{ "networkType": "lycoris" }]
        }))));
    }

    #[test]
    fn flux2_txt2img_edit_and_lycoris_all_route_mlx() {
        // FLUX.2 is MLX-only: txt2img (sc-3025), edit/reference (sc-3029), and — since epic 3641 —
        // third-party LyCORIS all route MLX.
        assert!(flux2_mlx_eligible(&object(
            json!({ "prompt": "a red fox" })
        )));
        assert!(flux2_mlx_eligible(&object(json!({ "mode": "edit_image" }))));
        assert!(flux2_mlx_eligible(&object(
            json!({ "referenceAssetId": "asset_1" })
        )));
        assert!(flux2_mlx_eligible(&object(json!({
            "loras": [{ "networkType": "lycoris" }]
        }))));
    }

    #[test]
    fn sdxl_eligible_for_txt2img_edit_reference_lokr_and_lycoris() {
        assert!(sdxl_mlx_eligible(&object(json!({ "prompt": "a red fox" }))));
        // peft LoKr stays on MLX (the Rust SDXL path supports LoKr, unlike the old vendored path).
        assert!(sdxl_mlx_eligible(&object(json!({
            "loras": [{ "networkType": "lokr" }]
        }))));
        // sc-3060: the Rust engine now handles the advanced shapes, so edit_image
        // (img2img / inpaint / outpaint) and reference/IP-Adapter route to MLX too.
        assert!(sdxl_mlx_eligible(&object(json!({ "mode": "edit_image" }))));
        assert!(sdxl_mlx_eligible(&object(
            json!({ "referenceAssetId": "asset_1" })
        )));
        assert!(sdxl_mlx_eligible(&object(json!({
            "mode": "edit_image",
            "maskAssetId": "mask_1"
        }))));
        // Third-party LyCORIS now applies on the SDXL merge path (epic 3641, sc-3671) → MLX,
        // including on an edit job.
        assert!(sdxl_mlx_eligible(&object(json!({
            "loras": [{ "networkType": "lycoris" }]
        }))));
        assert!(sdxl_mlx_eligible(&object(json!({
            "mode": "edit_image",
            "loras": [{ "networkType": "lycoris" }]
        }))));
    }

    #[test]
    fn instantid_routes_all_character_modes_to_mlx() {
        // The full InstantID surface is native (sc-3345 identity + angle; sc-3381 pose + restore):
        // every character_image + referenceAssetId shape routes to MLX.
        for advanced in [
            json!({}),
            json!({ "angleSet": true }),
            json!({ "poses": [{ "id": "a" }] }),
            json!({ "faceRestore": true }),
            json!({ "poses": [{ "id": "a" }], "faceRestore": true }),
        ] {
            let payload = object(json!({
                "model": "instantid_realvisxl",
                "mode": "character_image",
                "referenceAssetId": "asset_1",
                "advanced": advanced,
            }));
            assert!(instantid_mlx_eligible(&payload));
            assert!(image_request_mlx_eligible("instantid_realvisxl", &payload));
        }

        // No reference face → not eligible.
        assert!(!instantid_mlx_eligible(&object(json!({
            "model": "instantid_realvisxl",
            "mode": "character_image"
        }))));

        // Non-character mode → not eligible (InstantID is a character flow).
        assert!(!instantid_mlx_eligible(&object(json!({
            "model": "instantid_realvisxl",
            "mode": "text_to_image"
        }))));
    }

    #[test]
    fn video_mode_eligibility_admits_flf_only_on_flf_capable_engines() {
        // image_to_video is MLX on every routed model EXCEPT Bernini (text_to_video only — its
        // renderer is Wan2.2-T2V, no still-image-to-video) and SCAIL-2 (animate_character only);
        // text_to_video on every routed model EXCEPT SVD (image-conditioned only, sc-3523) and
        // SCAIL-2 (animate_character only — sc-5448).
        for model in VIDEO_MLX_ROUTED_MODELS {
            assert_eq!(
                video_mode_is_mlx_eligible(model, "image_to_video"),
                *model != "bernini" && *model != "scail2_14b",
                "image_to_video eligibility for {model}"
            );
            assert_eq!(
                video_mode_is_mlx_eligible(model, "text_to_video"),
                *model != "svd" && *model != "scail2_14b",
                "text_to_video eligibility for {model}"
            );
        }
        // SVD serves image_to_video ONLY — no text_to_video, FLF, or anything else.
        assert!(video_mode_is_mlx_eligible("svd", "image_to_video"));
        for mode in [
            "text_to_video",
            "first_last_frame",
            "replace_person",
            "nonsense",
        ] {
            assert!(!video_mode_is_mlx_eligible("svd", mode));
        }
        // Bernini serves text_to_video + the planner editing/reference video modes (sc-4703:
        // video_to_video / reference_to_video / reference_video_to_video) + the multi-source
        // modes (sc-5425: multi_video_to_video / ads2v). It has no classic still-image-to-video
        // / FLF / replace_person (its renderer is Wan2.2-T2V).
        for mode in [
            "text_to_video",
            "video_to_video",
            "reference_to_video",
            "reference_video_to_video",
            "multi_video_to_video",
            "ads2v",
        ] {
            assert!(
                video_mode_is_mlx_eligible("bernini", mode),
                "bernini should serve {mode}"
            );
        }
        for mode in [
            "image_to_video",
            "first_last_frame",
            "extend_clip",
            "video_bridge",
            "replace_person",
            "nonsense",
        ] {
            assert!(
                !video_mode_is_mlx_eligible("bernini", mode),
                "bernini should not serve {mode}"
            );
        }
        // The editing/reference + multi-source modes are Bernini-only — every other routed
        // model rejects them.
        for model in VIDEO_MLX_ROUTED_MODELS {
            if *model == "bernini" {
                continue;
            }
            for mode in [
                "video_to_video",
                "reference_to_video",
                "reference_video_to_video",
                "multi_video_to_video",
                "ads2v",
            ] {
                assert!(
                    !video_mode_is_mlx_eligible(model, mode),
                    "{mode} should be Bernini-only, not eligible on {model}"
                );
            }
        }
        // SCAIL-2 serves the standalone character-animation mode (sc-5448, the worker paints its
        // masks from native SAM3) AND cross-identity replace_person (sc-5452, the integrated backend
        // behind the person-track pipeline). No text/image-to-video.
        for mode in ["animate_character", "replace_person"] {
            assert!(
                video_mode_is_mlx_eligible("scail2_14b", mode),
                "scail2 should serve {mode}"
            );
        }
        for mode in [
            "text_to_video",
            "image_to_video",
            "first_last_frame",
            "extend_clip",
            "video_bridge",
            "video_to_video",
            "nonsense",
        ] {
            assert!(
                !video_mode_is_mlx_eligible("scail2_14b", mode),
                "scail2 should not serve {mode}"
            );
        }
        // animate_character is SCAIL-2-only — every other routed model rejects it.
        for model in VIDEO_MLX_ROUTED_MODELS {
            if *model == "scail2_14b" {
                continue;
            }
            assert!(
                !video_mode_is_mlx_eligible(model, "animate_character"),
                "animate_character should be SCAIL-2-only, not eligible on {model}"
            );
        }
        // first_last_frame: MLX on LTX (base + eros) + Wan TI2V-5B (sc-3055 cutover).
        assert!(video_mode_is_mlx_eligible("ltx_2_3", "first_last_frame"));
        assert!(video_mode_is_mlx_eligible(
            "ltx_2_3_eros",
            "first_last_frame"
        ));
        assert!(video_mode_is_mlx_eligible("wan_2_2", "first_last_frame"));
        // FLF stays torch on the 14B Wan MoE engines (no engine Keyframe path).
        assert!(!video_mode_is_mlx_eligible(
            "wan_2_2_t2v_14b",
            "first_last_frame"
        ));
        assert!(!video_mode_is_mlx_eligible(
            "wan_2_2_i2v_14b",
            "first_last_frame"
        ));
        // extend_clip / video_bridge: MLX on the LTX IC-LoRA path (sc-3522) and Wan TI2V-5B
        // (`wan_2_2`, single-frame boundary keyframe conditioning — sc-3357).
        for mode in ["extend_clip", "video_bridge"] {
            assert!(video_mode_is_mlx_eligible("ltx_2_3", mode));
            assert!(video_mode_is_mlx_eligible("ltx_2_3_eros", mode));
            assert!(video_mode_is_mlx_eligible("wan_2_2", mode));
            // The 14B Wan MoE engines have no `Keyframe` path → torch.
            assert!(!video_mode_is_mlx_eligible("wan_2_2_t2v_14b", mode));
            assert!(!video_mode_is_mlx_eligible("wan_2_2_i2v_14b", mode));
        }
        // replace_person → native Wan-VACE is MLX on the replace-capable models (sc-3521).
        assert!(video_mode_is_mlx_eligible("ltx_2_3", "replace_person"));
        assert!(video_mode_is_mlx_eligible("ltx_2_3_eros", "replace_person"));
        assert!(video_mode_is_mlx_eligible("wan_2_2", "replace_person"));
        // Unknown modes are never eligible.
        assert!(!video_mode_is_mlx_eligible("ltx_2_3", "nonsense"));
    }
}
