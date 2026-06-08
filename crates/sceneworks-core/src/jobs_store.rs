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
use crate::store_util::parse_string_enum;
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
    RetryLimit { max_attempts: u32 },
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
        let now = utc_now();
        transaction.execute(
            "
            update jobs
               set status = 'interrupted',
                   stage = 'interrupted',
                   message = 'Job was interrupted by a backend restart.',
                   error = 'The backend restarted before this job finished.',
                   completed_at = ?1,
                   updated_at = ?1,
                   worker_id = null
             where status in ('preparing', 'downloading', 'loading_model', 'running', 'saving')
            ",
            params![now],
        )?;
        transaction.execute(
            "update workers set status = 'offline', current_job_id = null where status != 'offline'",
            [],
        )?;
        transaction.commit()?;
        Ok(interrupted)
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
                   and status in ('preparing', 'downloading', 'loading_model', 'running', 'saving')
                "
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
        let has_active_gpu_job = transaction
            .query_row(
                &format!(
                    "
                select id from jobs
                 where assigned_gpu = ?1
                   and status in ('preparing', 'downloading', 'loading_model', 'running', 'saving')
                   and type not in ({})
                 limit 1
                ",
                    non_gpu_job_types_sql()
                ),
                params![worker.gpu_id],
                |_row| Ok(()),
            )
            .optional()?
            .is_some();

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
            merge_training_sample_history(&transaction, job_id, result)?;
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
        let jobs = self.list_jobs(None, None, 500)?;
        let workers = self.list_workers()?;
        let mut counts = JOB_STATUSES
            .iter()
            .map(|status| (parse_string_enum::<JobStatus>(status), 0))
            .collect::<std::collections::BTreeMap<_, _>>();
        for job in &jobs {
            *counts.entry(job.status.clone()).or_insert(0) += 1;
        }
        Ok(QueueSummary {
            counts,
            active_jobs: jobs
                .into_iter()
                .filter(|job| !is_terminal_status(job.status.as_str()))
                .collect(),
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
            Err(_) => {
                remove_sqlite_sidecars(&self.db_path);
                connection.pragma_update(None, "journal_mode", "delete")?;
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
               and status in ('preparing', 'downloading', 'loading_model', 'running', 'saving')
            "
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

fn merge_training_sample_history(
    connection: &Connection,
    job_id: &str,
    incoming: &mut Map<String, Value>,
) -> JobsStoreResult<()> {
    let has_training_samples = incoming
        .get("trainingSamples")
        .and_then(Value::as_array)
        .is_some();
    let has_latest_training_samples = incoming
        .get("latestTrainingSamples")
        .and_then(Value::as_array)
        .is_some();
    if !has_training_samples && !has_latest_training_samples {
        return Ok(());
    }

    let existing_json: Option<String> = connection
        .query_row(
            "select result_json from jobs where id = ?1",
            params![job_id],
            |row| row.get(0),
        )
        .optional()?;
    let existing = loads_object(existing_json.as_deref());
    let mut samples = Vec::new();
    let mut seen = std::collections::HashSet::new();
    append_training_samples(&mut samples, &mut seen, existing.get("trainingSamples"));
    append_training_samples(&mut samples, &mut seen, incoming.get("trainingSamples"));
    append_training_samples(
        &mut samples,
        &mut seen,
        incoming.get("latestTrainingSamples"),
    );

    if !samples.is_empty() {
        incoming.insert("trainingSamples".to_owned(), Value::Array(samples));
    }
    Ok(())
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

fn remove_sqlite_sidecars(db_path: &Path) {
    for suffix in ["-wal", "-shm"] {
        let sidecar = db_path.with_file_name(format!(
            "{}{suffix}",
            db_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
        ));
        let _ = fs::remove_file(sidecar);
    }
}

fn ensure_column(
    connection: &Connection,
    table: &str,
    column: &str,
    definition: &str,
) -> JobsStoreResult<()> {
    let mut statement = connection.prepare(&format!("pragma table_info({table})"))?;
    let columns = statement
        .query_map([], |row| row.get::<_, String>("name"))?
        .collect::<Result<Vec<_>, _>>()?;
    if !columns.iter().any(|existing| existing == column) {
        connection.execute(
            &format!("alter table {table} add column {column} {definition}"),
            [],
        )?;
    }
    Ok(())
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
        || training_job_is_mlx_eligible(job)
        || caption_job_is_mlx_eligible(job)
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
        // MLX-agnostic job types: metadata/utility work, ffmpeg, and prompt refine run
        // in-process on macOS with no Python torch dependency.
        JobType::Placeholder
        | JobType::ModelDownload
        | JobType::ModelImport
        | JobType::LoraImport
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

        JobType::ImageVqa | JobType::ImageInterleave => Err(UnsupportedReason::new(
            model,
            "image understanding / interleave",
            "image VQA / interleaved generation is the SenseNova-U1 understanding surface; it lands with the SenseNova port.",
            Some("epic 3180"),
        )),

        JobType::VideoGenerate => Err(classify_video_gap(&job.payload)),

        JobType::VideoExtend | JobType::VideoBridge => Err(UnsupportedReason::new(
            model,
            "advanced video (extend / bridge)",
            "video_extend / video_bridge are torch-only advanced video modes.",
            Some("epic 3040"),
        )),

        JobType::PersonReplace => Err(UnsupportedReason::new(
            model,
            "replace_person",
            "person replacement is a torch-only advanced video mode (also depends on person detect/track).",
            Some("epic 3040 (+ sc-3488)"),
        )),

        JobType::PersonDetect | JobType::PersonTrack => Err(UnsupportedReason::new(
            None,
            "person detect / track (YOLO/SAM2)",
            "person detection/tracking runs on the Python onnxruntime/torch path.",
            Some("sc-3488"),
        )),

        // DWPose pose detection is now ported to the Rust worker (sc-3487): RTMW
        // whole-body via `ort`/CoreML on the macOS MLX worker, so the Pose Library
        // "create from photo" flow + InstantID pose conditioning run Python-free.
        JobType::PoseDetect => Ok(()),

        JobType::ImageUpscale => Err(UnsupportedReason::new(
            model,
            "image_upscale (Real-ESRGAN)",
            "standalone image upscaling runs on the Python torch Real-ESRGAN / AuraSR path.",
            Some("sc-3489"),
        )),

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
    /// Third-party LyCORIS (LoHa / non-peft LoKr) adapters — never in the Rust flow yet (sc-3537).
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
            lycoris: false,
            video_modes: BTreeMap::new(),
        },
    }
}

/// The `video_generate` modes the UI offers, in display order, so the gating mirrors
/// [`video_mode_is_mlx_eligible`] for every mode a Mac user could pick.
const VIDEO_UI_MODES: &[&str] = &[
    "text_to_video",
    "image_to_video",
    "first_last_frame",
    "replace_person",
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
    let mut features = BTreeMap::new();
    features.insert(
        "lycoris".to_owned(),
        MacFeatureSupport::unsupported(
            "third-party LyCORIS LoRA",
            "the Rust engine/worker apply LoRA + peft LoKr, but not arbitrary third-party LyCORIS (LoHa / non-peft LoKr).",
            "sc-3537",
        ),
    );
    features.insert(
        "imageUpscale".to_owned(),
        MacFeatureSupport::unsupported(
            "image_upscale (Real-ESRGAN)",
            "standalone image upscaling runs on the Python torch Real-ESRGAN / AuraSR path.",
            "sc-3489",
        ),
    );
    features.insert(
        "poseFromPhoto".to_owned(),
        MacFeatureSupport::unsupported(
            "DWPose pose detection",
            "photo→skeleton pose detection runs on Python onnxruntime (DWPose).",
            "sc-3487",
        ),
    );
    features.insert(
        "personDetect".to_owned(),
        MacFeatureSupport::unsupported(
            "person detect / track (YOLO/SAM2)",
            "person detection/tracking runs on the Python onnxruntime/torch path.",
            "sc-3488",
        ),
    );
    features.insert(
        "datasetCaptioning".to_owned(),
        MacFeatureSupport {
            supported: true,
            reason: None,
        },
    );
    features.insert(
        "advancedVideoModes".to_owned(),
        MacFeatureSupport::unsupported(
            "advanced video (extend / bridge / replace_person)",
            "the advanced video job types and modes are torch-only.",
            "epic 3040",
        ),
    );
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
fn torch_only_image_model_epic(model: &str) -> Option<&'static str> {
    match model {
        "kolors" => Some("epic 3532"),
        "instantid_realvisxl" => Some("epic 3109"),
        "pulid_flux_dev" => Some("epic 3069"),
        "z_image_edit" => Some("epic 3529"),
        m if m.starts_with("sensenova") => Some("epic 3180"),
        m if m.starts_with("lens") => Some("epic 3164"),
        m if m.starts_with("chroma") => Some("epic 3531"),
        _ => None,
    }
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
    if request_has_lycoris_lora(payload) {
        return UnsupportedReason::new(
            Some(model),
            "third-party LyCORIS LoRA",
            "the Rust engine/worker apply LoRA + peft LoKr, but not arbitrary third-party LyCORIS (LoHa / non-peft LoKr) — port-or-drop spike.",
            Some("sc-3537"),
        );
    }
    let is_edit = payload.get("mode").and_then(Value::as_str) == Some("edit_image");
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
        "z_image_turbo" if is_edit => UnsupportedReason::new(
            Some(model),
            "edit_image",
            "Z-Image img2img edit stays on the Python torch path (folds into the Z-Image-Edit port).",
            Some("epic 3529"),
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
        // flux2 / sdxl / realvisxl only fall out via LyCORIS (handled above) — defensive.
        _ => UnsupportedReason::new(
            Some(model),
            "unsupported configuration",
            "this model/feature combination is not in the Rust/MLX flow.",
            None,
        ),
    }
}

/// Name the precise gap for an ineligible `video_generate` job: a torch-only model (incl. SVD),
/// an advanced mode, a third-party LyCORIS, or LoKr-on-Wan. Mirrors `video_job_is_mlx_eligible`.
fn classify_video_gap(payload: &Map<String, Value>) -> UnsupportedReason {
    let Some(model) = payload.get("model").and_then(Value::as_str) else {
        return UnsupportedReason::new(None, "video generation", "no model specified.", None);
    };
    if !VIDEO_MLX_ROUTED_MODELS.contains(&model) {
        return UnsupportedReason::new(
            Some(model),
            "torch-only video model (incl. SVD)",
            "this video model has no Rust/MLX engine (e.g. SVD); it runs on the Python torch path.",
            Some("epic 3040"),
        );
    }
    let mode = payload
        .get("mode")
        .and_then(Value::as_str)
        .unwrap_or("image_to_video");
    if !matches!(mode, "text_to_video" | "image_to_video") {
        return UnsupportedReason::new(
            Some(model),
            "advanced video mode",
            "advanced video_generate modes (first_last_frame / replace_person) are torch-only.",
            Some("epic 3040"),
        );
    }
    if request_has_lycoris_lora(payload) {
        return UnsupportedReason::new(
            Some(model),
            "third-party LyCORIS LoRA",
            "the mlx-video path can't apply arbitrary third-party LyCORIS adapters — port-or-drop spike.",
            Some("sc-3537"),
        );
    }
    if is_wan_video_model(model) && request_has_lokr_lora(payload) {
        return UnsupportedReason::new(
            Some(model),
            "LoKr-on-Wan",
            "the mlx Wan path can't merge a Kronecker (LoKr) adapter; LoKr-on-Wan stays on torch.",
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
        Some("kolors_lora") => UnsupportedReason::new(
            None,
            "Kolors LoRA training",
            "the Kolors trainer (SDXL + ChatGLM3) has no mlx-gen Rust trainer.",
            Some("epic 3039"),
        ),
        Some("lens_lora") => UnsupportedReason::new(
            None,
            "Lens LoRA training",
            "the Lens trainer runs in a Python sidecar with no mlx-gen Rust trainer.",
            Some("epic 3039"),
        ),
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
    for candidate in candidates {
        if !worker_supports_job(&candidate, job)
            || active_gpu_job_exists(connection, &candidate.gpu_id)?
        {
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
    for candidate in candidates {
        if worker_supports_job(&candidate, job)
            && !active_gpu_job_exists(connection, &candidate.gpu_id)?
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn active_gpu_job_exists(connection: &Connection, gpu_id: &str) -> JobsStoreResult<bool> {
    Ok(connection
        .query_row(
            &format!(
                "
            select id from jobs
             where assigned_gpu = ?1
               and status in ('preparing', 'downloading', 'loading_model', 'running', 'saving')
               and type not in ({})
             limit 1
            ",
                non_gpu_job_types_sql()
            ),
            params![gpu_id],
            |_row| Ok(()),
        )
        .optional()?
        .is_some())
}

/// Models the in-process Rust MLX worker generates today, by id. This set grows
/// one family story at a time as each lands real generation in
/// `sceneworks-worker::image_jobs` — sc-3022 Z-Image, sc-3023 FLUX.1, sc-3024 Qwen,
/// sc-3025 FLUX.2, sc-3026 SDXL (live). A model id absent here is never routed to the
/// mlx worker, so the Python torch path stays authoritative for it.
const MLX_ROUTED_MODELS: &[&str] = &[
    "z_image_turbo",
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
        "z_image_turbo" => z_image_mlx_eligible(payload),
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
        // Every model in MLX_ROUTED_MODELS must have an arm.
        _ => false,
    }
}

/// Does this `image_detail` job belong on the in-process Rust MLX worker? sc-3060 (epic 3041)
/// ports the tile-ControlNet detail refine onto the engine. Detail is SDXL-family only
/// (`sdxl` / `realvisxl`, the detail-capable backbones; the payload defaults to `realvisxl`),
/// and a third-party LyCORIS LoRA falls back to torch like every other SDXL shape. On
/// Windows/Linux no `mlx` worker exists, so detail stays on the Python torch path.
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
    matches!(model, "sdxl" | "realvisxl") && !request_has_lycoris_lora(&job.payload)
}

/// Whether the in-process MLX worker can serve this GPU job (image_generate or image_detail).
fn job_is_mlx_eligible(job: &JobSnapshot) -> bool {
    image_job_is_mlx_eligible(job) || image_detail_mlx_eligible(job)
}

/// SDXL MLX-routing conditions. sc-3026 brought txt2img + LoRA; sc-3060 (epic 3041) adds the
/// advanced shapes the Rust `mlx-gen-sdxl` engine now handles — reference/IP-Adapter, img2img
/// `edit_image`, masked inpaint, and outpaint — so they route to the in-process MLX worker on
/// Mac instead of the Python torch `SdxlDiffusersAdapter`. The torch path stays authoritative
/// on Windows/Linux (no `mlx` worker registered → nothing defers) and as the Mac fallback.
/// A third-party LyCORIS LoRA still falls back to torch (the engine/worker apply LoRA + peft
/// LoKr natively, but not arbitrary LyCORIS); unlike the old Python gate, peft LoKr stays on MLX.
/// `image_detail` is a separate job type with its own routing (see `image_detail_mlx_eligible`).
fn sdxl_mlx_eligible(payload: &Map<String, Value>) -> bool {
    !request_has_lycoris_lora(payload)
}

/// FLUX.2-klein MLX-routing conditions. FLUX.2-klein is an **MLX-only** family (no
/// torch backend), so everything it does runs on MLX: txt2img (sc-3025) AND
/// edit/reference + KV-cache + multi-reference (sc-3029). The only exclusion is a
/// third-party LyCORIS LoRA neither the engine nor the worker can apply.
fn flux2_mlx_eligible(payload: &Map<String, Value>) -> bool {
    !request_has_lycoris_lora(payload)
}

/// Qwen-Image (sc-3024 / strict pose sc-3575) MLX-routing conditions: text-to-image,
/// plus the base-Qwen strict pose tier (`advanced.poses`) handled by the `qwen_image_control`
/// engine variant. A reference without poses (character/edit flow) and `edit_image` stay on
/// the Python torch path. A third-party LyCORIS LoRA also falls back to torch (engine + worker
/// apply LoRA + peft LoKr, but not arbitrary LyCORIS).
fn qwen_mlx_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    if request_has_lycoris_lora(payload) {
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
/// (sc-3398) shares the same gate (its sampler + distill-LoRA are worker-local). A
/// third-party LyCORIS LoRA falls back to torch (engine + worker apply LoRA + peft LoKr, but
/// not arbitrary LyCORIS).
fn qwen_edit_mlx_eligible(payload: &Map<String, Value>) -> bool {
    if request_has_lycoris_lora(payload) {
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
/// A third-party LyCORIS LoRA falls back to torch; SceneWorks peft LoKr stays MLX.
fn flux_mlx_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    !request_has_lycoris_lora(payload)
}

/// Z-Image (sc-3022) MLX-routing conditions, ported from
/// `_should_route_z_image_to_mlx`: text-to-image, reference-identity img2img-init
/// (sc-3619 — `referenceAssetId` without a pose set, the plain img2img path the
/// base engine already supports), and reference+pose (the Fun-ControlNet pose tier
/// lives only on MLX — sc-2257/sc-2328, so a reference+pose job must NOT divert to
/// torch, which would honour count while dropping the poses). `edit_image`
/// img2img-edit stays on torch (epic 3529); a third-party LyCORIS LoRA falls back
/// to torch while SceneWorks peft LoKr stays on MLX (sc-2216).
fn z_image_mlx_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    !request_has_lycoris_lora(payload)
}

/// True when any LoRA in the request is a third-party LyCORIS adapter, by its
/// *recorded* `networkType` (`networkType`, or `compatibility.networkType`). A
/// `lokr` stamp is SceneWorks peft LoKr, which applies natively on the MLX
/// Z-Image path (sc-2216) and does NOT force torch. Mirrors the worker's
/// `_request_has_lycoris_lora` minus the safetensors-header sniff (the API has
/// no file access — see `image_job_is_mlx_eligible`).
fn request_has_lycoris_lora(payload: &Map<String, Value>) -> bool {
    let Some(loras) = payload.get("loras").and_then(Value::as_array) else {
        return false;
    };
    loras.iter().any(|lora| {
        lora.as_object()
            .and_then(|lora| {
                lora.get("networkType").and_then(Value::as_str).or_else(|| {
                    lora.get("compatibility")
                        .and_then(Value::as_object)
                        .and_then(|compat| compat.get("networkType"))
                        .and_then(Value::as_str)
                })
            })
            .is_some_and(|recorded| recorded.trim().eq_ignore_ascii_case("lycoris"))
    })
}

/// Whether any request LoRA records `networkType == "lokr"` (SceneWorks peft LoKr).
/// Sibling of [`request_has_lycoris_lora`]; used by the video routing to keep a
/// LoKr-on-Wan job on the torch path (see [`video_job_is_mlx_eligible`]).
fn request_has_lokr_lora(payload: &Map<String, Value>) -> bool {
    let Some(loras) = payload.get("loras").and_then(Value::as_array) else {
        return false;
    };
    loras.iter().any(|lora| {
        lora.as_object()
            .and_then(|lora| {
                lora.get("networkType").and_then(Value::as_str).or_else(|| {
                    lora.get("compatibility")
                        .and_then(Value::as_object)
                        .and_then(|compat| compat.get("networkType"))
                        .and_then(Value::as_str)
                })
            })
            .is_some_and(|recorded| recorded.trim().eq_ignore_ascii_case("lokr"))
    })
}

/// Video models the in-process Rust MLX worker generates today (sc-3034 Wan2.2,
/// sc-3035 LTX-2.3 + audio). Mirrors `MlxVideoAdapter._supported_models`. A model
/// id absent here is never routed to the mlx worker — the Python torch path stays
/// authoritative for it (and for SVD, which has no MLX crate).
const VIDEO_MLX_ROUTED_MODELS: &[&str] = &[
    "ltx_2_3",
    "ltx_2_3_eros",
    "wan_2_2",
    "wan_2_2_t2v_14b",
    "wan_2_2_i2v_14b",
];

/// Whether `model` is a Wan2.2 video family id (vs LTX).
fn is_wan_video_model(model: &str) -> bool {
    model.starts_with("wan")
}

/// Epic 3018 routing (sc-3036, the video sibling of [`image_job_is_mlx_eligible`]):
/// does this video job belong on the in-process Rust MLX worker? Encodes today's
/// Python `create_video_adapter` MLX-eligibility (video_adapters.py) at the claim
/// layer, minus the worker-local gates (MPS presence / sidecar) — those are now
/// expressed by whether an `mlx` worker is registered and idle (see
/// [`should_defer_video_to_mlx_worker`]).
///
/// MLX covers `text_to_video` + `image_to_video` on Wan/LTX, plus `first_last_frame`
/// on the FLF-capable engines (LTX + Wan TI2V-5B `wan_2_2`; sc-3055 cutover — see
/// [`video_mode_is_mlx_eligible`]). Still on the Python torch path: the advanced job
/// types (`video_extend`/`video_bridge`/`person_replace`) and the `replace_person`
/// mode (a later Wan-VACE cutover slice), SVD (svd_xt not linked in the worker yet), a
/// non-MLX model, a third-party LyCORIS LoRA (the mlx worker's `classify_adapter`
/// rejects it), and **LoKr-on-Wan** (the diffusers-Wan path applies LoKr via PEFT;
/// the mlx-video path can't — mirrors `create_video_adapter`). LoKr-on-LTX stays
/// MLX (the torch LTX path has no LoKr loader; the Rust engine applies it natively).
fn video_job_is_mlx_eligible(job: &JobSnapshot) -> bool {
    // Only the base video_generate job type is MLX-eligible; the advanced job types
    // (extend/bridge/person-replace) are torch-only.
    if !matches!(job.job_type, JobType::VideoGenerate) {
        return false;
    }
    let Some(model) = job.payload.get("model").and_then(Value::as_str) else {
        return false;
    };
    if !VIDEO_MLX_ROUTED_MODELS.contains(&model) {
        return false;
    }
    // Mode defaults to `image_to_video`, mirroring `video_request_from_job`.
    let mode = job
        .payload
        .get("mode")
        .and_then(Value::as_str)
        .unwrap_or("image_to_video");
    if !video_mode_is_mlx_eligible(model, mode) {
        return false;
    }
    if request_has_lycoris_lora(&job.payload) {
        return false;
    }
    if is_wan_video_model(model) && request_has_lokr_lora(&job.payload) {
        return false;
    }
    true
}

/// Which `video_generate` modes the in-process Rust MLX worker serves for `model`. Every
/// routed model serves `text_to_video` + `image_to_video` (sc-3034/3035); `first_last_frame`
/// is additionally MLX on the FLF-capable engines — LTX (`ltx_2_3`/`ltx_2_3_eros`, the
/// reference-grounded `Keyframe` path, sc-3052) and Wan TI2V-5B (`wan_2_2`, the mask-blend
/// multi-keyframe path, sc-3357). The 14B Wan MoE engines have no `Keyframe` path, so FLF on
/// them stays torch. The advanced clip modes (`extend_clip`/`video_bridge`) + `replace_person`
/// ride dedicated job types / the Wan-VACE path and are separate cutover slices (sc-3055).
fn video_mode_is_mlx_eligible(model: &str, mode: &str) -> bool {
    match mode {
        "text_to_video" | "image_to_video" => true,
        "first_last_frame" => matches!(model, "ltx_2_3" | "ltx_2_3_eros" | "wan_2_2"),
        _ => false,
    }
}

/// SceneWorks training kernels with a native mlx-gen Rust trainer (epic 3039):
/// the engine registers `z_image_turbo`/`sdxl`/`ltx_2_3`/`wan2_2_*` trainers, which
/// the worker reaches via these SceneWorks kernel ids (the mlx worker maps kernel +
/// base model → engine trainer id). `kolors_lora` (SDXL + ChatGLM3) and `lens_lora`
/// (sidecar) have no mlx-gen crate, so they stay on the Python torch worker. A
/// kernel absent here is never routed to the mlx worker.
const MLX_ROUTED_TRAINING_KERNELS: &[&str] = &[
    "z_image_lora",
    "sdxl_lora",
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
        // torch-only edit model (z_image_edit/kolors/sensenova/lens/pulid/instantid)
        // is not MLX-eligible, so the mlx worker refuses it and it stays on torch.
        if matches!(
            job.job_type,
            JobType::ImageGenerate | JobType::ImageEdit | JobType::ImageDetail
        ) && !job_is_mlx_eligible(job)
        {
            return false;
        }
        // Video (sc-3036): the mlx worker claims only MLX-eligible `video_generate`
        // jobs (Wan/LTX text_to_video / image_to_video). The advanced video job types
        // (extend / bridge / person-replace) and torch-only `video_generate` cases
        // (advanced modes, SVD, non-MLX model, LoKr-on-Wan) stay on the Python worker.
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
        // (z_image / sdxl / wan / ltx) via `mlx_gen::load_trainer`. `kolors_lora` +
        // `lens_lora` (no mlx-gen crate) and LoKr-on-Wan stay on the Python torch
        // worker. Applies to both dry-run and real runs.
        if matches!(job.job_type, JobType::LoraTrain) && !training_job_is_mlx_eligible(job) {
            return false;
        }
        // Dataset captioning (sc-3556): the mlx worker claims only JoyCaption jobs
        // backed by the mlx-gen provider. Any future non-JoyCaption captioner stays
        // on the worker that advertises that capability.
        if matches!(job.job_type, JobType::TrainingCaption) && !caption_job_is_mlx_eligible(job) {
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
mod mlx_routing_tests {
    use super::{
        flux2_mlx_eligible, flux_mlx_eligible, qwen_edit_mlx_eligible, qwen_mlx_eligible,
        request_has_lycoris_lora, sdxl_mlx_eligible, video_mode_is_mlx_eligible,
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
    fn z_image_edit_mode_is_not_eligible() {
        assert!(!z_image_mlx_eligible(&object(
            json!({ "mode": "edit_image" })
        )));
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
    fn z_image_third_party_lycoris_falls_back_but_lokr_stays() {
        // SceneWorks peft LoKr applies natively on the MLX Z-Image path → eligible.
        assert!(z_image_mlx_eligible(&object(json!({
            "loras": [{ "path": "a.safetensors", "networkType": "lokr" }]
        }))));
        // Third-party LyCORIS has no native MLX loader → torch.
        assert!(!z_image_mlx_eligible(&object(json!({
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
    fn flux_edit_and_lycoris_fall_back_to_torch() {
        // edit_image (no FLUX.1 edit on any platform — future Kontext) and third-party
        // LyCORIS stay on Python. Reference does NOT fall back anymore (see above).
        assert!(!flux_mlx_eligible(&object(json!({ "mode": "edit_image" }))));
        assert!(!flux_mlx_eligible(&object(json!({
            "loras": [{ "networkType": "lycoris" }]
        }))));
        // A reference with a LyCORIS LoRA still falls back (LyCORIS forces torch).
        assert!(!flux_mlx_eligible(&object(json!({
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
    fn qwen_edit_reference_and_lycoris_fall_back_but_pose_routes_mlx() {
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
        assert!(!qwen_mlx_eligible(&object(json!({
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
    fn qwen_edit_without_reference_or_with_lycoris_falls_back_to_torch() {
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
        // Third-party LyCORIS has no native MLX loader → torch.
        assert!(!qwen_edit_mlx_eligible(&object(json!({
            "mode": "edit_image", "sourceAssetId": "src_1",
            "loras": [{ "networkType": "lycoris" }]
        }))));
    }

    #[test]
    fn flux2_txt2img_and_edit_are_eligible_lycoris_is_not() {
        // FLUX.2 is MLX-only: txt2img (sc-3025) AND edit/reference (sc-3029) all route MLX.
        assert!(flux2_mlx_eligible(&object(
            json!({ "prompt": "a red fox" })
        )));
        assert!(flux2_mlx_eligible(&object(json!({ "mode": "edit_image" }))));
        assert!(flux2_mlx_eligible(&object(
            json!({ "referenceAssetId": "asset_1" })
        )));
        // Only a third-party LyCORIS LoRA (unapplicable on the MLX path) is excluded.
        assert!(!flux2_mlx_eligible(&object(json!({
            "loras": [{ "networkType": "lycoris" }]
        }))));
    }

    #[test]
    fn sdxl_eligible_for_txt2img_edit_reference_lokr_but_not_lycoris() {
        assert!(sdxl_mlx_eligible(&object(json!({ "prompt": "a red fox" }))));
        // peft LoKr stays on MLX (the Rust SDXL path supports LoKr, unlike the old
        // vendored path) — only third-party LyCORIS falls back to torch.
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
        // A third-party LyCORIS LoRA still falls back to torch, even on an edit job.
        assert!(!sdxl_mlx_eligible(&object(json!({
            "loras": [{ "networkType": "lycoris" }]
        }))));
        assert!(!sdxl_mlx_eligible(&object(json!({
            "mode": "edit_image",
            "loras": [{ "networkType": "lycoris" }]
        }))));
    }

    #[test]
    fn lycoris_detection_reads_recorded_network_type_only() {
        assert!(!request_has_lycoris_lora(&Map::new()));
        assert!(!request_has_lycoris_lora(&object(json!({ "loras": [] }))));
        // Recorded directly on the LoRA.
        assert!(request_has_lycoris_lora(&object(json!({
            "loras": [{ "networkType": "LyCORIS" }]
        }))));
        // Recorded under compatibility.networkType.
        assert!(request_has_lycoris_lora(&object(json!({
            "loras": [{ "compatibility": { "networkType": "lycoris" } }]
        }))));
        // lokr is SceneWorks peft LoKr, not third-party LyCORIS.
        assert!(!request_has_lycoris_lora(&object(json!({
            "loras": [{ "networkType": "lokr" }]
        }))));
        // No recorded type → the API can't sniff the header → treated as not-lycoris
        // (the mlx worker's classify_adapter backstops an unstamped file).
        assert!(!request_has_lycoris_lora(&object(json!({
            "loras": [{ "path": "unstamped.safetensors" }]
        }))));
    }

    #[test]
    fn video_mode_eligibility_admits_flf_only_on_flf_capable_engines() {
        // Base modes are MLX on every routed model.
        for model in VIDEO_MLX_ROUTED_MODELS {
            assert!(video_mode_is_mlx_eligible(model, "text_to_video"));
            assert!(video_mode_is_mlx_eligible(model, "image_to_video"));
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
        // The advanced clip modes + replace_person are not video_generate-mode-eligible
        // (they ride dedicated job types / the Wan-VACE path — separate cutover slices).
        for mode in ["extend_clip", "video_bridge", "replace_person", "nonsense"] {
            assert!(!video_mode_is_mlx_eligible("ltx_2_3", mode));
        }
    }
}
