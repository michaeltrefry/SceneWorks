from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

from sceneworks_shared import utc_now

ACTIVE_STATUSES = ("preparing", "downloading", "loading_model", "running", "saving")
TERMINAL_STATUSES = ("completed", "failed", "canceled", "interrupted")
JOB_STATUSES = ("queued", *ACTIVE_STATUSES, *TERMINAL_STATUSES)
NON_GPU_JOB_TYPES = ("model_download", "lora_import")
MAX_JOB_ATTEMPTS = 5


def desired_model_keys(payload: dict) -> set[str]:
    keys: set[str] = set()
    model = payload.get("model")
    if isinstance(model, str) and model:
        keys.add(model)
    repo = payload.get("repo")
    if isinstance(repo, str) and repo:
        keys.add(repo)
    advanced = payload.get("advanced")
    if isinstance(advanced, dict):
        for field in ("modelRepo", "repo"):
            value = advanced.get(field)
            if isinstance(value, str) and value:
                keys.add(value)
    return keys


def job_matches_loaded_model(row: sqlite3.Row, loaded_models: set[str]) -> bool:
    if row["requested_gpu"] != "auto" or row["type"] in NON_GPU_JOB_TYPES or not loaded_models:
        return False
    return bool(desired_model_keys(loads(row["payload_json"], {})) & loaded_models)


def choose_claimable_job(rows: list[sqlite3.Row], capabilities: set[str], loaded_models: set[str]) -> sqlite3.Row | None:
    compatible = [row for row in rows if row["type"] in capabilities]
    if not compatible:
        return None

    first = compatible[0]
    if first["type"] in NON_GPU_JOB_TYPES or first["requested_gpu"] != "auto":
        return first

    explicit_gpu_job = next(
        (row for row in compatible if row["type"] not in NON_GPU_JOB_TYPES and row["requested_gpu"] != "auto"),
        None,
    )
    if explicit_gpu_job is not None:
        return explicit_gpu_job

    return next((row for row in compatible if job_matches_loaded_model(row, loaded_models)), first)


def dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {}, separators=(",", ":"), sort_keys=True)


def loads(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    return json.loads(value)


class JobsStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        self._use_wal = True

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = self.open_connection()
        if self._use_wal:
            try:
                connection.execute("pragma journal_mode = wal")
            except sqlite3.OperationalError:
                connection.close()
                self._use_wal = False
                self.remove_sqlite_sidecars()
                connection = self.open_connection()
                connection.execute("pragma journal_mode = delete")
        connection.execute("pragma foreign_keys = on")
        return connection

    def open_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def remove_sqlite_sidecars(self) -> None:
        for suffix in ("-wal", "-shm"):
            try:
                self.db_path.with_name(f"{self.db_path.name}{suffix}").unlink(missing_ok=True)
            except OSError:
                pass

    def initialize(self) -> None:
        with self._lock, self.connect() as connection:
            connection.executescript(
                """
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
                  registered_at text not null,
                  last_seen_at text not null
                );
                """
            )

    def mark_interrupted_on_startup(self) -> list[dict]:
        now = utc_now()
        with self._lock, self.connect() as connection:
            rows = connection.execute(
                f"select * from jobs where status in ({','.join('?' for _ in ACTIVE_STATUSES)})",
                ACTIVE_STATUSES,
            ).fetchall()
            connection.execute(
                f"""
                update jobs
                   set status = 'interrupted',
                       stage = 'interrupted',
                       message = 'Job was interrupted by a backend restart.',
                       error = 'The backend restarted before this job finished.',
                       completed_at = ?,
                       updated_at = ?,
                       worker_id = null
                 where status in ({','.join('?' for _ in ACTIVE_STATUSES)})
                """,
                (now, now, *ACTIVE_STATUSES),
            )
            connection.execute(
                "update workers set status = 'offline', current_job_id = null where status != 'offline'"
            )
            return [self.row_to_job(row) for row in rows]

    def create_job(
        self,
        *,
        job_type: str,
        project_id: str | None,
        project_name: str | None,
        payload: dict,
        requested_gpu: str,
        source_job_id: str | None = None,
        duplicate_of_job_id: str | None = None,
        attempts: int = 1,
    ) -> dict:
        now = utc_now()
        job_id = f"job_{uuid4().hex}"
        with self._lock, self.connect() as connection:
            connection.execute(
                """
                insert into jobs (
                  id, type, status, project_id, project_name, payload_json, result_json,
                  requested_gpu, progress, stage, message, attempts, source_job_id,
                  duplicate_of_job_id, created_at, updated_at
                ) values (?, ?, 'queued', ?, ?, ?, '{}', ?, 0, 'queued', ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    job_type,
                    project_id,
                    project_name,
                    dumps(payload),
                    requested_gpu or "auto",
                    "Waiting for an available worker.",
                    attempts,
                    source_job_id,
                    duplicate_of_job_id,
                    now,
                    now,
                ),
            )
            return self.get_job(job_id, connection=connection)

    def list_jobs(
        self,
        *,
        project_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        filters = []
        params: list[Any] = []
        if project_id:
            filters.append("project_id = ?")
            params.append(project_id)
        if status:
            filters.append("status = ?")
            params.append(status)

        where_clause = f"where {' and '.join(filters)}" if filters else ""
        with self._lock, self.connect() as connection:
            rows = connection.execute(
                f"select * from jobs {where_clause} order by created_at desc limit ?",
                (*params, max(1, min(limit, 500))),
            ).fetchall()
        return [self.row_to_job(row) for row in rows]

    def get_job(self, job_id: str, *, connection: sqlite3.Connection | None = None) -> dict:
        owns_connection = connection is None
        connection = connection or self.connect()
        try:
            row = connection.execute("select * from jobs where id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            return self.row_to_job(row)
        finally:
            if owns_connection:
                connection.close()

    def cancel_job(self, job_id: str) -> dict:
        now = utc_now()
        with self._lock, self.connect() as connection:
            job = self.get_job(job_id, connection=connection)
            if job["status"] in TERMINAL_STATUSES:
                return job

            if job["status"] == "queued":
                connection.execute(
                    """
                    update jobs
                       set status = 'canceled',
                           stage = 'canceled',
                           progress = 1,
                           cancel_requested = 1,
                           message = 'Canceled before a worker started.',
                           canceled_at = ?,
                           completed_at = ?,
                           updated_at = ?
                     where id = ?
                    """,
                    (now, now, now, job_id),
                )
            else:
                connection.execute(
                    """
                    update jobs
                       set cancel_requested = 1,
                           message = 'Cancellation requested. Waiting for worker acknowledgement.',
                           updated_at = ?
                     where id = ?
                    """,
                    (now, job_id),
                )
            return self.get_job(job_id, connection=connection)

    def retry_job(self, job_id: str) -> dict:
        with self._lock, self.connect() as connection:
            job = self.get_job(job_id, connection=connection)
        if job["attempts"] >= MAX_JOB_ATTEMPTS:
            raise ValueError(f"Job retry limit reached after {MAX_JOB_ATTEMPTS} attempts.")
        return self.create_job(
            job_type=job["type"],
            project_id=job["projectId"],
            project_name=job["projectName"],
            payload=job["payload"],
            requested_gpu=job["requestedGpu"],
            source_job_id=job["id"],
            attempts=job["attempts"] + 1,
        )

    def duplicate_job(self, job_id: str, *, payload_changes: dict, requested_gpu: str | None) -> dict:
        with self._lock, self.connect() as connection:
            job = self.get_job(job_id, connection=connection)
        payload = {**job["payload"], **payload_changes}
        return self.create_job(
            job_type=job["type"],
            project_id=job["projectId"],
            project_name=job["projectName"],
            payload=payload,
            requested_gpu=requested_gpu or job["requestedGpu"],
            duplicate_of_job_id=job["id"],
        )

    def register_worker(
        self,
        *,
        worker_id: str,
        gpu_id: str,
        gpu_name: str | None,
        capabilities: list[str],
        loaded_models: list[str],
    ) -> dict:
        now = utc_now()
        with self._lock, self.connect() as connection:
            connection.execute(
                """
                insert into workers (
                  id, gpu_id, gpu_name, status, current_job_id, capabilities_json,
                  loaded_models_json, registered_at, last_seen_at
                ) values (?, ?, ?, 'idle', null, ?, ?, ?, ?)
                on conflict(id) do update set
                  gpu_id = excluded.gpu_id,
                  gpu_name = excluded.gpu_name,
                  status = case when workers.current_job_id is null then 'idle' else workers.status end,
                  capabilities_json = excluded.capabilities_json,
                  loaded_models_json = excluded.loaded_models_json,
                  last_seen_at = excluded.last_seen_at
                """,
                (
                    worker_id,
                    gpu_id,
                    gpu_name,
                    dumps(capabilities),
                    dumps(loaded_models),
                    now,
                    now,
                ),
            )
            return self.get_worker(worker_id, connection=connection)

    def heartbeat_worker(
        self,
        *,
        worker_id: str,
        status: str,
        current_job_id: str | None,
        loaded_models: list[str],
    ) -> dict:
        now = utc_now()
        with self._lock, self.connect() as connection:
            worker = self.get_worker(worker_id, connection=connection)
            previous_job_id = worker["currentJobId"]
            if not current_job_id and previous_job_id:
                previous_job = self.get_job(previous_job_id, connection=connection)
                if previous_job["status"] in ACTIVE_STATUSES:
                    connection.execute(
                        """
                        update jobs
                           set status = 'interrupted',
                               stage = 'interrupted',
                               message = 'Job was interrupted after its worker restarted.',
                               error = 'Worker heartbeat no longer referenced the active job.',
                               completed_at = ?,
                               updated_at = ?,
                               worker_id = null
                         where id = ?
                        """,
                        (now, now, previous_job_id),
                    )
            connection.execute(
                """
                update workers
                   set status = ?,
                       current_job_id = ?,
                       loaded_models_json = ?,
                       last_seen_at = ?
                 where id = ?
                """,
                (status, current_job_id, dumps(loaded_models), now, worker_id),
            )
            if current_job_id:
                connection.execute(
                    "update jobs set last_heartbeat_at = ?, updated_at = ? where id = ?",
                    (now, now, current_job_id),
                )
            return self.get_worker(worker_id, connection=connection)

    def mark_stale_workers_interrupted(self, timeout_seconds: int) -> dict[str, list[dict]]:
        now = datetime.now(UTC).replace(microsecond=0)
        cutoff = (now - timedelta(seconds=max(1, timeout_seconds))).isoformat().replace("+00:00", "Z")
        now_text = now.isoformat().replace("+00:00", "Z")
        with self._lock, self.connect() as connection:
            stale_workers = connection.execute(
                """
                select * from workers
                 where status != 'offline'
                   and last_seen_at < ?
                """,
                (cutoff,),
            ).fetchall()
            if not stale_workers:
                return {"workers": [], "jobs": []}

            worker_ids = [row["id"] for row in stale_workers]
            active_jobs = connection.execute(
                f"""
                select * from jobs
                 where worker_id in ({','.join('?' for _ in worker_ids)})
                   and status in ({','.join('?' for _ in ACTIVE_STATUSES)})
                """,
                (*worker_ids, *ACTIVE_STATUSES),
            ).fetchall()
            connection.execute(
                f"""
                update jobs
                   set status = 'interrupted',
                       stage = 'interrupted',
                       message = 'Job was interrupted after its worker stopped sending heartbeats.',
                       error = 'Worker heartbeat timed out.',
                       completed_at = ?,
                       updated_at = ?,
                       worker_id = null
                 where worker_id in ({','.join('?' for _ in worker_ids)})
                   and status in ({','.join('?' for _ in ACTIVE_STATUSES)})
                """,
                (now_text, now_text, *worker_ids, *ACTIVE_STATUSES),
            )
            connection.execute(
                f"""
                update workers
                   set status = 'offline',
                       current_job_id = null,
                       last_seen_at = ?
                 where id in ({','.join('?' for _ in worker_ids)})
                """,
                (now_text, *worker_ids),
            )
            updated_workers = [
                self.get_worker(worker_id, connection=connection)
                for worker_id in worker_ids
            ]
            updated_jobs = [self.get_job(row["id"], connection=connection) for row in active_jobs]
            return {"workers": updated_workers, "jobs": updated_jobs}

    def claim_next_job(self, worker_id: str) -> dict | None:
        now = utc_now()
        with self._lock, self.connect() as connection:
            worker = self.get_worker(worker_id, connection=connection)
            gpu_id = worker["gpuId"]
            capabilities = set(worker["capabilities"])
            loaded_models = set(worker["loadedModels"])
            active_gpu_job = connection.execute(
                f"""
                select id from jobs
                 where assigned_gpu = ?
                   and status in ({','.join('?' for _ in ACTIVE_STATUSES)})
                   and type not in ({','.join('?' for _ in NON_GPU_JOB_TYPES)})
                 limit 1
                """,
                (gpu_id, *ACTIVE_STATUSES, *NON_GPU_JOB_TYPES),
            ).fetchone()

            queued_rows = connection.execute(
                f"""
                select * from jobs
                 where status = 'queued'
                   and (type in ({','.join('?' for _ in NON_GPU_JOB_TYPES)}) or requested_gpu = 'auto' or requested_gpu = ?)
                   and (? = 0 or type in ({','.join('?' for _ in NON_GPU_JOB_TYPES)}))
                 order by created_at asc
                 limit 50
                """,
                (*NON_GPU_JOB_TYPES, gpu_id, int(active_gpu_job is not None), *NON_GPU_JOB_TYPES),
            ).fetchall()
            # Keep this bounded while the queue is still small; revisit before large multi-tenant queues
            # so capability-incompatible jobs cannot hide a later compatible job indefinitely.
            queued = choose_claimable_job(queued_rows, capabilities, loaded_models)
            if queued is None:
                return None
            assigned_gpu = "cpu" if queued["type"] in NON_GPU_JOB_TYPES else gpu_id

            connection.execute(
                """
                update jobs
                   set status = 'preparing',
                       assigned_gpu = ?,
                       worker_id = ?,
                       stage = 'preparing',
                       message = 'Worker claimed job.',
                       started_at = coalesce(started_at, ?),
                       updated_at = ?
                 where id = ? and status = 'queued'
                """,
                (assigned_gpu, worker_id, now, now, queued["id"]),
            )
            connection.execute(
                "update workers set status = 'busy', current_job_id = ?, last_seen_at = ? where id = ?",
                (queued["id"], now, worker_id),
            )
            return self.get_job(queued["id"], connection=connection)

    def update_job_progress(
        self,
        job_id: str,
        *,
        status: str,
        stage: str,
        progress: float,
        message: str,
        error: str | None = None,
        result: dict | None = None,
        eta_seconds: float | None = None,
    ) -> dict:
        if status not in JOB_STATUSES:
            raise ValueError(f"Unsupported job status: {status}")

        now = utc_now()
        completed_at = now if status in TERMINAL_STATUSES else None
        canceled_at = now if status == "canceled" else None
        with self._lock, self.connect() as connection:
            connection.execute(
                """
                update jobs
                   set status = ?,
                       stage = ?,
                       progress = ?,
                       message = ?,
                       error = ?,
                       result_json = coalesce(?, result_json),
                       eta_seconds = ?,
                       completed_at = coalesce(?, completed_at),
                       canceled_at = coalesce(?, canceled_at),
                       updated_at = ?
                 where id = ?
                """,
                (
                    status,
                    stage,
                    max(0, min(1, progress)),
                    message,
                    error,
                    dumps(result) if result is not None else None,
                    eta_seconds,
                    completed_at,
                    canceled_at,
                    now,
                    job_id,
                ),
            )
            job = self.get_job(job_id, connection=connection)
            if status in TERMINAL_STATUSES and job["workerId"]:
                connection.execute(
                    "update workers set status = 'idle', current_job_id = null, last_seen_at = ? where id = ?",
                    (now, job["workerId"]),
                )
            return job

    def list_workers(self) -> list[dict]:
        with self._lock, self.connect() as connection:
            rows = connection.execute("select * from workers order by gpu_id, id").fetchall()
        return [self.row_to_worker(row) for row in rows]

    def get_worker(self, worker_id: str, *, connection: sqlite3.Connection | None = None) -> dict:
        owns_connection = connection is None
        connection = connection or self.connect()
        try:
            row = connection.execute("select * from workers where id = ?", (worker_id,)).fetchone()
            if row is None:
                raise KeyError(worker_id)
            return self.row_to_worker(row)
        finally:
            if owns_connection:
                connection.close()

    def row_to_job(self, row: sqlite3.Row) -> dict:
        created_at = row["created_at"]
        started_at = row["started_at"]
        completed_at = row["completed_at"]
        elapsed_seconds = None
        if started_at:
            end = completed_at or utc_now()
            elapsed_seconds = max(
                0,
                int(
                    (
                        datetime.fromisoformat(end.replace("Z", "+00:00"))
                        - datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                    ).total_seconds()
                ),
            )

        return {
            "id": row["id"],
            "type": row["type"],
            "status": row["status"],
            "projectId": row["project_id"],
            "projectName": row["project_name"],
            "payload": loads(row["payload_json"], {}),
            "result": loads(row["result_json"], {}),
            "requestedGpu": row["requested_gpu"],
            "assignedGpu": row["assigned_gpu"],
            "workerId": row["worker_id"],
            "progress": row["progress"],
            "stage": row["stage"],
            "message": row["message"],
            "error": row["error"],
            "etaSeconds": row["eta_seconds"],
            "elapsedSeconds": elapsed_seconds,
            "attempts": row["attempts"],
            "sourceJobId": row["source_job_id"],
            "duplicateOfJobId": row["duplicate_of_job_id"],
            "cancelRequested": bool(row["cancel_requested"]),
            "createdAt": created_at,
            "updatedAt": row["updated_at"],
            "startedAt": started_at,
            "completedAt": completed_at,
            "canceledAt": row["canceled_at"],
            "lastHeartbeatAt": row["last_heartbeat_at"],
        }

    def row_to_worker(self, row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "gpuId": row["gpu_id"],
            "gpuName": row["gpu_name"],
            "status": row["status"],
            "currentJobId": row["current_job_id"],
            "capabilities": loads(row["capabilities_json"], []),
            "loadedModels": loads(row["loaded_models_json"], []),
            "registeredAt": row["registered_at"],
            "lastSeenAt": row["last_seen_at"],
        }
