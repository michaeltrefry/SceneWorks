use super::*;

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct WorkerSpec {
    pub(crate) worker_id: String,
    pub(crate) gpu_id: String,
}

pub(crate) struct SupervisedChild {
    pub(crate) spec: WorkerSpec,
    pub(crate) process: Child,
    pub(crate) restart_attempt: u32,
    /// When the current process was (re)spawned, so a child that ran healthily
    /// for a while resets its restart backoff instead of ratcheting it up forever
    /// (sc-4282 / F-MLXW-20).
    pub(crate) spawned_at: Instant,
}

/// A child that stayed alive at least this long before exiting is treated as
/// having had a healthy run, so its restart-backoff counter resets rather than
/// saturating upward across rare, widely-spaced crashes (sc-4282 / F-MLXW-20).
const HEALTHY_UPTIME_RESET: Duration = Duration::from_secs(300);

/// Whether a child's `uptime` (time since its last spawn) was long enough to
/// count as a healthy run and reset the restart backoff.
fn backoff_resets_after_healthy_uptime(uptime: Duration) -> bool {
    uptime >= HEALTHY_UPTIME_RESET
}

/// Whether a reaped child died abnormally and should be attributed to the user as a
/// real job FAILURE (vs. left to the generic heartbeat-sweep `interrupted`). True
/// for an uncatchable signal death (`signal` set) or a non-zero self-exit
/// (`exit_code` set and non-zero, e.g. a Rust panic → 101). A clean exit-0 — the
/// child caught a shutdown signal and exited itself — is graceful and reports
/// nothing (sc-4881 signals; sc-6320 non-signal exits).
pub(crate) fn child_died_abnormally(signal: Option<i32>, exit_code: Option<i32>) -> bool {
    signal.is_some() || exit_code.is_some_and(|code| code != 0)
}

pub(crate) async fn supervise_auto_workers(settings: Settings) -> WorkerResult<()> {
    let gpus = discover_gpus().await;
    if gpus.is_empty() {
        let specs = utility_worker_specs(&settings.worker_id, settings.utility_workers);
        return supervise_children(settings, specs).await;
    }

    let specs = auto_worker_specs(&settings.worker_id, &gpus);
    supervise_children(settings, specs).await
}

/// Spawn the given child workers and keep them running, restarting any that exit
/// (with backoff) until a shutdown signal arrives.
pub(crate) async fn supervise_children(
    settings: Settings,
    specs: Vec<WorkerSpec>,
) -> WorkerResult<()> {
    let mut children = HashMap::new();
    for spec in specs {
        let process = start_child_worker(&settings, &spec)?;
        children.insert(
            spec.worker_id.clone(),
            SupervisedChild {
                spec,
                process,
                restart_attempt: 0,
                spawned_at: Instant::now(),
            },
        );
    }

    let mut interval = tokio::time::interval(Duration::from_secs(1));
    interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            _ = shutdown_signal() => {
                stop_children(&settings, &mut children).await;
                return Ok(());
            }
            _ = interval.tick() => {
                restart_exited_children(&settings, &mut children).await?;
            }
        }
    }
}

pub(crate) async fn restart_exited_children(
    settings: &Settings,
    children: &mut HashMap<String, SupervisedChild>,
) -> WorkerResult<()> {
    restart_exited_children_with_spawner(settings, children, start_child_worker).await
}

pub(crate) async fn restart_exited_children_with_spawner<F>(
    settings: &Settings,
    children: &mut HashMap<String, SupervisedChild>,
    mut spawner: F,
) -> WorkerResult<()>
where
    F: FnMut(&Settings, &WorkerSpec) -> WorkerResult<Child>,
{
    let mut exited = Vec::new();
    for (worker_id, child) in children.iter_mut() {
        if let Some(status) = child.process.try_wait()? {
            // A child that ran healthily for a while before exiting starts its
            // backoff fresh, so rare widely-spaced crashes don't ratchet the delay
            // up to the cap forever (sc-4282 / F-MLXW-20).
            if backoff_resets_after_healthy_uptime(child.spawned_at.elapsed()) {
                child.restart_attempt = 0;
            }
            // Advance the attempt once here so the logged ETA and the actual
            // backoff below both read the same stored value.
            child.restart_attempt = child.restart_attempt.saturating_add(1);
            let delay = retry_delay(settings.poll_seconds, child.restart_attempt);
            // A child that terminated abnormally never got to report it. A
            // terminating signal (SIGKILL/OOM, SIGABRT, SIGSEGV, …) is an
            // uncatchable death; a non-zero exit code is a self-terminated process
            // (e.g. a Rust panic that unwound to exit 101). Carry both so we can
            // attribute its active job as a real FAILURE before restarting, rather
            // than letting the heartbeat sweep mark it the generic `interrupted`
            // (sc-4881 signals; sc-6320 non-signal exits). `status.code()` is `None`
            // when the child died by signal and `Some(code)` when it exited itself.
            let signal = terminating_signal(&status);
            let exit_code = status.code();
            emit_event_value(
                Level::INFO,
                json!({
                    "event": "worker_exited",
                    "workerId": worker_id,
                    "gpuId": child.spec.gpu_id,
                    "exitCode": exit_code,
                    "signal": signal,
                    "restartInSeconds": delay,
                }),
            );
            exited.push((worker_id.clone(), signal, exit_code));
        }
    }
    for (worker_id, signal, exit_code) in exited {
        // Surface an abnormal death to the user before the backoff sleep, so a job
        // that died fails promptly instead of hanging until restart. A clean exit-0
        // is a graceful stop (e.g. the child caught a shutdown signal and exited
        // itself) and is never reported (sc-6320).
        if child_died_abnormally(signal, exit_code) {
            report_worker_terminated(settings, &worker_id, signal, exit_code).await;
        }
        let Some(mut child) = children.remove(&worker_id) else {
            continue;
        };
        let delay = retry_delay(settings.poll_seconds, child.restart_attempt);
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_secs(delay)) => {}
            _ = shutdown_signal() => {
                children.insert(worker_id, child);
                stop_children(settings, children).await;
                return Ok(());
            }
        }
        let process = spawner(settings, &child.spec)?;
        child.process = process;
        child.spawned_at = Instant::now();
        children.insert(child.spec.worker_id.clone(), child);
    }
    Ok(())
}

pub(crate) async fn stop_children(
    settings: &Settings,
    children: &mut HashMap<String, SupervisedChild>,
) {
    for child in children.values_mut() {
        terminate_child(&mut child.process).await;
    }
    let deadline = tokio::time::sleep(Duration::from_secs(
        settings.shutdown_timeout_seconds.max(1),
    ));
    tokio::pin!(deadline);
    loop {
        let mut remaining = 0_usize;
        for child in children.values_mut() {
            match child.process.try_wait() {
                Ok(Some(_)) => {}
                Ok(None) => remaining += 1,
                Err(_) => {}
            }
        }
        if remaining == 0 {
            children.clear();
            return;
        }
        tokio::select! {
            _ = &mut deadline => break,
            _ = tokio::time::sleep(Duration::from_millis(100)) => {}
        }
    }
    for child in children.values_mut() {
        let _ = child.process.start_kill();
        let _ = child.process.wait().await;
    }
    children.clear();
}

pub(crate) async fn terminate_child(child: &mut Child) {
    #[cfg(unix)]
    {
        if let Some(pid) = child.id() {
            let _ = nix::sys::signal::kill(
                nix::unistd::Pid::from_raw(pid as i32),
                nix::sys::signal::Signal::SIGTERM,
            );
            return;
        }
    }
    let _ = child.start_kill();
}

pub(crate) fn start_child_worker(_settings: &Settings, spec: &WorkerSpec) -> WorkerResult<Child> {
    let executable = std::env::current_exe()?;
    emit_event_value(
        Level::INFO,
        json!({
            "event": "starting_worker",
            "workerId": spec.worker_id,
            "gpuId": spec.gpu_id,
        }),
    );
    let mut command = Command::new(executable);
    command.envs(child_environment(spec));
    command.spawn().map_err(Into::into)
}

/// The Unix signal that terminated a child, if it died by one (`WIFSIGNALED`).
/// `None` for a normal exit (`status.code()` set) or on non-Unix platforms where
/// the concept doesn't apply. This is the only place the death-by-signal can be
/// observed — it is uncatchable in the dying child itself (sc-4881).
#[cfg(unix)]
pub(crate) fn terminating_signal(status: &std::process::ExitStatus) -> Option<i32> {
    use std::os::unix::process::ExitStatusExt;
    status.signal()
}

#[cfg(not(unix))]
pub(crate) fn terminating_signal(_status: &std::process::ExitStatus) -> Option<i32> {
    None
}

/// Best-effort: tell the API that a worker child terminated abnormally — by
/// `signal` (uncatchable death) or a non-zero `exit_code` (self-exit / panic) — so
/// its active job is failed (attributed) rather than swept to the generic
/// `interrupted` (sc-4881 signals; sc-6320 non-signal exits). The dying worker
/// can't report this itself, so the supervisor does. A failure here (API down, job
/// already terminal) must never disrupt the restart loop — the heartbeat sweep
/// remains the backstop — so it is logged, not raised.
async fn report_worker_terminated(
    settings: &Settings,
    worker_id: &str,
    signal: Option<i32>,
    exit_code: Option<i32>,
) {
    let api = ApiClient::new(settings);
    let path = format!("/api/v1/workers/{worker_id}/terminated");
    let outcome: WorkerResult<Value> = api
        .post_json(&path, &json!({ "signal": signal, "exitCode": exit_code }))
        .await;
    if let Err(error) = outcome {
        emit_event_value(
            Level::ERROR,
            json!({
                "event": "worker_termination_report_failed",
                "workerId": worker_id,
                "signal": signal,
                "exitCode": exit_code,
                "error": error.to_string(),
            }),
        );
    }
}

pub(crate) fn child_environment(spec: &WorkerSpec) -> BTreeMap<String, String> {
    let mut env = BTreeMap::new();
    env.insert("SCENEWORKS_WORKER_CHILD".to_owned(), "1".to_owned());
    env.insert("SCENEWORKS_WORKER_ID".to_owned(), spec.worker_id.clone());
    env.insert("SCENEWORKS_GPU_ID".to_owned(), spec.gpu_id.clone());
    if spec.gpu_id == "cpu" {
        env.insert("CUDA_VISIBLE_DEVICES".to_owned(), String::new());
        env.insert("SCENEWORKS_UTILITY_JOBS".to_owned(), "1".to_owned());
    } else {
        env.insert("CUDA_VISIBLE_DEVICES".to_owned(), spec.gpu_id.clone());
        env.insert("SCENEWORKS_UTILITY_JOBS".to_owned(), "0".to_owned());
    }
    env
}

pub(crate) fn auto_worker_specs(base_worker_id: &str, gpus: &[DiscoveredGpu]) -> Vec<WorkerSpec> {
    let mut specs = gpus
        .iter()
        .map(|gpu| WorkerSpec {
            worker_id: gpu_worker_id(base_worker_id, &gpu.id),
            gpu_id: gpu.id.clone(),
        })
        .collect::<Vec<_>>();
    specs.push(WorkerSpec {
        worker_id: cpu_worker_id(base_worker_id),
        gpu_id: "cpu".to_owned(),
    });
    specs
}

/// Specs for the dedicated CPU/utility worker pool. The first worker keeps the
/// historical `<base>-cpu` id (so a single-worker setup is unchanged); each
/// additional worker is suffixed `-1`, `-2`, ... A count of 0 still yields one.
pub(crate) fn utility_worker_specs(base_worker_id: &str, count: usize) -> Vec<WorkerSpec> {
    (0..count.max(1))
        .map(|index| WorkerSpec {
            worker_id: utility_worker_id(base_worker_id, index),
            gpu_id: "cpu".to_owned(),
        })
        .collect()
}

pub(crate) fn utility_worker_id(base_worker_id: &str, index: usize) -> String {
    let cpu_id = cpu_worker_id(base_worker_id);
    if index == 0 {
        cpu_id
    } else {
        format!("{cpu_id}-{index}")
    }
}

#[cfg(test)]
mod backoff_tests {
    use super::{backoff_resets_after_healthy_uptime, HEALTHY_UPTIME_RESET};
    use std::time::Duration;

    /// sc-4282 / F-MLXW-20: the backoff resets only once a child has been up for
    /// at least the healthy-uptime threshold.
    #[test]
    fn backoff_resets_only_after_the_healthy_uptime_threshold() {
        assert!(!backoff_resets_after_healthy_uptime(Duration::from_secs(0)));
        assert!(!backoff_resets_after_healthy_uptime(
            HEALTHY_UPTIME_RESET - Duration::from_secs(1)
        ));
        assert!(backoff_resets_after_healthy_uptime(HEALTHY_UPTIME_RESET));
        assert!(backoff_resets_after_healthy_uptime(
            HEALTHY_UPTIME_RESET + Duration::from_secs(60)
        ));
    }
}
