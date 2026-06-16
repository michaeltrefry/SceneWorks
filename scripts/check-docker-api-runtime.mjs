import { spawn } from "node:child_process";
import { mkdtemp, mkdir, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { setTimeout as sleep } from "node:timers/promises";

const runtime = "rust";
const port = process.env.SCENEWORKS_API_PORT || "18000";
const projectName = "sceneworks-rust-api-check";
const compose = ["compose", "-p", projectName];

function runDocker(args, env) {
  return new Promise((resolve, reject) => {
    const child = spawn("docker", args, { env, stdio: "inherit", shell: false });
    child.on("error", reject);
    child.on("exit", (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`docker ${args.join(" ")} exited with code ${code}`));
      }
    });
  });
}

async function waitForHealth() {
  const url = `http://127.0.0.1:${port}/api/v1/health`;
  const deadline = Date.now() + 120_000;
  let lastError;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) {
        const health = await response.json();
        if (health.runtime !== runtime) {
          throw new Error(`expected runtime ${runtime}, got ${health.runtime}`);
        }
        console.log(`SceneWorks ${runtime} API Docker health check passed on ${url}.`);
        return;
      }
      lastError = `HTTP ${response.status}`;
    } catch (error) {
      lastError = error.message;
    }
    await sleep(2_000);
  }
  throw new Error(`SceneWorks ${runtime} API did not become healthy: ${lastError}`);
}

async function main() {
  const tempRoot = await mkdtemp(path.join(os.tmpdir(), "sceneworks-rust-api-"));
  const tempData = path.join(tempRoot, "data");
  await mkdir(path.join(tempData, "cache"), { recursive: true });

  const env = {
    ...process.env,
    SCENEWORKS_API_PORT: port,
    SCENEWORKS_DATA_BIND: tempData,
    SCENEWORKS_CONFIG_BIND: path.resolve("config"),
    // This smoke intentionally exercises the Docker API container's wider in-container
    // bind through a host-loopback published port, without provisioning auth.
    SCENEWORKS_ALLOW_OPEN_BIND: "1",
    SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE:
      process.env.SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE || "1",
    // Run the container as this (host) user so the api isn't root and the files it
    // seeds into the bind-mounted data dir stay owned by the runner (sc-4285 /
    // F-INFRA-10), matching the compose `user:` default.
    ...(typeof process.getuid === "function"
      ? {
          SCENEWORKS_UID: String(process.getuid()),
          SCENEWORKS_GID: String(process.getgid()),
        }
      : {}),
  };

  // Keep-alive: a ref'd timer guarantees the event loop always has an active handle
  // for the whole run. `docker compose up -d` detaches (the container is owned by the
  // daemon and the CLI child exits quickly), so between the detached `up` finishing and
  // waitForHealth's first `fetch()` registering a socket, Node can momentarily have zero
  // active handles. Combined with running off the module top level (main() below, not a
  // top-level await), this closes both routes to the intermittent exit-13 CI flake
  // ("unsettled top-level await") seen while the container is in fact healthy. Cleared in
  // `finally` so the process can exit normally once the smoke completes.
  const keepAlive = setInterval(() => {}, 1000);

  try {
    await runDocker([...compose, "up", "--build", "-d", "api"], env).catch((error) => {
      // The health check below is the real gate (it polls for 120s and fails if the
      // container is not serving), so a spurious non-zero `up` is tolerated rather than
      // aborting before we ever probe the container.
      console.error(`compose up reported: ${error.message} (continuing to health check)`);
    });
    // Surface the container's boot logs so a genuine CI failure has the API banner/errors
    // in the run output. (This is no longer load-bearing for exit-13 avoidance — the
    // keepAlive handle and main() wrapper cover that — it's purely diagnostic now.)
    await runDocker([...compose, "logs", "--no-color", "api"], env).catch(() => {});
    await waitForHealth();
  } finally {
    // The api service runs as root in the container, so anything it seeds into the
    // bind-mounted data dir is root-owned — notably data/projects/global-poses.sceneworks,
    // the global pose store the API creates on boot (apps/rust-api ensure_global_poses_project).
    // The host CI user then can't remove those files (EACCES on rmdir; fs.rm's `force`
    // ignores ENOENT, not EACCES). Delete the root-owned tree from inside a one-off root
    // container first. Scope it to data/projects so we never touch the Hugging Face cache
    // bind-mounted under data/cache/huggingface.
    await runDocker([
      ...compose, "run", "--rm", "--no-deps", "--entrypoint", "rm", "api", "-rf", "/sceneworks/data/projects",
    ], env).catch((error) => {
      console.error(error.message);
    });
    await runDocker([...compose, "down", "--remove-orphans"], env).catch((error) => {
      console.error(error.message);
    });
    await rm(tempRoot, { recursive: true, force: true });
    clearInterval(keepAlive);
  }
}

main().catch((error) => {
  // A genuine failure (unhealthy container hitting the 120s waitForHealth deadline,
  // an unexpected throw, etc.) surfaces as a normal rejection here and exits non-zero —
  // not via Node's top-level-await exit-13 path.
  console.error(error);
  process.exitCode = 1;
});
