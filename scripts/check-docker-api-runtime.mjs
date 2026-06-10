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
const tempRoot = await mkdtemp(path.join(os.tmpdir(), "sceneworks-rust-api-"));
const tempData = path.join(tempRoot, "data");
await mkdir(path.join(tempData, "cache"), { recursive: true });

const env = {
  ...process.env,
  SCENEWORKS_API_PORT: port,
  SCENEWORKS_DATA_BIND: tempData,
  SCENEWORKS_CONFIG_BIND: path.resolve("config"),
  SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE:
    process.env.SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE || "1",
};

function runDocker(args) {
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

try {
  await runDocker([...compose, "up", "--build", "-d", "api"]).catch((error) => {
    // The health check below is the real gate (it polls for 120s and fails if the
    // container is not serving), so a spurious non-zero `up` is tolerated rather than
    // aborting before we ever probe the container.
    console.error(`compose up reported: ${error.message} (continuing to health check)`);
  });
  // Surface the container's boot logs AND keep an active handle across the detached
  // up -> health-poll transition. Without this, `docker compose up -d` (stdio inherited,
  // the container owned by the daemon) can momentarily drain Node's event loop and exit
  // 13 ("unsettled top-level await") before waitForHealth's first request registers a
  // socket — an intermittent CI flake unrelated to the API (the container is healthy).
  await runDocker([...compose, "logs", "--no-color", "api"]).catch(() => {});
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
  ]).catch((error) => {
    console.error(error.message);
  });
  await runDocker([...compose, "down", "--remove-orphans"]).catch((error) => {
    console.error(error.message);
  });
  await rm(tempRoot, { recursive: true, force: true });
}
