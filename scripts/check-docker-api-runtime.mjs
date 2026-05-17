import { spawn } from "node:child_process";
import { mkdtemp, mkdir, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { setTimeout as sleep } from "node:timers/promises";

const runtime = process.argv[2] ?? "rust";
const dockerfile =
  runtime === "rust" ? "docker/rust-api.Dockerfile" : "docker/api.Dockerfile";
const port = process.env.SCENEWORKS_API_PORT || (runtime === "rust" ? "18000" : "18001");
const projectName = `sceneworks-${runtime}-api-check`;
const compose = ["compose", "-p", projectName];
const tempRoot = await mkdtemp(path.join(os.tmpdir(), `sceneworks-${runtime}-api-`));
const tempData = path.join(tempRoot, "data");
await mkdir(path.join(tempData, "cache"), { recursive: true });

const env = {
  ...process.env,
  SCENEWORKS_API_RUNTIME: runtime,
  SCENEWORKS_API_DOCKERFILE: dockerfile,
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
  await runDocker([...compose, "up", "--build", "-d", "api"]);
  await waitForHealth();
} finally {
  await runDocker([...compose, "down", "--remove-orphans"]).catch((error) => {
    console.error(error.message);
  });
  await rm(tempRoot, { recursive: true, force: true });
}
