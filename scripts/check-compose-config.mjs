import { mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";

async function composeConfig(envFile) {
  const result = spawnSync(
    "docker",
    ["compose", "--env-file", envFile, "config", "--format", "json"],
    { encoding: "utf8", shell: false },
  );
  if (result.status !== 0) {
    throw new Error(result.stderr || result.stdout || "docker compose config failed");
  }
  return JSON.parse(result.stdout);
}

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label}: expected ${expected}, got ${actual}`);
  }
}

function assertRuntimeDefaults(config, label) {
  const api = config.services?.api;
  const worker = config.services?.worker;
  assertEqual(api?.build?.dockerfile, "docker/rust-api.Dockerfile", `${label} api dockerfile`);
  assertEqual(api?.environment?.SCENEWORKS_API_RUNTIME, "rust", `${label} api runtime`);
  assertEqual(worker?.environment?.SCENEWORKS_UTILITY_JOBS, "1", `${label} python utility jobs`);
  assertEqual(worker?.environment?.SCENEWORKS_WORKER_ID, "python-inference-worker-0", `${label} python worker id`);
}

const tempRoot = await mkdtemp(path.join(os.tmpdir(), "sceneworks-compose-config-"));
const emptyEnv = path.join(tempRoot, "empty.env");

try {
  await writeFile(emptyEnv, "", "utf8");
  assertRuntimeDefaults(await composeConfig(emptyEnv), "compose defaults");
  assertRuntimeDefaults(await composeConfig(".env.example"), ".env.example");
  console.log("SceneWorks compose config check passed.");
} finally {
  await rm(tempRoot, { recursive: true, force: true });
}
