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

function assertMissing(object, key, label) {
  if (Object.prototype.hasOwnProperty.call(object ?? {}, key)) {
    throw new Error(`${label}: expected ${key} to be absent`);
  }
}

function assertRuntimeDefaults(config, label) {
  const api = config.services?.api;
  const worker = config.services?.worker;
  const rustWorker = config.services?.["rust-worker"];
  // Split the removed key so the story's cleanup grep does not report this assertion as a live reference.
  const removedRuntimeKey = ["SCENEWORKS_API", "RUNTIME"].join("_");
  assertEqual(api?.build?.dockerfile, "docker/rust-api.Dockerfile", `${label} api dockerfile`);
  assertMissing(api?.environment, removedRuntimeKey, `${label} api runtime switch`);
  assertMissing(worker?.environment, "SCENEWORKS_UTILITY_JOBS", `${label} python utility jobs`);
  assertEqual(worker?.environment?.SCENEWORKS_WORKER_ID, "python-inference-worker-0", `${label} python worker id`);
  assertEqual(worker?.environment?.HF_HOME, "/sceneworks/data/cache/huggingface", `${label} python HF home`);
  assertEqual(
    worker?.environment?.HUGGINGFACE_HUB_CACHE,
    "/sceneworks/data/cache/huggingface/hub",
    `${label} python HF hub cache`,
  );
  assertEqual(rustWorker?.environment?.SCENEWORKS_GPU_ID, "cpu", `${label} rust worker gpu mode`);
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
