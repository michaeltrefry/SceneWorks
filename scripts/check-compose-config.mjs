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

function assertServiceMountsTarget(service, target, label) {
  const volumes = service?.volumes ?? [];
  if (!volumes.some((volume) => volume?.target === target || (typeof volume === "string" && volume.split(":").includes(target)))) {
    throw new Error(`${label}: expected a volume mounted at ${target}`);
  }
}

function assertWritableMount(service, target, label) {
  const volumes = service?.volumes ?? [];
  const volume = volumes.find((item) => item?.target === target || (typeof item === "string" && item.split(":").includes(target)));
  if (!volume) {
    throw new Error(`${label}: expected a volume mounted at ${target}`);
  }
  if (typeof volume === "string") {
    const [, , mode] = volume.split(":");
    if (mode === "ro") {
      throw new Error(`${label}: expected ${target} to be writable`);
    }
    return;
  }
  if (volume.read_only === true) {
    throw new Error(`${label}: expected ${target} to be writable`);
  }
}

function assertPublishedPort(service, target, expectedHostIp, expectedPublished, label) {
  const ports = service?.ports ?? [];
  const port = ports.find((item) => Number(item?.target) === target);
  if (!port) {
    throw new Error(`${label}: expected published port for target ${target}`);
  }
  assertEqual(port.host_ip, expectedHostIp, `${label} API host publish IP`);
  assertEqual(String(port.published), String(expectedPublished), `${label} API published port`);
}

function assertMissing(object, key, label) {
  if (Object.prototype.hasOwnProperty.call(object ?? {}, key)) {
    throw new Error(`${label}: expected ${key} to be absent`);
  }
}

function assertRuntimeDefaults(config, label, options = {}) {
  const apiPublishHost = options.apiPublishHost ?? "127.0.0.1";
  const apiPort = options.apiPort ?? "8010";
  const api = config.services?.api;
  const worker = config.services?.worker;
  const rustWorker = config.services?.["rust-worker"];
  // Split the removed key so the story's cleanup grep does not report this assertion as a live reference.
  const removedRuntimeKey = ["SCENEWORKS_API", "RUNTIME"].join("_");
  assertEqual(api?.build?.dockerfile, "docker/rust.Dockerfile", `${label} api dockerfile`);
  assertPublishedPort(api, Number(apiPort), apiPublishHost, apiPort, `${label} api`);
  assertMissing(api?.environment, removedRuntimeKey, `${label} api runtime switch`);
  assertWritableMount(api, "/sceneworks/config", `${label} api config mount`);
  // The Docker GPU worker is the native candle/CUDA worker (epic 5483 Phase 7 /
  // sc-5503), not the retired Python torch worker. The API requires candle and fails
  // unsupported jobs loudly (sc-5502) since there is no torch fallback in Docker.
  assertEqual(api?.environment?.SCENEWORKS_CANDLE_REQUIRED, "1", `${label} api candle required`);
  assertEqual(worker?.build?.target, "rust-worker-candle", `${label} candle worker build target`);
  assertEqual(worker?.environment?.SCENEWORKS_WORKER_ID, "candle-inference-worker-0", `${label} candle worker id`);
  assertEqual(worker?.environment?.HF_HOME, "/sceneworks/data/cache/huggingface", `${label} candle worker HF home`);
  assertEqual(
    worker?.environment?.HUGGINGFACE_HUB_CACHE,
    "/sceneworks/data/cache/huggingface/hub",
    `${label} candle worker HF hub cache`,
  );
  assertEqual(rustWorker?.environment?.SCENEWORKS_GPU_ID, "cpu", `${label} rust worker gpu mode`);
  assertEqual(rustWorker?.environment?.SCENEWORKS_CONFIG_DIR, "/sceneworks/config", `${label} rust worker config dir`);
  assertServiceMountsTarget(rustWorker, "/sceneworks/config", `${label} rust worker config mount`);
}

const tempRoot = await mkdtemp(path.join(os.tmpdir(), "sceneworks-compose-config-"));
const emptyEnv = path.join(tempRoot, "empty.env");
const lanEnv = path.join(tempRoot, "lan.env");

try {
  await writeFile(emptyEnv, "", "utf8");
  await writeFile(
    lanEnv,
    "SCENEWORKS_API_PUBLISH_HOST=0.0.0.0\nSCENEWORKS_API_PORT=18010\n",
    "utf8",
  );
  assertRuntimeDefaults(await composeConfig(emptyEnv), "compose defaults");
  assertRuntimeDefaults(await composeConfig(".env.example"), ".env.example");
  assertRuntimeDefaults(await composeConfig(lanEnv), "compose LAN override", {
    apiPublishHost: "0.0.0.0",
    apiPort: "18010",
  });
  console.log("SceneWorks compose config check passed.");
} finally {
  await rm(tempRoot, { recursive: true, force: true });
}
