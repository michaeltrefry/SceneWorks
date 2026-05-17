import { access, constants, readFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const root = process.cwd();

const requiredPaths = [
  "apps/web/package.json",
  "apps/web/src/main.jsx",
  "apps/web/src/styles.css",
  "apps/api/sceneworks_api/main.py",
  "apps/api/sceneworks_api/projects.py",
  "apps/api/sceneworks_api/security.py",
  "apps/rust-api/Cargo.toml",
  "apps/rust-api/src/main.rs",
  "apps/worker/scene_worker/runtime.py",
  "crates/sceneworks-core/Cargo.toml",
  "crates/sceneworks-core/src/lib.rs",
  "Cargo.toml",
  "rust-toolchain.toml",
  "packages/schemas/project.schema.json",
  "config/manifests/builtin.models.jsonc",
  "config/manifests/builtin.loras.jsonc",
  "data/projects/.gitkeep",
  "data/models/.gitkeep",
  "data/loras/.gitkeep",
  "data/cache/.gitkeep",
  "docker-compose.yml",
  "docker/api.Dockerfile",
  "docker/rust-api.Dockerfile",
  "docker/web.Dockerfile",
  "docker/worker.Dockerfile",
];

async function assertReadable(relativePath) {
  const absolutePath = path.join(root, relativePath);
  await access(absolutePath, constants.R_OK);
}

async function assertContains(relativePath, expected) {
  const body = await readFile(path.join(root, relativePath), "utf8");
  if (!body.includes(expected)) {
    throw new Error(`${relativePath} does not contain ${expected}`);
  }
}

for (const requiredPath of requiredPaths) {
  await assertReadable(requiredPath);
}

await assertContains("apps/web/src/App.jsx", "/api/v1/health");
await assertContains("apps/api/sceneworks_api/main.py", "/api/v1/health");
await assertContains("apps/api/sceneworks_api/jobs.py", "/jobs/events");
await assertContains("Cargo.toml", "apps/rust-api");
await assertContains("crates/sceneworks-core/src/lib.rs", "/api/v1/health");
await assertContains("docker-compose.yml", "NVIDIA_VISIBLE_DEVICES");
await assertContains("docker-compose.yml", "SCENEWORKS_API_DOCKERFILE:-docker/rust-api.Dockerfile");
await assertContains("docker-compose.yml", "SCENEWORKS_API_RUNTIME:-rust");
await assertContains("docker-compose.yml", "SCENEWORKS_PYTHON_UTILITY_JOBS:-1");
await assertContains("docker-compose.yml", "/sceneworks/data/cache/jobs.db");
await assertContains(".env.example", "SCENEWORKS_API_RUNTIME=rust");
await assertContains(".env.example", "SCENEWORKS_API_DOCKERFILE=docker/rust-api.Dockerfile");
await assertContains(".env.example", "SCENEWORKS_PYTHON_UTILITY_JOBS=1");
await assertContains("docker/rust-api.Dockerfile", "sceneworks-rust-api");
await assertContains("scripts/check-docker-api-runtime.mjs", "SCENEWORKS_API_RUNTIME");
await assertContains("README.md", "SCENEWORKS_ACCESS_TOKEN");

console.log("SceneWorks scaffold check passed.");
