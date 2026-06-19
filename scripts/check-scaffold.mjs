import { access, constants, readFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const root = process.cwd();

const requiredPaths = [
  "apps/web/package.json",
  "apps/web/src/main.jsx",
  "apps/web/src/styles.css",
  "apps/rust-api/Cargo.toml",
  "apps/rust-api/src/main.rs",
  "apps/desktop/Cargo.toml",
  "apps/desktop/src/main.rs",
  "apps/worker/scene_worker/runtime.py",
  "crates/sceneworks-core/Cargo.toml",
  "crates/sceneworks-core/src/lib.rs",
  "Cargo.toml",
  "rust-toolchain.toml",
  "packages/schemas/model-manifest.schema.json",
  "packages/schemas/lora-manifest.schema.json",
  "packages/schemas/recipe-preset.schema.json",
  "config/manifests/builtin.models.jsonc",
  "config/manifests/builtin.loras.jsonc",
  "config/manifests/builtin.recipe-presets.jsonc",
  "data/projects/.gitkeep",
  "data/models/.gitkeep",
  "data/loras/.gitkeep",
  "data/cache/.gitkeep",
  "docker-compose.yml",
  "docker/rust.Dockerfile",
  "docker/web.Dockerfile",
];

const manifestSchemaPaths = [
  "packages/schemas/model-manifest.schema.json",
  "packages/schemas/lora-manifest.schema.json",
  "packages/schemas/recipe-preset.schema.json",
];

const manifestPaths = [
  "config/manifests/builtin.models.jsonc",
  "config/manifests/builtin.loras.jsonc",
  "config/manifests/builtin.recipe-presets.jsonc",
];

const manifestSchemaPairs = [
  {
    manifestPath: "config/manifests/builtin.models.jsonc",
    schemaPath: "packages/schemas/model-manifest.schema.json",
  },
  {
    manifestPath: "config/manifests/builtin.loras.jsonc",
    schemaPath: "packages/schemas/lora-manifest.schema.json",
  },
  {
    manifestPath: "config/manifests/builtin.recipe-presets.jsonc",
    schemaPath: "packages/schemas/recipe-preset.schema.json",
  },
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

function stripJsoncComments(body) {
  let result = "";
  let inString = false;
  let escaped = false;
  for (let index = 0; index < body.length; index += 1) {
    const char = body[index];
    const next = body[index + 1];
    if (inString) {
      result += char;
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === '"') {
        inString = false;
      }
      continue;
    }
    if (char === '"') {
      inString = true;
      result += char;
      continue;
    }
    if (char === "/" && next === "/") {
      while (index < body.length && body[index] !== "\n") {
        index += 1;
      }
      result += "\n";
      continue;
    }
    if (char === "/" && next === "*") {
      index += 2;
      while (index < body.length && !(body[index] === "*" && body[index + 1] === "/")) {
        index += 1;
      }
      index += 1;
      continue;
    }
    result += char;
  }
  return result;
}

async function readJsonc(relativePath) {
  return JSON.parse(stripJsoncComments(await readFile(path.join(root, relativePath), "utf8")));
}

async function assertManifestSchemasParse() {
  for (const schemaPath of manifestSchemaPaths) {
    const schema = JSON.parse(await readFile(path.join(root, schemaPath), "utf8"));
    if (schema.$schema !== "https://json-schema.org/draft/2020-12/schema") {
      throw new Error(`${schemaPath} must declare JSON Schema draft 2020-12`);
    }
    if (!schema.$id?.startsWith("https://sceneworks.local/schemas/")) {
      throw new Error(`${schemaPath} must declare a sceneworks.local schema id`);
    }
  }
}

async function assertManifestSchemaReferences() {
  for (const manifestPath of manifestPaths) {
    const manifest = await readJsonc(manifestPath);
    if (typeof manifest.$schema !== "string") {
      throw new Error(`${manifestPath} must declare a local $schema path`);
    }
    if (!manifest.$schema.startsWith("../../packages/schemas/")) {
      throw new Error(`${manifestPath} must reference packages/schemas, got ${manifest.$schema}`);
    }
    await assertReadable(path.normalize(path.join(path.dirname(manifestPath), manifest.$schema)));
  }
}

function assertJsonType(relativePath, key, value, expectedType) {
  if (expectedType === "integer") {
    if (!Number.isInteger(value)) {
      throw new Error(`${relativePath} ${key} must be an integer`);
    }
    return;
  }
  if (expectedType === "array") {
    if (!Array.isArray(value)) {
      throw new Error(`${relativePath} ${key} must be an array`);
    }
    return;
  }
  if (typeof value !== expectedType) {
    throw new Error(`${relativePath} ${key} must be a ${expectedType}`);
  }
}

async function assertManifestRootsMatchSchemas() {
  for (const { manifestPath, schemaPath } of manifestSchemaPairs) {
    const manifest = await readJsonc(manifestPath);
    const schema = JSON.parse(await readFile(path.join(root, schemaPath), "utf8"));
    for (const key of schema.required ?? []) {
      if (!(key in manifest)) {
        throw new Error(`${manifestPath} is missing required schema field ${key}`);
      }
      const expectedType = schema.properties?.[key]?.type;
      if (typeof expectedType === "string") {
        assertJsonType(manifestPath, key, manifest[key], expectedType);
      }
    }
  }
}

async function assertBuiltinPromptGuides() {
  const manifestPath = "config/manifests/builtin.models.jsonc";
  const manifest = await readJsonc(manifestPath);
  for (const model of manifest.models ?? []) {
    const guide = model.ui?.promptGuide;
    if (!guide?.title || !guide?.path) {
      throw new Error(`${manifestPath} model ${model.id} is missing ui.promptGuide title/path`);
    }
    if (!Array.isArray(guide.sources) || guide.sources.length === 0) {
      throw new Error(`${manifestPath} model ${model.id} promptGuide needs source links`);
    }
    if (!guide.path.startsWith("/prompt-guides/") || !guide.path.endsWith(".md")) {
      throw new Error(`${manifestPath} model ${model.id} promptGuide path is invalid: ${guide.path}`);
    }
    await assertReadable(path.join("apps/web/public", guide.path.slice(1)));
  }
}

async function assertCharacterImageTuningSurface() {
  // Catch the sc-2017 picker UX mismatch at build time: a model that opts out
  // of the IP-Adapter Reference-strength slider via `ui.hideReferenceStrength`
  // must surface a `ui.variationStrength` slider in its place, otherwise the
  // "With character" picker leaves the user with no identity-tuning control.
  // The engine-wiring half of the sc-2018 guard lives in pytest because the
  // scaffold can't see worker MODEL_TARGETS without parsing Python.
  const manifestPath = "config/manifests/builtin.models.jsonc";
  const manifest = await readJsonc(manifestPath);
  const unbalanced = [];
  for (const model of manifest.models ?? []) {
    const ui = model.ui ?? {};
    if (ui.hideReferenceStrength && !ui.variationStrength) {
      unbalanced.push(model.id);
    }
  }
  if (unbalanced.length) {
    throw new Error(
      `${manifestPath} models hide the Reference-strength slider without declaring ` +
        `ui.variationStrength: ${unbalanced.join(", ")}. Add variationStrength or ` +
        `drop hideReferenceStrength.`,
    );
  }
}

for (const requiredPath of requiredPaths) {
  await assertReadable(requiredPath);
}

await assertContains("apps/web/src/App.jsx", "/api/v1/health");
await assertContains("Cargo.toml", "apps/rust-api");
await assertContains("Cargo.toml", "apps/desktop");
await assertContains("crates/sceneworks-core/src/lib.rs", "/api/v1/health");
await assertContains("docker-compose.yml", "NVIDIA_VISIBLE_DEVICES");
await assertContains("docker-compose.yml", "dockerfile: docker/rust.Dockerfile");
await assertContains("docker-compose.yml", "SCENEWORKS_RUST_WORKER_GPU_ID:-cpu");
await assertContains("docker-compose.yml", "/sceneworks/data/cache/jobs.db");
await assertContains("docker-compose.yml", "SCENEWORKS_ALLOW_OPEN_BIND");
await assertContains(".env.example", "SCENEWORKS_RUST_WORKER_GPU_ID=cpu");
await assertContains("docker/rust.Dockerfile", "sceneworks-rust-api");
await assertContains("README.md", "SCENEWORKS_ACCESS_TOKEN");
await assertManifestSchemasParse();
await assertManifestSchemaReferences();
await assertManifestRootsMatchSchemas();
await assertBuiltinPromptGuides();
await assertCharacterImageTuningSurface();

console.log("SceneWorks scaffold check passed.");
