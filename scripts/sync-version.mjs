#!/usr/bin/env node
// Propagate the version to every file Tauri + npm read, so one `npm version` at
// the repo root bumps the whole product atomically. Wired as the root
// package.json `version` lifecycle script: npm bumps the root package.json first,
// then runs this (before the version commit), so the synced files land in the
// SAME commit + tag — the git tag can never drift from the shipped version.
//
//   npm version 0.3.0        # bump everything -> one commit "0.3.0" -> tag v0.3.0
//   git push --follow-tags   # triggers .github/workflows/release.yml
//
// tauri.conf.json's `version` names the bundled .app / DMG (the artifact-critical
// field). The sub-package versions are bumped with `npm version` so each
// package.json AND its package-lock.json stay in lockstep — a mismatch can fail
// the `npm ci` the release workflow runs.
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

// `npm version` already wrote the new version into the root package.json; mirror it.
const version = JSON.parse(
  readFileSync(join(repoRoot, "package.json"), "utf8"),
).version;
if (!version) {
  console.error("sync-version: no version in root package.json");
  process.exit(1);
}

// Sub-packages with lockfiles: use `npm version` so package.json AND
// package-lock.json move together. --allow-same-version makes an already-aligned
// package a no-op (idempotent re-runs).
for (const dir of ["apps/desktop", "apps/web"]) {
  execFileSync(
    "npm",
    ["version", version, "--no-git-tag-version", "--allow-same-version"],
    { cwd: join(repoRoot, dir), stdio: "inherit" },
  );
}

// tauri.conf.json has no lockfile — surgical replace of the first `"version": "…"`
// to preserve exact formatting (no JSON reparse/reformat churn).
const conf = join(repoRoot, "apps", "desktop", "tauri.conf.json");
const before = readFileSync(conf, "utf8");
const after = before.replace(/"version":\s*"[^"]*"/, `"version": "${version}"`);
if (after === before) {
  console.error('sync-version: no "version" field found in tauri.conf.json');
  process.exit(1);
}
writeFileSync(conf, after);

console.log(
  `sync-version: apps/desktop + apps/web + tauri.conf.json synced to ${version}`,
);
