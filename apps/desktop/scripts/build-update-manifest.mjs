#!/usr/bin/env node
// Builds (or merges into) the Tauri updater manifest `latest.json` (sc-1355).
//
// The macOS and Windows release jobs build on separate runners, so neither can
// produce a complete manifest alone. The macOS job runs this in "init" mode to
// create latest.json with its `darwin-aarch64` entry; the Windows job (which
// `needs: macos`) downloads that file and runs this again with `--in` to merge in
// its `windows-x86_64` entry. The result is uploaded to the GitHub Release and
// served at plugins.updater.endpoints (…/releases/latest/download/latest.json).
//
// The per-platform `signature` is the CONTENTS of the `.sig` file emitted next to
// the updater bundle (createUpdaterArtifacts) — the app verifies it against
// plugins.updater.pubkey. The `url` is the release asset's canonical download URL
// (…/releases/download/<tag>/<asset>); we construct it deterministically rather
// than reading it back, because a draft release reports an `untagged-*` URL that
// only becomes the tag URL once published.
//
// Usage:
//   node build-update-manifest.mjs \
//     --target darwin-aarch64 --version 0.4.1 \
//     --url https://github.com/OWNER/REPO/releases/download/v0.4.1/SceneWorks.app.tar.gz \
//     --sig path/to/SceneWorks.app.tar.gz.sig \
//     [--notes "..."] [--in latest.json] --out latest.json

import { readFileSync, writeFileSync, existsSync } from "node:fs";

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i += 2) {
    const key = argv[i];
    if (!key.startsWith("--")) {
      throw new Error(`expected a --flag, got "${key}"`);
    }
    out[key.slice(2)] = argv[i + 1];
  }
  return out;
}

const args = parseArgs(process.argv.slice(2));

for (const required of ["target", "version", "url", "sig", "out"]) {
  if (!args[required]) {
    console.error(`build-update-manifest: missing --${required}`);
    process.exit(1);
  }
}

const signature = readFileSync(args.sig, "utf8").trim();
if (!signature) {
  console.error(`build-update-manifest: signature file ${args.sig} is empty`);
  process.exit(1);
}

let manifest;
if (args.in && existsSync(args.in)) {
  manifest = JSON.parse(readFileSync(args.in, "utf8"));
  manifest.platforms ??= {};
} else {
  manifest = {
    version: args.version,
    notes: args.notes ?? "See the release notes for what changed.",
    pub_date: new Date().toISOString(),
    platforms: {},
  };
}

manifest.platforms[args.target] = { signature, url: args.url };

writeFileSync(args.out, `${JSON.stringify(manifest, null, 2)}\n`);
console.log(
  `build-update-manifest: wrote ${args.out} with platform "${args.target}" ` +
    `(version ${manifest.version}, ${Object.keys(manifest.platforms).length} platform(s))`,
);
