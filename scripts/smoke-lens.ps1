<#
.SYNOPSIS
  sc-5126 Lens / Lens-Turbo candle (Windows/CUDA) e2e smoke -- the reproducible recipe in a script.

.DESCRIPTION
  Launches the BUILT Rust API + candle worker against a scratch data-dir, submits real Lens jobs
  through the deployed worker (POST /api/v1/image/jobs), polls to completion, and asserts each renders
  via the native candle path (asset recipe.adapter = "candle_lens") -- NOT the retired Python
  lens_runner sidecar. Also verifies the worker log never touches lens_runner / the /opt/lens-venv
  sidecar (the sc-5126 teardown). Companion to docs/sc-5099/candle-lane-smoke.md.

  The candle worker claims Lens jobs via the "candle" capability marker + worker_supports_job; the
  API's mlx_route_decision log (which only knows about MLX) will say fell_back_to_torch -- ignore it,
  the candle_lens asset label is the proof the candle lane ran.

.PREREQUISITES
  - Build the binaries (VS2022 v143 BuildTools / CUDA 12.9; sm_120 native PTX shown):
      cmd /c 'call "<vcvars64.bat>" && set CUDA_COMPUTE_CAP=120 && cargo build --release ^
        -p sceneworks-rust-api -p sceneworks-rust-worker --features sceneworks-worker/backend-candle'
  - microsoft/Lens + microsoft/Lens-Turbo snapshots in the HF cache (~/.cache/huggingface/hub).

.EXAMPLE
  pwsh scripts/smoke-lens.ps1
  pwsh scripts/smoke-lens.ps1 -DataDir D:\sceneworks-candle-smoke -Port 8011
#>
[CmdletBinding()]
param(
  # Repo root (where target\release\*.exe live). Defaults to this script's parent.
  [string]$Repo = (Split-Path -Parent $PSScriptRoot),
  # Scratch data-dir for the API/worker (projects, cache, jobs.db, config/manifests).
  [string]$DataDir = (Join-Path $env:TEMP "sceneworks-lens-smoke"),
  [int]$Port = 8011,
  [int]$JobTimeoutSec = 1800,
  [string]$Gpu = "0"
)
$ErrorActionPreference = "Stop"
$base = "http://127.0.0.1:$Port"
$apiExe    = Join-Path $Repo "target\release\sceneworks-rust-api.exe"
$workerExe = Join-Path $Repo "target\release\sceneworks-rust-worker.exe"
foreach ($e in @($apiExe, $workerExe)) {
  if (-not (Test-Path -LiteralPath $e)) { throw "missing binary: $e (build with --features sceneworks-worker/backend-candle first)." }
}

# The candle worker dlopens the CUDA runtime (cudart) at launch; ensure CUDA\bin is on PATH.
$cudaBin = Join-Path $(if ($env:CUDA_PATH) { $env:CUDA_PATH } else { "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9" }) "bin"
if ((Test-Path -LiteralPath $cudaBin) -and ($env:PATH -notlike "*$cudaBin*")) { $env:PATH = "$cudaBin;$env:PATH" }

# Bootstrap the data-dir + seed the builtin manifests (the API needs the lens model rows) if absent.
$cfgManifests = Join-Path $DataDir "config\manifests"
New-Item -ItemType Directory -Force -Path $cfgManifests, (Join-Path $DataDir "cache") | Out-Null
if (-not (Test-Path (Join-Path $cfgManifests "builtin.models.jsonc"))) {
  foreach ($m in @("builtin.models.jsonc", "builtin.loras.jsonc", "builtin.recipe-presets.jsonc")) {
    $src = Join-Path $Repo "config\manifests\$m"
    if (Test-Path $src) { Copy-Item $src (Join-Path $cfgManifests $m) -Force }
  }
}

$stamp       = Get-Date -Format "yyyyMMdd-HHmmss"
$apiLog      = Join-Path $DataDir "lens-smoke-api-$stamp.log"
$apiErr      = Join-Path $DataDir "lens-smoke-api-$stamp.err.log"
$workerLog   = Join-Path $DataDir "lens-smoke-worker-$stamp.log"
$workerErr   = Join-Path $DataDir "lens-smoke-worker-$stamp.err.log"
$resultsPath = Join-Path $DataDir "lens-smoke-results-$stamp.json"

$apiProc = $null; $workerProc = $null
$results = @()

# Clean slate: kill any leftover API/worker from a prior run (else they hold port $Port / the GPU).
Get-Process -ErrorAction SilentlyContinue |
  Where-Object { $_.ProcessName -in @("sceneworks-rust-api", "sceneworks-rust-worker") } |
  ForEach-Object { try { Stop-Process -Id $_.Id -Force -ErrorAction Stop } catch {} }
Start-Sleep -Seconds 1
# Remove any stale recent-projects.json: POST /projects reads it, and a corrupt/BOM copy from a prior
# run makes the API fail project creation with "expected value at line 1 column 1". A clean BOM-less
# one is written after the project is created.
Remove-Item (Join-Path $DataDir "recent-projects.json") -Force -ErrorAction SilentlyContinue

function Submit-Job($body) {
  Invoke-RestMethod -Method Post -Uri "$base/api/v1/image/jobs" -ContentType "application/json" -Body ($body | ConvertTo-Json -Depth 8)
}
function Get-Job($id) { Invoke-RestMethod -Method Get -Uri "$base/api/v1/jobs/$id" -TimeoutSec 10 }
function Wait-Job($id, $timeoutSec) {
  $deadline = (Get-Date).AddSeconds($timeoutSec)
  while ((Get-Date) -lt $deadline) {
    $j = Get-Job $id
    if ($j.status -in @("completed","failed","canceled","interrupted")) { return $j }
    Start-Sleep -Seconds 3
  }
  Get-Job $id
}

try {
  # ---- launch API (router) ----
  $env:SCENEWORKS_DATA_DIR     = $DataDir
  $env:SCENEWORKS_CONFIG_DIR   = (Join-Path $DataDir "config")
  $env:SCENEWORKS_API_HOST     = "127.0.0.1"
  $env:SCENEWORKS_API_PORT     = "$Port"
  $env:SCENEWORKS_JOBS_DB_PATH = (Join-Path $DataDir "cache\jobs.db")
  # Let HF_HOME default to ~/.cache/huggingface (holds the Lens snapshots) -- clear any override.
  Remove-Item Env:\HF_HOME, Env:\HF_HUB_CACHE, Env:\HUGGINGFACE_HUB_CACHE -ErrorAction SilentlyContinue
  Write-Host "launching API ($apiExe) on $base ..."
  $apiProc = Start-Process -FilePath $apiExe -PassThru -RedirectStandardOutput $apiLog -RedirectStandardError $apiErr -WindowStyle Hidden
  $ok = $false
  for ($i = 0; $i -lt 60; $i++) {
    try { if ((Invoke-WebRequest -Uri "$base/api/v1/health" -TimeoutSec 2 -UseBasicParsing).StatusCode -eq 200) { $ok = $true; break } } catch {}
    if ($apiProc.HasExited) { throw "API exited early (code $($apiProc.ExitCode)). See $apiErr" }
    Start-Sleep -Milliseconds 500
  }
  if (-not $ok) { throw "API did not become healthy on $base" }
  Write-Host "API healthy."

  # ---- launch the candle worker ----
  $env:SCENEWORKS_API_URL                = $base
  $env:SCENEWORKS_WORKER_ID              = "lens-smoke"
  $env:SCENEWORKS_GPU_ID                 = $Gpu
  $env:SCENEWORKS_BACKEND_CANDLE_ENABLED = "1"
  $env:SCENEWORKS_POLL_SECONDS           = "1"
  $env:SCENEWORKS_HEARTBEAT_SECONDS      = "5"
  Write-Host "launching candle worker ($workerExe) on GPU $Gpu ..."
  $workerProc = Start-Process -FilePath $workerExe -PassThru -RedirectStandardOutput $workerLog -RedirectStandardError $workerErr -WindowStyle Hidden
  $reg = $false
  for ($i = 0; $i -lt 90; $i++) {
    try {
      $ws = Invoke-RestMethod -Uri "$base/api/v1/workers" -TimeoutSec 3
      if ($ws | Where-Object { $_.capabilities -contains "candle" }) { $reg = $true; break }
    } catch {}
    if ($workerProc.HasExited) { throw "worker exited early (code $($workerProc.ExitCode)). See $workerErr" }
    Start-Sleep -Seconds 1
  }
  if (-not $reg) { throw "candle worker never advertised the 'candle' capability. See $workerErr" }
  Write-Host "candle worker registered."

  # ---- project ----
  $proj = Invoke-RestMethod -Method Post -Uri "$base/api/v1/projects" -ContentType "application/json" -Body (@{ name = "Lens Smoke $stamp" } | ConvertTo-Json)
  $projId = $proj.id
  # The Rust worker resolves the project via recent-projects.json. Write it as a BOM-less UTF-8
  # ARRAY: PS 5.1 `Set-Content -Encoding utf8` prepends a BOM (serde -> "expected value at line 1
  # column 1") and ConvertTo-Json unwraps a single-element array into an object.
  $rpJson = "[" + ((@{ id = $projId; path = $proj.path }) | ConvertTo-Json -Compress) + "]"
  [System.IO.File]::WriteAllText((Join-Path $DataDir "recent-projects.json"), $rpJson)
  Write-Host "project $projId at $($proj.path)"

  # ---- cases (candle Lens path; a live LoRA run needs a trained Lens adapter -- sc-5147) ----
  $cases = @(
    @{ name = "lens_turbo 4-step (Q8 default)"; body = @{ projectId = $projId; prompt = "a red fox in a sunlit forest, photo"; model = "lens_turbo"; count = 1; width = 1024; height = 1024; requestedGpu = "auto" } },
    @{ name = "lens 20-step / CFG 5.0";          body = @{ projectId = $projId; prompt = "a red fox in a sunlit forest, photo"; model = "lens";       count = 1; width = 1024; height = 1024; requestedGpu = "auto" } },
    @{ name = "lens_turbo Q8 (explicit)";        body = @{ projectId = $projId; prompt = "a calico cat on a windowsill, photo"; model = "lens_turbo"; count = 1; width = 1024; height = 1024; requestedGpu = "auto"; advanced = @{ mlxQuantize = 8 } } },
    @{ name = "lens bucket 1280x720";            body = @{ projectId = $projId; prompt = "a wide mountain landscape at dawn";   model = "lens";       count = 1; width = 1280; height = 720;  requestedGpu = "auto" } }
  )

  foreach ($c in $cases) {
    Write-Host "`n=== $($c.name) ==="
    $job = Submit-Job $c.body
    Write-Host "submitted $($job.id)"
    $j = Wait-Job $job.id $JobTimeoutSec
    $adapter = $null; $assetPath = $null
    if ($j.result -and $j.result.assets -and @($j.result.assets).Count -gt 0) {
      $a = @($j.result.assets)[0]
      $adapter = $a.recipe.adapter
      $assetPath = $a.file.path
    }
    $pass = ($j.status -eq "completed") -and ($adapter -eq "candle_lens")
    $results += [pscustomobject]@{
      case = $c.name; jobId = $job.id; status = $j.status; adapter = $adapter
      asset = $assetPath; pass = $pass; message = $j.message
    }
    Write-Host ("status={0} adapter={1} asset={2} pass={3}" -f $j.status, $adapter, $assetPath, $pass)
    if ($j.status -ne "completed") { Write-Host "  message: $($j.message)"; Write-Host "  worker err tail:"; Get-Content $workerErr -Tail 25 -EA SilentlyContinue | ForEach-Object { Write-Host "    $_" } }
  }

  # ---- teardown verification: the worker never used the retired Python sidecar ----
  $workerText = (Get-Content $workerLog -Raw -EA SilentlyContinue) + (Get-Content $workerErr -Raw -EA SilentlyContinue)
  $sawSidecar = [bool]($workerText -match "lens_runner|lens_sidecar|/opt/lens-venv")
  $sawCandle  = [bool]($workerText -match "candle")

  $results | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 $resultsPath
  Write-Host "`n===== SUMMARY ====="
  $results | Format-Table case, status, adapter, pass -AutoSize | Out-String | Write-Host
  Write-Host ("worker log mentions candle: {0} ; mentions retired lens_runner/sidecar: {1}" -f $sawCandle, $sawSidecar)
  Write-Host "results JSON: $resultsPath"
  $allPass = ((@($results | Where-Object { -not $_.pass }).Count) -eq 0) -and (-not $sawSidecar)
  Write-Host ("OVERALL: {0}" -f $(if ($allPass) { "PASS" } else { "FAIL" }))
  if (-not $allPass) { exit 1 }
}
finally {
  # EAP=Continue so a taskkill hiccup never masks the real try-block error; run taskkill INSIDE cmd
  # (>nul 2>&1) so PS 5.1 doesn't wrap its stderr into a terminating NativeCommandError.
  $ErrorActionPreference = "Continue"
  foreach ($p in @($workerProc, $apiProc)) {
    if ($p -and -not $p.HasExited) { try { & cmd /c "taskkill /F /T /PID $($p.Id) >nul 2>&1" } catch {} }
  }
}
