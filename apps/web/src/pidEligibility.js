// PiD (pixel-diffusion) decoder eligibility/availability for the per-generation toggle
// (epic 7840, sc-7851). PiD is an OPTIONAL replacement for a model's VAE decoder: when a
// generation opts in (`advanced.usePid`), the worker swaps `vae.decode()` for a PiD
// decode + 2K/4K super-resolve pass (sc-7849). PiD is tied to a LATENT SPACE, not a model,
// so eligibility keys on the model's declared backbone checkpoint (`ui.pid.checkpointId`,
// mirroring the worker `pid_backbone_for` map). The checkpoint is a separate, non-commercial
// (NSCLv1) download provisioned by sc-7852; until it is installed the toggle stays hidden so
// the user never flips a decoder that would silently fall back to the native VAE.
//
// The two gates the Studio applies (Success criteria: "Toggle appears only for eligible +
// available models"):
//   (a) pidToggleEligible(model)          — the model's latent space has a PiD backbone.
//   (b) pidDecoderAvailable(model, models) — that backbone's PiD checkpoint is installed.
// pidToggleVisible(model, models) is both — the single predicate the toggle renders on.

// The catalog id of the PiD checkpoint download a model's latent space needs, or null when
// the model has no PiD backbone (non-eligible — the SenseNova et al. guard). Pure manifest
// read: the eligible models declare `ui.pid.checkpointId` (qwenimage today; flux/flux2/sdxl
// light up as sc-7846/47/48 land their backbones).
export function pidCheckpointId(model) {
  const id = model?.ui?.pid?.checkpointId;
  return typeof id === "string" && id.length ? id : null;
}

// (a) Does the model's latent space have a PiD backbone at all? Independent of whether the
// checkpoint is downloaded — this is the "hidden for non-eligible models" gate.
export function pidToggleEligible(model) {
  return pidCheckpointId(model) !== null;
}

// (b) Is the PiD checkpoint for this model's backbone downloaded? The checkpoint rides the
// normal catalog as its own entry (sc-7852, a utility/decoder asset); we reuse the shared
// `installState` mechanism rather than a bespoke availability field. Returns false for a
// non-eligible model, or when the checkpoint entry is absent from / not installed in the
// catalog (fail-closed: the worker would no-op to the native VAE, so a hidden toggle is the
// honest UX).
export function pidDecoderAvailable(model, models) {
  const checkpointId = pidCheckpointId(model);
  if (!checkpointId) {
    return false;
  }
  return (models ?? []).some(
    (entry) => entry?.id === checkpointId && entry?.installState === "installed",
  );
}

// Both gates: the toggle is shown only when the model is eligible AND its checkpoint is
// installed. This is the predicate the Image Studio advanced panel renders the toggle on.
export function pidToggleVisible(model, models) {
  return pidToggleEligible(model) && pidDecoderAvailable(model, models);
}
