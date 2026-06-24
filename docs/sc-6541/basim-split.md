# sc-6541 — Basim train / held-out reference split (frozen)

The reference-leakage guard from
[`closed-loop-protocol.md`](./closed-loop-protocol.md) §5. Computed **once**, deterministically,
**before** any variant dataset is constructed. The reference pool **never** enters any training
set — it is the sole ArcFace target for identity-fidelity, so overlap would reward memorization.

**Source:** `~/Datasets/Basim` — 64 photos of one person, two contiguous capture ranges:
- `IMG_0400.JPG`–`IMG_0431.JPG` — 32 high-res camera photos (~1000–2500 px).
- `IMG_1730.PNG`–`IMG_1761.PNG` — 32 lower-res images (~400–780 px).

**Split rule (reproducible, no randomness):** sort all 64 filenames ascending; an image is a
**reference** image iff its 0-based index `i` satisfies `i % 5 == 2`, else it is **train**.
Because the sorted list is all JPGs then all PNGs, this naturally stratifies across both capture
ranges, so neither pool is all-hi-res or all-lo-res.

Reproduce:
```sh
cd ~/Datasets/Basim && ls -1 | sort | awk '{ if ((NR-1) % 5 == 2) print "REF  " $0; else print "TRN  " $0 }'
```

## Held-out reference pool — 13 images (6 JPG + 7 PNG)
```
IMG_0402.JPG  IMG_0407.JPG  IMG_0412.JPG  IMG_0417.JPG  IMG_0422.JPG  IMG_0427.JPG
IMG_1730.PNG  IMG_1735.PNG  IMG_1740.PNG  IMG_1745.PNG  IMG_1750.PNG  IMG_1755.PNG  IMG_1760.PNG
```

## Training pool — 51 images (26 JPG + 25 PNG)
The remaining 51 files (every `i % 5 != 2`). All three pilot variants (clean / blurry /
low-diversity) are derived **from this pool only** — see protocol §2.

> Note on reference selection: a strided sample gives temporal/quality spread without
> cherry-picking. A hand-curated frontal/well-lit reference set would be marginally stronger for
> identity matching, but stride avoids selection bias and keeps the split reproducible from this
> doc alone. If face-detect rate on the reference pool turns out low (lo-res PNGs), fall back to
> the JPG-only reference subset (the 6 `IMG_04xx.JPG`) and record that change here.
>
> _Resolved (results §0): face-detect rate on the 13-image reference pool is 1.0 — the fallback
> is **not** needed._

## Calibration / clean-variant training subset — 21 images (11 JPG + 10 PNG)

A strided ~20-image subset of the 51-image train pool (`i % 5 < 2` over the sorted train list),
stratified across both capture ranges. This is the **clean** variant's training set and the
Stage-0 calibration subset (protocol §2.1); the blurry and low-diversity variants are derived
from the *same* subset so item count (21) is held fixed.
```
IMG_0400.JPG IMG_0401.JPG IMG_0406.JPG IMG_0408.JPG IMG_0413.JPG IMG_0414.JPG IMG_0419.JPG
IMG_0420.JPG IMG_0425.JPG IMG_0426.JPG IMG_0431.JPG IMG_1731.PNG IMG_1736.PNG IMG_1737.PNG
IMG_1742.PNG IMG_1743.PNG IMG_1748.PNG IMG_1749.PNG IMG_1754.PNG IMG_1756.PNG IMG_1761.PNG
```
Staged at `~/Datasets/lora-eval/basim-train-cal/`. Trigger token: `sks man` (training caption
"a photo of sks man"; the generation prompt grid uses the same trigger).
