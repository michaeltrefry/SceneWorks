//! NAX-presence guard for the SceneWorks workspace build seam (sc-3019, epic 3018).
//!
//! The native MLX path is only fast (and, for 16-bit, only *correct*) when MLX's
//! Apple matrix-unit ("NAX") Metal kernels are compiled at a macOS 26.2-or-newer
//! deployment target. That pin must live in the SceneWorks workspace
//! `/.cargo/config.toml`, because Cargo does not read a dependency's config — so if
//! the pin is removed or regressed, mlx-gen's own guard (which builds under mlx-gen's
//! config) would NOT catch it, while this consumer-side build would silently
//! miscompile 16-bit GEMM/SDPA to garbage (right scale, uncorrelated). This test is
//! the loud tripwire: it exercises a 16-bit fused SDPA op against an f32 ground truth
//! and fails when the kernels were compiled below the 26.2 floor.
//!
//! Mirrors `mlx-gen/mlx-gen-qwen-image/tests/sdpa_nax_repro.rs` (sc-2770/sc-2772).
//! Needs no weights, only MLX + a Metal device, so it is gated to macOS and runs as
//! the sole MLX-touching test in this crate (no cross-test MLX concurrency).

#![cfg(target_os = "macos")]

use mlx_gen::array::scalar;
use mlx_rs::{
    fast::scaled_dot_product_attention,
    ops::{matmul, multiply, softmax_axis},
    random, Array, Dtype,
};

/// Mean-absolute relative error: sum|a-b| / sum|b|, computed in f32.
fn rel(a: &Array, b: &Array) -> f64 {
    let n = b.shape().iter().product::<i32>();
    let a = a.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let num: f64 = a.iter().zip(b).map(|(x, y)| (*x - *y).abs() as f64).sum();
    let den: f64 = b.iter().map(|y| y.abs() as f64).sum();
    num / den
}

/// Manual attention on logical `[1,N,L,D]`: softmax((q @ kᵀ) * scale, -1) @ v — the
/// correct f32 reference at every dtype.
fn manual_attn(q: &Array, k: &Array, v: &Array, scale: f32) -> Array {
    let kt = k.transpose_axes(&[0, 1, 3, 2]).unwrap();
    let scores = multiply(matmul(q, &kt).unwrap(), scalar(scale)).unwrap();
    let probs = softmax_axis(&scores, -1, true).unwrap();
    matmul(&probs, v).unwrap()
}

#[test]
fn nax_16bit_sdpa_is_correct() {
    let shapes = [(24i32, 64i32, 128i32), (8, 256, 64), (16, 1024, 64)];
    let dtypes = [Dtype::Float32, Dtype::Bfloat16, Dtype::Float16];

    let mut worst_16bit = 0.0f64;
    let mut worst_f32 = 0.0f64;
    let mut worst_manual = 0.0f64;

    for (n, l, d) in shapes {
        let scale = (d as f32).powf(-0.5);
        let (kq, kk, kv) = (
            random::key(0).unwrap(),
            random::key(1).unwrap(),
            random::key(2).unwrap(),
        );
        for layout in ["contiguous", "transp-view"] {
            let (qf, kf, vf) = if layout == "contiguous" {
                let g = |key| random::normal::<f32>(&[1, n, l, d], None, None, Some(key)).unwrap();
                (g(&kq), g(&kk), g(&kv))
            } else {
                let g = |key| {
                    random::normal::<f32>(&[1, l, n, d], None, None, Some(key))
                        .unwrap()
                        .transpose_axes(&[0, 2, 1, 3])
                        .unwrap()
                };
                (g(&kq), g(&kk), g(&kv))
            };
            let gt = manual_attn(&qf, &kf, &vf, scale);
            for dt in dtypes {
                let (q, k, v) = (
                    qf.as_dtype(dt).unwrap(),
                    kf.as_dtype(dt).unwrap(),
                    vf.as_dtype(dt).unwrap(),
                );
                let fast = scaled_dot_product_attention(&q, &k, &v, scale, None, None).unwrap();
                let man = manual_attn(&q, &k, &v, scale);
                let (r_fast, r_man) = (rel(&fast, &gt), rel(&man, &gt));
                worst_manual = worst_manual.max(r_man);
                if dt == Dtype::Float32 {
                    worst_f32 = worst_f32.max(r_fast);
                } else {
                    worst_16bit = worst_16bit.max(r_fast);
                }
            }
        }
    }

    assert!(
        worst_manual < 0.05,
        "manual softmax(q·kᵀ)·v reference diverged ({worst_manual:.4} ≥ 0.05) — the f32 ground \
         truth is unreliable; re-characterize before trusting the fast-SDPA verdict."
    );
    assert!(
        worst_f32 < 0.05,
        "f32 fast SDPA diverged ({worst_f32:.4} ≥ 0.05) — the NAX f32 attention path regressed."
    );
    assert!(
        worst_16bit < 0.05,
        "NAX 16-bit fast SDPA is GARBAGE ({worst_16bit:.4} ≥ 0.05): the metal kernels were compiled \
         below the macOS 26.2 NAX floor. Verify MACOSX_DEPLOYMENT_TARGET >= 26.2 in the SceneWorks \
         workspace /.cargo/config.toml (a clean rebuild of pmetal-mlx-sys is needed after a change)."
    );
}
