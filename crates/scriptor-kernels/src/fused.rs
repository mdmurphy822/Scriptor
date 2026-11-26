//! Fused decoder kernels: norm + residual + norm joins and M-RoPE + paged KV append
//! (bf16 as raw `u16` bits, fp32 math).

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use scriptor_cuda::Result;

use crate::launch::row_launch;
use crate::norm::{load_module, to_config};
use crate::rope_kv::RopeKvPlan;

/// Fused norm-residual-norm and RoPE-KV-append launchers.
pub struct FusedKernels {
    norm_add_norm: CudaFunction,
    rope_kv: CudaFunction,
    rope_kv_split: CudaFunction,
    split_qkv: CudaFunction,
    rope_kv_shared: CudaFunction,
    rope_kv_decode: CudaFunction,
    rope_kv_shared_decode: CudaFunction,
}

impl FusedKernels {
    /// Loads the kernels into `ctx`.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self> {
        let norm = load_module(ctx, "fused_norm_residual")?;
        let rope = load_module(ctx, "fused_rope_kv")?;
        let split = load_module(ctx, "qkv_split")?;
        let variants = load_module(ctx, "rope_kv_variants")?;
        Ok(Self {
            norm_add_norm: norm.kernel("scriptor_norm_add_norm_bf16")?,
            rope_kv: rope.kernel("scriptor_fused_rope_kv_bf16")?,
            rope_kv_split: split.kernel("scriptor_fused_rope_kv_split_bf16")?,
            split_qkv: split.kernel("scriptor_split_qkv_bf16")?,
            rope_kv_shared: variants.kernel("scriptor_rope_kv_split_shared_bf16")?,
            rope_kv_decode: variants.kernel("scriptor_rope_kv_split_decode_bf16")?,
            rope_kv_shared_decode: variants.kernel("scriptor_rope_kv_split_shared_decode_bf16")?,
        })
    }

    /// Per row: `y = rms_norm(x) * w_post`, `sum_out = bf16(residual + y)` and
    /// `out = rms_norm(sum_out) * w_next`. Equal bit for bit to `rms_norm` followed by
    /// `add_rms_norm`. `sum_out` may alias `residual`.
    #[allow(clippy::too_many_arguments)]
    pub fn norm_add_norm(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<u16>,
        residual: &CudaSlice<u16>,
        w_post: &CudaSlice<u16>,
        w_next: &CudaSlice<u16>,
        sum_out: &mut CudaSlice<u16>,
        out: &mut CudaSlice<u16>,
        rows: usize,
        n: usize,
        eps: f32,
    ) -> Result<()> {
        let cfg = to_config(row_launch(rows, n, 4, 1024));
        let n = n as i32;
        // SAFETY: argument order and types match the kernel signature; buffers cover
        // `rows * n` elements (`w_post`, `w_next`: `n`) by the caller's contract.
        unsafe {
            stream
                .launch_builder(&self.norm_add_norm)
                .arg(x)
                .arg(residual)
                .arg(w_post)
                .arg(w_next)
                .arg(sum_out)
                .arg(out)
                .arg(&n)
                .arg(&eps)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Applies GPT-J M-RoPE in place to `q` / `k` and appends the rotated keys and the values
    /// `v` (`[tokens, kv_heads * head_dim]`) to `k_cache` / `v_cache` at the flat slots in
    /// `slots` (negative = skip the append). Equal bit for bit to `RopeKernels::mrope`
    /// followed by `KvAppendKernels::append`.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_kv_append(
        &self,
        stream: &Arc<CudaStream>,
        q: &mut CudaSlice<u16>,
        k: &mut CudaSlice<u16>,
        v: &CudaSlice<u16>,
        k_cache: &mut CudaSlice<u16>,
        v_cache: &mut CudaSlice<u16>,
        positions: &CudaSlice<i32>,
        slots: &CudaSlice<i32>,
        tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        theta: f32,
        sections: [usize; 3],
    ) -> Result<()> {
        assert_eq!(sections.iter().sum::<usize>() * 2, head_dim);
        let cfg = LaunchConfig {
            grid_dim: (tokens as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (qh, kh, hd, s0, s1) = (
            q_heads as i32,
            kv_heads as i32,
            head_dim as i32,
            sections[0] as i32,
            sections[1] as i32,
        );
        // SAFETY: argument order and types match the kernel signature; `q`/`k`/`v` cover
        // `tokens` rows, `positions` `3 * tokens` entries, `slots` `tokens` entries and the
        // caches every non-negative slot.
        unsafe {
            stream
                .launch_builder(&self.rope_kv)
                .arg(q)
                .arg(k)
                .arg(v)
                .arg(k_cache)
                .arg(v_cache)
                .arg(positions)
                .arg(slots)
                .arg(&qh)
                .arg(&kh)
                .arg(&hd)
                .arg(&theta)
                .arg(&s0)
                .arg(&s1)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Splits the fused projection output `qkv` (`[tokens, q_dim + 2 * kv_dim]`) into `q`
    /// (`[tokens, q_dim]`), `k` and `v` (`[tokens, kv_dim]`).
    #[allow(clippy::too_many_arguments)]
    pub fn split_qkv(
        &self,
        stream: &Arc<CudaStream>,
        qkv: &CudaSlice<u16>,
        q: &mut CudaSlice<u16>,
        k: &mut CudaSlice<u16>,
        v: &mut CudaSlice<u16>,
        tokens: usize,
        q_dim: usize,
        kv_dim: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (tokens as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (qd, kd) = (q_dim as i32, kv_dim as i32);
        // SAFETY: argument order and types match the kernel signature; the buffers cover
        // `tokens` rows of their widths by the caller's contract.
        unsafe {
            stream
                .launch_builder(&self.split_qkv)
                .arg(qkv)
                .arg(q)
                .arg(k)
                .arg(v)
                .arg(&qd)
                .arg(&kd)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// [`Self::split_qkv`] fused with [`Self::rope_kv_append`]: `q` and `k` receive the rotated
    /// rows, `v` the value rows, and the rotated keys and values are appended to the caches at
    /// `slots`. Equal bit for bit to the split followed by `rope_kv_append`.
    #[allow(clippy::too_many_arguments)]
    pub fn split_rope_kv_append(
        &self,
        stream: &Arc<CudaStream>,
        qkv: &CudaSlice<u16>,
        q: &mut CudaSlice<u16>,
        k: &mut CudaSlice<u16>,
        v: &mut CudaSlice<u16>,
        k_cache: &mut CudaSlice<u16>,
        v_cache: &mut CudaSlice<u16>,
        positions: &CudaSlice<i32>,
        slots: &CudaSlice<i32>,
        tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        theta: f32,
        sections: [usize; 3],
    ) -> Result<()> {
        assert_eq!(sections.iter().sum::<usize>() * 2, head_dim);
        let cfg = LaunchConfig {
            grid_dim: (tokens as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (qh, kh, hd, s0, s1) = (
            q_heads as i32,
            kv_heads as i32,
            head_dim as i32,
            sections[0] as i32,
            sections[1] as i32,
        );
        // SAFETY: argument order and types match the kernel signature; `qkv`, `q`, `k`, `v`
        // cover `tokens` rows, `positions` `3 * tokens` entries, `slots` `tokens` entries and
        // the caches every non-negative slot.
        unsafe {
            stream
                .launch_builder(&self.rope_kv_split)
                .arg(qkv)
                .arg(q)
                .arg(k)
                .arg(v)
                .arg(k_cache)
                .arg(v_cache)
                .arg(positions)
                .arg(slots)
                .arg(&qh)
                .arg(&kh)
                .arg(&hd)
                .arg(&theta)
                .arg(&s0)
                .arg(&s1)
                .launch(cfg)?;
        }
        Ok(())
    }
    /// [`Self::split_rope_kv_append`] in the variant `plan` selects. `scratch` is the standalone
    /// `(k, v)` row buffers; it must be `Some` unless `plan.skip_kv_scratch`, in which case the
    /// buffers are neither read nor written. A reference plan launches the reference kernel.
    /// Every variant produces `q`, the caches and, when written, `k` / `v` byte for byte equal
    /// to the reference kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn split_rope_kv_append_variant(
        &self,
        stream: &Arc<CudaStream>,
        plan: RopeKvPlan,
        qkv: &CudaSlice<u16>,
        q: &mut CudaSlice<u16>,
        scratch: Option<(&mut CudaSlice<u16>, &mut CudaSlice<u16>)>,
        k_cache: &mut CudaSlice<u16>,
        v_cache: &mut CudaSlice<u16>,
        positions: &CudaSlice<i32>,
        slots: &CudaSlice<i32>,
        tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        theta: f32,
        sections: [usize; 3],
    ) -> Result<()> {
        assert_eq!(sections.iter().sum::<usize>() * 2, head_dim);
        assert_eq!(
            scratch.is_none(),
            plan.skip_kv_scratch,
            "standalone k/v buffers are required exactly when the plan writes them"
        );
        let func = match (plan.shared_trig, plan.skip_kv_scratch) {
            (false, false) => {
                let (k, v) = scratch.expect("checked above");
                return self.split_rope_kv_append(
                    stream, qkv, q, k, v, k_cache, v_cache, positions, slots, tokens, q_heads,
                    kv_heads, head_dim, theta, sections,
                );
            }
            (true, false) => &self.rope_kv_shared,
            (false, true) => &self.rope_kv_decode,
            (true, true) => &self.rope_kv_shared_decode,
        };
        let cfg = LaunchConfig {
            grid_dim: (tokens as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: plan.shared_bytes(head_dim),
        };
        let (qh, kh, hd, s0, s1) = (
            q_heads as i32,
            kv_heads as i32,
            head_dim as i32,
            sections[0] as i32,
            sections[1] as i32,
        );
        let null = 0u64;
        // SAFETY: argument order and types match the kernel signature; `qkv` and `q` cover
        // `tokens` rows, `positions` `3 * tokens` entries, `slots` `tokens` entries and the
        // caches every non-negative slot. The standalone-row arguments are the buffers' device
        // pointers when written and a null pointer the kernel never dereferences when skipped.
        unsafe {
            let mut launch = stream.launch_builder(func);
            launch.arg(qkv).arg(q);
            match scratch {
                Some((k, v)) => {
                    launch.arg(k).arg(v);
                }
                None => {
                    launch.arg(&null).arg(&null);
                }
            }
            launch
                .arg(k_cache)
                .arg(v_cache)
                .arg(positions)
                .arg(slots)
                .arg(&qh)
                .arg(&kh)
                .arg(&hd)
                .arg(&theta)
                .arg(&s0)
                .arg(&s1)
                .launch(cfg)?;
        }
        Ok(())
    }
}
