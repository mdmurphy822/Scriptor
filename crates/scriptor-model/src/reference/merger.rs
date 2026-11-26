//! Vision adapter: 2x2 downsample convolution and the patch merger.

use scriptor_kernels::reference::activation::silu_mul;
use scriptor_kernels::reference::conv::downsample_2x2;
use scriptor_kernels::reference::norm::layer_norm_gelu;
use scriptor_spec::VisionSpec;

use super::{Linear, ModelError, TensorSource};

/// Layer-norm epsilon of the merger's post-projection norm.
const MERGER_NORM_EPS: f32 = 1e-5;

/// Downsample convolution followed by `proj -> LayerNorm -> GELU -> SwiGLU` with the text hidden
/// size as output width.
#[derive(Debug, Clone)]
pub struct Merger {
    pub downsample_weight: Vec<f32>,
    pub downsample_bias: Vec<f32>,
    pub proj: Linear,
    pub norm_weight: Vec<f32>,
    pub norm_bias: Vec<f32>,
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl Merger {
    /// Loads the adapter tensors (`model.visual.downsample.*`, `model.visual.merger.*`).
    pub fn load(src: &TensorSource, spec: &VisionSpec) -> Result<Self, ModelError> {
        let (inp, out, mid) = (
            spec.hidden_size,
            spec.out_hidden_size,
            spec.merger_intermediate,
        );
        let m = "model.visual.merger";
        Ok(Self {
            downsample_weight: src.tensor(
                "model.visual.downsample.weight",
                &[out, inp, spec.spatial_merge_size, spec.spatial_merge_size],
            )?,
            downsample_bias: src.tensor("model.visual.downsample.bias", &[out])?,
            proj: src.linear(&format!("{m}.proj.weight"), None, out, out)?,
            norm_weight: src.tensor(&format!("{m}.post_projection_norm.weight"), &[out])?,
            norm_bias: src.tensor(&format!("{m}.post_projection_norm.bias"), &[out])?,
            gate: src.linear(&format!("{m}.gate_proj.weight"), None, mid, out)?,
            up: src.linear(&format!("{m}.up_proj.weight"), None, mid, out)?,
            down: src.linear(&format!("{m}.down_proj.weight"), None, out, mid)?,
            in_dim: inp,
            out_dim: out,
        })
    }

    /// Maps `[N, vision_hidden]` post-layer-norm states (merge-window order) to
    /// `[N / 4, out_dim]` image embeddings.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let merged = downsample_2x2(
            x,
            &self.downsample_weight,
            &self.downsample_bias,
            self.in_dim,
            self.out_dim,
        );
        let rows = merged.len() / self.out_dim;
        let projected = self.proj.apply(&merged, rows);
        let activated = layer_norm_gelu(
            &projected,
            &self.norm_weight,
            &self.norm_bias,
            self.out_dim,
            MERGER_NORM_EPS,
        );
        let act = silu_mul(
            &self.gate.apply(&activated, rows),
            &self.up.apply(&activated, rows),
        );
        self.down.apply(&act, rows)
    }
}
