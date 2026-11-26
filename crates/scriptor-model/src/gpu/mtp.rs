//! Multi-token-prediction layer on the device: fuses the next token's embedding with the target
//! model's hidden state and runs one decoder layer against its own key/value pool.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use scriptor_spec::TextSpec;
use scriptor_weights::SafetensorsDir;

use super::decoder::{
    GpuDecoderWeights, GpuKvPool, GpuLayerWeights, GpuTextError, TextGeometry, tensors_identical,
    upload_bf16,
};
use super::forward::{DevicePlan, HiddenSource, StepInput, StepOutput, TextExecutor, gemm_bf16};

/// Result of [`TextExecutor::mtp_forward`]: draft logits and, when requested, the MTP layer
/// output rows (before the shared head's norm).
pub type MtpOutput = StepOutput;

/// The MTP head stored as decoder layer index `num_hidden_layers`.
pub struct GpuMtpWeights {
    pub embed_tokens: Arc<CudaSlice<u16>>,
    pub enorm: CudaSlice<u16>,
    pub hnorm: CudaSlice<u16>,
    /// `[hidden, 2 * hidden]`: columns act on `[enorm(embed), hnorm(hidden)]`.
    pub eh_proj: CudaSlice<u16>,
    pub layer: GpuLayerWeights,
    pub shared_norm: CudaSlice<u16>,
    pub shared_head: Arc<CudaSlice<u16>>,
}

impl GpuMtpWeights {
    /// Uploads the head from `model.language_model.layers.{num_layers}.*`, each tensor separately.
    pub fn load(
        stream: &Arc<CudaStream>,
        dir: &SafetensorsDir,
        spec: &TextSpec,
    ) -> Result<Self, GpuTextError> {
        Self::load_inner(stream, dir, spec, None)
    }

    /// Uploads the head, reusing the decoder's embedding table and language-model head handles
    /// for the MTP embedding and shared head whose checkpoint bytes are identical to them; any
    /// tensor that differs is uploaded separately. Prints one line stating what was shared.
    pub fn load_sharing(
        stream: &Arc<CudaStream>,
        dir: &SafetensorsDir,
        spec: &TextSpec,
        decoder: &GpuDecoderWeights,
    ) -> Result<Self, GpuTextError> {
        let mtp = Self::load_inner(stream, dir, spec, Some(decoder))?;
        let word = |shared: bool| if shared { "aliased" } else { "uploaded" };
        eprintln!(
            "scriptor: mtp embedding {}, mtp shared head {}",
            word(Arc::ptr_eq(&mtp.embed_tokens, &decoder.embed_tokens)),
            word(Arc::ptr_eq(&mtp.shared_head, &decoder.lm_head)),
        );
        Ok(mtp)
    }

    fn load_inner(
        stream: &Arc<CudaStream>,
        dir: &SafetensorsDir,
        spec: &TextSpec,
        decoder: Option<&GpuDecoderWeights>,
    ) -> Result<Self, GpuTextError> {
        let g = TextGeometry::from_spec(spec)?;
        let p = format!("model.language_model.layers.{}", spec.num_layers);
        let (h, vocab) = (g.hidden, g.vocab);
        let up = |suffix: &str, shape: &[usize]| {
            upload_bf16(stream, dir, &format!("{p}.{suffix}"), shape)
        };
        let shared = |suffix: &str,
                      decoder_name: &str,
                      handle: Option<&Arc<CudaSlice<u16>>>|
         -> Result<Arc<CudaSlice<u16>>, GpuTextError> {
            let own = format!("{p}.{suffix}");
            if let Some(handle) = handle
                && tensors_identical(dir, &own, decoder_name, &[vocab, h])?
            {
                return Ok(Arc::clone(handle));
            }
            Ok(Arc::new(upload_bf16(stream, dir, &own, &[vocab, h])?))
        };
        Ok(Self {
            embed_tokens: shared(
                "embed_tokens.weight",
                "model.language_model.embed_tokens.weight",
                decoder.map(|d| &d.embed_tokens),
            )?,
            enorm: up("enorm.weight", &[h])?,
            hnorm: up("hnorm.weight", &[h])?,
            eh_proj: up("eh_proj.weight", &[h, 2 * h])?,
            layer: GpuLayerWeights::load(stream, dir, &p, &g)?,
            shared_norm: up("shared_head.norm.weight", &[h])?,
            shared_head: shared(
                "shared_head.head.weight",
                "lm_head.weight",
                decoder.map(|d| &d.lm_head),
            )?,
        })
    }
}

impl TextExecutor {
    /// Runs the MTP layer over the rows of `input`, whose tokens are the tokens that follow
    /// each row. `hidden` is the target model's post-final-norm hidden state of every batch row
    /// (`[rows, hidden]` bf16). Rows at temporal position 0 use a zero token embedding. Keys and
    /// values go to `kv`, a one-layer pool with the target pool's block geometry. Returns fp32
    /// draft logits for `input.sample_rows`.
    pub fn mtp_forward(
        &mut self,
        mtp: &GpuMtpWeights,
        kv: &mut GpuKvPool,
        input: &StepInput<'_>,
        hidden: &CudaSlice<u16>,
    ) -> Result<MtpOutput, GpuTextError> {
        let g = self.geometry;
        if kv.layer_count() != 1 {
            return Err(GpuTextError::Input(format!(
                "the MTP pool must hold one layer, not {}",
                kv.layer_count()
            )));
        }
        let plan = DevicePlan::build(&self.stream, &g, kv, input)?;
        self.gemm
            .set_shape_phase(super::forward::gemm_phase(&plan, true));
        if hidden.len() < plan.rows * g.hidden {
            return Err(GpuTextError::Input(
                "hidden state is shorter than the batch rows".into(),
            ));
        }
        self.work
            .ensure(&self.stream, &g, plan.rows, plan.sample_count)?;
        let rows = plan.rows;

        self.scatter.embedding_gather(
            &self.stream,
            &mtp.embed_tokens,
            &plan.ids,
            &mut self.work.t1,
            rows,
            g.hidden,
            g.vocab,
        )?;
        for &row in &plan.zero_rows {
            let mut view = self.work.t1.slice_mut(row * g.hidden..(row + 1) * g.hidden);
            self.stream.memset_zeros(&mut view)?;
        }
        self.norm.rms_norm(
            &self.stream,
            &self.work.t1,
            &mtp.enorm,
            &mut self.work.t2,
            rows,
            g.hidden,
            g.eps,
        )?;
        self.ops.copy_rows(
            &self.stream,
            &self.work.t2,
            0,
            g.hidden,
            &mut self.work.cat,
            0,
            2 * g.hidden,
            rows,
            g.hidden,
        )?;
        self.norm.rms_norm(
            &self.stream,
            hidden,
            &mtp.hnorm,
            &mut self.work.t1,
            rows,
            g.hidden,
            g.eps,
        )?;
        self.ops.copy_rows(
            &self.stream,
            &self.work.t1,
            0,
            g.hidden,
            &mut self.work.cat,
            g.hidden,
            2 * g.hidden,
            rows,
            g.hidden,
        )?;
        gemm_bf16(
            &mut self.gemm,
            &self.stream,
            &self.work.cat,
            &mtp.eh_proj,
            &mut self.work.h,
            rows,
            g.hidden,
            2 * g.hidden,
        )?;
        self.norm.rms_norm(
            &self.stream,
            &self.work.h,
            &mtp.layer.input_norm,
            &mut self.work.hn,
            rows,
            g.hidden,
            g.eps,
        )?;
        self.run_stack(
            std::slice::from_ref(&mtp.layer),
            kv.layers_mut(),
            &mtp.shared_norm,
            &plan,
        )?;
        self.finish(
            &plan,
            &mtp.shared_head,
            HiddenSource::Residual,
            input.keep_hidden,
        )
    }
}
