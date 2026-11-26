//! Device-resident vision weights converted from the reference tensors.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use scriptor_cuda::Result;

use super::bf16_bits;
use super::merger::GpuMerger;
use crate::reference::{Linear, Merger, VisionTower};

/// A dense layer with BF16 weight `[out_dim, in_dim]` and optional bias `[out_dim]`.
pub struct GpuLinear {
    pub weight: CudaSlice<u16>,
    pub bias: Option<CudaSlice<u16>>,
    pub out_dim: usize,
    pub in_dim: usize,
}

impl GpuLinear {
    /// Uploads `layer` as BF16.
    pub fn from_reference(stream: &Arc<CudaStream>, layer: &Linear) -> Result<Self> {
        Self::from_parts(
            stream,
            &layer.weight,
            layer.bias.as_deref(),
            layer.out_dim,
            layer.in_dim,
        )
    }

    /// Uploads a row-major `[out_dim, in_dim]` weight and optional bias as BF16.
    pub fn from_parts(
        stream: &Arc<CudaStream>,
        weight: &[f32],
        bias: Option<&[f32]>,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Self> {
        Ok(Self {
            weight: stream.clone_htod(&bf16_bits(weight))?,
            bias: bias.map(|b| stream.clone_htod(&bf16_bits(b))).transpose()?,
            out_dim,
            in_dim,
        })
    }

    /// Stacks layers with equal input width along the output axis, so one GEMM yields their
    /// outputs side by side per row. Biases are stacked when every layer has one.
    pub fn stacked(stream: &Arc<CudaStream>, layers: &[&Linear]) -> Result<Self> {
        let in_dim = layers[0].in_dim;
        assert!(layers.iter().all(|l| l.in_dim == in_dim));
        let weight: Vec<f32> = layers.iter().flat_map(|l| l.weight.clone()).collect();
        let bias = if layers.iter().all(|l| l.bias.is_some()) {
            Some(
                layers
                    .iter()
                    .flat_map(|l| l.bias.clone().unwrap_or_default())
                    .collect::<Vec<f32>>(),
            )
        } else {
            None
        };
        let out_dim = layers.iter().map(|l| l.out_dim).sum();
        Self::from_parts(stream, &weight, bias.as_deref(), out_dim, in_dim)
    }
}

/// One vision block's weights; the MLP gate and up projections are stacked.
pub struct GpuVisionBlock {
    pub norm1: CudaSlice<u16>,
    pub norm2: CudaSlice<u16>,
    pub qkv: GpuLinear,
    pub q_norm: CudaSlice<u16>,
    pub k_norm: CudaSlice<u16>,
    pub proj: GpuLinear,
    pub gate_up: GpuLinear,
    pub down: GpuLinear,
}

/// Every vision-tower and merger weight on the device plus the tower's shape constants.
pub struct GpuVisionParams {
    pub patch_embed: GpuLinear,
    pub blocks: Vec<GpuVisionBlock>,
    pub post_layernorm: CudaSlice<u16>,
    pub merger: GpuMerger,
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub eps: f32,
    pub rope_theta: f32,
}

impl GpuVisionParams {
    /// Uploads the reference tower and merger as BF16.
    pub fn from_reference(
        stream: &Arc<CudaStream>,
        tower: &VisionTower,
        merger: &Merger,
    ) -> Result<Self> {
        let up = |v: &[f32]| stream.clone_htod(&bf16_bits(v));
        let mut blocks = Vec::with_capacity(tower.blocks.len());
        for b in &tower.blocks {
            blocks.push(GpuVisionBlock {
                norm1: up(&b.norm1)?,
                norm2: up(&b.norm2)?,
                qkv: GpuLinear::from_reference(stream, &b.qkv)?,
                q_norm: up(&b.q_norm)?,
                k_norm: up(&b.k_norm)?,
                proj: GpuLinear::from_reference(stream, &b.proj)?,
                gate_up: GpuLinear::stacked(stream, &[&b.gate, &b.up])?,
                down: GpuLinear::from_reference(stream, &b.down)?,
            });
        }
        let intermediate = tower.blocks.first().map_or(0, |b| b.gate.out_dim);
        Ok(Self {
            patch_embed: GpuLinear::from_reference(stream, &tower.patch_embed)?,
            blocks,
            post_layernorm: up(&tower.post_layernorm)?,
            merger: GpuMerger::from_reference(stream, merger)?,
            hidden: tower.hidden,
            heads: tower.heads,
            head_dim: tower.head_dim,
            intermediate,
            eps: tower.eps,
            rope_theta: tower.rope_theta as f32,
        })
    }
}
