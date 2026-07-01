//! Linear layer
//!
//! This layer applies a linear transformation to the incoming data, `y = x@w.t() + b`.
//! The bias is optional. The `forward` method can be used to apply the layer, it supports input
//! with a batch dimension (so of shape `(b_sz, in_c)`) or without (of shape `(in_c,)`), the
//! output has shape `(b_sz, out_c)` and `(out_c,)` respectively.

use candle::{Result, Tensor, DType};
use std::sync::Arc;

/// Dynamic execution backends supported by the foundational Linear layer
#[derive(Clone)]
pub enum LinearBackend {
    /// Standard full-precision dense projection
    Standard {
        weight: Tensor,
        bias: Option<Tensor>,
    },
    /// Saccade C-TARQ adaptive low-bit projection
    Saccade {
        op: Arc<saccade_core::SaccadeLinearOp>,
        bias: Option<Tensor>,
        bypass: bool,
    },
}

impl std::fmt::Debug for LinearBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Standard { weight, bias } => f.debug_struct("Standard")
                .field("weight", weight)
                .field("bias", bias)
                .finish(),
            Self::Saccade { op, bias, bypass } => f.debug_struct("Saccade")
                .field("in_features", &op.in_features)
                .field("out_features", &op.out_features)
                .field("bias", bias)
                .field("bypass", bypass)
                .finish(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Linear {
    backend: LinearBackend,
}

impl Linear {
    /// Construct a standard dense linear layer
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self {
            backend: LinearBackend::Standard { weight, bias },
        }
    }

    /// Construct a Saccade-optimized linear layer
    pub fn new_saccade(
        packed_base: Tensor,
        scale_base: Tensor,
        sparse_delta_q8: Option<saccade_core::config::SparseDeltaMatrix>,
        config: saccade_core::SaccadeConfig,
        in_features: usize,
        out_features: usize,
        bias: Option<Tensor>,
    ) -> Result<Self> {
        let op = saccade_core::SaccadeLinearOp::new(
            packed_base,
            scale_base,
            sparse_delta_q8,
            config,
            out_features,
            in_features,
        )?;

        Ok(Self {
            backend: LinearBackend::Saccade {
                op: Arc::new(op),
                bias,
                bypass: false,
            },
        })
    }

    pub fn weight(&self) -> &Tensor {
        match &self.backend {
            LinearBackend::Standard { weight, .. } => weight,
            LinearBackend::Saccade { op, .. } => &op.dequantized_weight,
        }
    }

    pub fn bias(&self) -> Option<&Tensor> {
        match &self.backend {
            LinearBackend::Standard { bias, .. } => bias.as_ref(),
            LinearBackend::Saccade { bias, .. } => bias.as_ref(),
        }
    }

    /// Dynamically toggle the execution bypass switch for this layer
    pub fn set_bypass(&mut self, enabled: bool) {
        if let LinearBackend::Saccade { ref mut bypass, .. } = self.backend {
            *bypass = enabled;
        }
    }

    pub fn backend(&self) -> &LinearBackend {
        &self.backend
    }
}

impl super::Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.backend {
            LinearBackend::Standard { weight, bias } => {
                // When possible, we avoid using a broadcasted matmul as it is much slower
                // than the standard matmul for the cuda and cpu backends.
                let y = match *x.dims() {
                    [b1, b2, m, k] => {
                        if x.is_contiguous() {
                            let w = weight.t()?;
                            x.reshape((b1 * b2 * m, k))?
                                .matmul(&w)?
                                .reshape((b1, b2, m, ()))?
                        } else {
                            let w = weight.broadcast_left((b1, b2))?.t()?;
                            x.matmul(&w)?
                        }
                    }
                    [bsize, m, k] => {
                        if x.is_contiguous() {
                            let w = weight.t()?;
                            x.reshape((bsize * m, k))?
                                .matmul(&w)?
                                .reshape((bsize, m, ()))?
                        } else {
                            let w = weight.broadcast_left(bsize)?.t()?;
                            x.matmul(&w)?
                        }
                    }
                    _ => {
                        let w = weight.t()?;
                        x.matmul(&w)?
                    }
                };
                match bias {
                    None => Ok(y),
                    Some(b) => y.broadcast_add(b),
                }
            }
            LinearBackend::Saccade { op, bias, bypass } => {
                if *bypass || saccade_core::is_c_tarq_bypassed() {
                    let start_time = std::time::Instant::now();
                    // Bypass Path: execute standard matmul using the pre-reconstructed float weight tensor
                    let dequantized_w = op.dequantized_weight.to_dtype(x.dtype())?;
                    let y = match *x.dims() {
                        [b1, b2, m, k] => {
                            if x.is_contiguous() {
                                let w = dequantized_w.t()?;
                                x.reshape((b1 * b2 * m, k))?
                                    .matmul(&w)?
                                    .reshape((b1, b2, m, ()))?
                            } else {
                                let w = dequantized_w.broadcast_left((b1, b2))?.t()?;
                                x.matmul(&w)?
                            }
                        }
                        [bsize, m, k] => {
                            if x.is_contiguous() {
                                let w = dequantized_w.t()?;
                                x.reshape((bsize * m, k))?
                                    .matmul(&w)?
                                    .reshape((bsize, m, ()))?
                            } else {
                                let w = dequantized_w.broadcast_left(bsize)?.t()?;
                                x.matmul(&w)?
                            }
                        }
                        _ => {
                            let w = dequantized_w.t()?;
                            x.matmul(&w)?
                        }
                    };

                    let elapsed = start_time.elapsed().as_nanos() as u64;
                    saccade_core::telemetry::TELEMETRY.total_elapsed_ns.fetch_add(elapsed, std::sync::atomic::Ordering::Relaxed);

                    // Record bypass telemetry
                    saccade_core::telemetry::log_bypass_decision(op.in_features, op.out_features);

                    match bias {
                        None => Ok(y),
                        Some(b) => y.broadcast_add(b),
                    }
                } else {
                    // C-TARQ Adaptive Path: forward execution to Saccade custom op kernel.
                    // The Saccade kernel operates on F16 activations.
                    let orig_dtype = x.dtype();
                    let x_f16 = if orig_dtype != candle::DType::F16 {
                        x.to_dtype(candle::DType::F16)?
                    } else {
                        x.clone()
                    };

                    let out = x_f16.apply_op1_no_bwd(op.as_ref())?;
                    let out_scaled = if orig_dtype != candle::DType::F16 {
                        out.to_dtype(orig_dtype)?
                    } else {
                        out
                    };

                    match bias {
                        Some(b) => out_scaled.broadcast_add(b),
                        None => Ok(out_scaled),
                    }
                }
            }
        }
    }
}

/// Helper to search up the VarBuilder prefix path for C-TARQ thresholds
fn find_thresholds(vb: &crate::VarBuilder) -> (f32, f32) {
    let mut t4 = 1.0f32;
    let mut t8 = 5.0f32;
    let prefix = vb.prefix();
    let parts: Vec<&str> = prefix.split('.').collect();
    
    // Walk up parent prefixes
    for len in (1..=parts.len()).rev() {
        let parent_prefix = parts[..len].join(".");
        let t4_key = format!("{}.saccade_t4", parent_prefix);
        let t8_key = format!("{}.saccade_t8", parent_prefix);
        
        let root_vb = vb.root();
        if root_vb.contains_tensor(&t4_key) {
            if let Ok(t) = root_vb.get_unchecked(&t4_key) {
                if let Ok(v) = saccade_core::SaccadeEngine::extract_scalar_f32_pub(Some(&t)) {
                    t4 = v;
                }
            }
        }
        if root_vb.contains_tensor(&t8_key) {
            if let Ok(t) = root_vb.get_unchecked(&t8_key) {
                if let Ok(v) = saccade_core::SaccadeEngine::extract_scalar_f32_pub(Some(&t)) {
                    t8 = v;
                }
            }
        }
    }
    (t4, t8)
}

/// Create or initialize a new linear layer.
///
/// This uses some default names for weights and biases, namely `"weight"` and `"bias"`.
pub fn linear(in_dim: usize, out_dim: usize, vb: crate::VarBuilder) -> Result<Linear> {
    if vb.contains_tensor("saccade_packed_base") {
        let packed_base = vb.get_with_hints_dtype((out_dim, in_dim / 8), "saccade_packed_base", Default::default(), DType::U32)?;
        let scale_base = vb.get_with_hints_dtype((out_dim,), "saccade_scale_base", Default::default(), DType::F16)?;
        
        let sparse_delta = if vb.contains_tensor("saccade_delta_row_ptrs") {
            let row_ptrs = vb.get_unchecked_dtype("saccade_delta_row_ptrs", DType::U32)?;
            let col_indices = vb.get_unchecked_dtype("saccade_delta_col_indices", DType::U32)?;
            let values = vb.get_unchecked_dtype("saccade_delta_values", DType::U8)?;
            let scale = vb.get_unchecked_dtype("saccade_delta_scale", DType::F16)?;
            Some(saccade_core::config::SparseDeltaMatrix {
                row_ptrs,
                col_indices,
                values,
                scale,
            })
        } else {
            None
        };

        let (t4, t8) = find_thresholds(&vb);
        let config = saccade_core::SaccadeConfig {
            t4,
            t8,
            block_size: 16,
            heuristic: saccade_core::variance_heuristic,
        };

        let bias = if vb.contains_tensor("bias") {
            Some(vb.get((out_dim,), "bias")?)
        } else {
            None
        };

        Linear::new_saccade(packed_base, scale_base, sparse_delta, config, in_dim, out_dim, bias)
    } else {
        let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
        let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
        let bound = 1. / (in_dim as f64).sqrt();
        let init_bs = crate::Init::Uniform {
            lo: -bound,
            up: bound,
        };
        let bs = vb.get_with_hints(out_dim, "bias", init_bs)?;
        Ok(Linear::new(ws, Some(bs)))
    }
}

/// Create or initialize a new linear layer without biases.
pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: crate::VarBuilder) -> Result<Linear> {
    if vb.contains_tensor("saccade_packed_base") {
        let packed_base = vb.get_with_hints_dtype((out_dim, in_dim / 8), "saccade_packed_base", Default::default(), DType::U32)?;
        let scale_base = vb.get_with_hints_dtype((out_dim,), "saccade_scale_base", Default::default(), DType::F16)?;
        
        let sparse_delta = if vb.contains_tensor("saccade_delta_row_ptrs") {
            let row_ptrs = vb.get_unchecked_dtype("saccade_delta_row_ptrs", DType::U32)?;
            let col_indices = vb.get_unchecked_dtype("saccade_delta_col_indices", DType::U32)?;
            let values = vb.get_unchecked_dtype("saccade_delta_values", DType::U8)?;
            let scale = vb.get_unchecked_dtype("saccade_delta_scale", DType::F16)?;
            Some(saccade_core::config::SparseDeltaMatrix {
                row_ptrs,
                col_indices,
                values,
                scale,
            })
        } else {
            None
        };

        let (t4, t8) = find_thresholds(&vb);
        let config = saccade_core::SaccadeConfig {
            t4,
            t8,
            block_size: 16,
            heuristic: saccade_core::variance_heuristic,
        };

        Linear::new_saccade(packed_base, scale_base, sparse_delta, config, in_dim, out_dim, None)
    } else {
        let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
        let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
        Ok(Linear::new(ws, None))
    }
}

pub fn linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: crate::VarBuilder,
) -> Result<Linear> {
    if bias {
        linear(in_dim, out_dim, vb)
    } else {
        linear_no_bias(in_dim, out_dim, vb)
    }
}
