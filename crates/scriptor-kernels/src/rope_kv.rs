//! Selection of the fused RoPE + KV-append kernel variant for one pass.
//!
//! Two independent, default-off variants exist beside the reference kernel
//! (`scriptor_fused_rope_kv_split_bf16`): shared per-token sin/cos, and, for pure decode, no
//! standalone `k` / `v` row writes. [`RopeKvSelect`] holds the caller's choice and
//! [`RopeKvSelect::plan`] resolves it against the row mix and head shape of one pass.

/// Largest rotary pair count the shared sin/cos variant stages (`head_dim` up to 1024).
pub const MAX_SHARED_PAIRS: usize = 512;

/// Variants requested by the caller; both off selects the reference kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RopeKvSelect {
    /// Compute each pair's sin/cos once per token block instead of once per head.
    pub shared_trig: bool,
    /// In a pure-decode pass, do not write the standalone `k` / `v` rows.
    pub skip_kv_scratch: bool,
}

/// Row mix and head shape of one pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RopeKvShape {
    /// Rows of the pass.
    pub rows: usize,
    /// Rows that belong to prefill sequences; the rest are decode rows.
    pub prefill_rows: usize,
    /// Elements per head.
    pub head_dim: usize,
    /// Rotary pair counts of the three M-RoPE position components.
    pub sections: [usize; 3],
}

/// The variant a pass launches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RopeKvPlan {
    /// Sin/cos staged in shared memory once per token block.
    pub shared_trig: bool,
    /// Standalone `k` / `v` rows left unwritten (and stale).
    pub skip_kv_scratch: bool,
}

impl RopeKvPlan {
    /// Whether the reference kernel runs.
    pub fn is_reference(&self) -> bool {
        !self.shared_trig && !self.skip_kv_scratch
    }

    /// Dynamic shared memory bytes the launch needs for a head of `head_dim` elements.
    pub fn shared_bytes(&self, head_dim: usize) -> u32 {
        if self.shared_trig {
            (2 * (head_dim / 2) * std::mem::size_of::<f32>()) as u32
        } else {
            0
        }
    }

    /// Whether a pass over `prefill_rows` prefill rows may use a graph or launch sequence built
    /// for this plan: a plan that leaves `k` / `v` unwritten never serves a pass that has
    /// prefill rows, which read them.
    pub fn serves(&self, prefill_rows: usize) -> bool {
        !self.skip_kv_scratch || prefill_rows == 0
    }
}

impl RopeKvSelect {
    /// Resolves the selection for `shape`. A variant is dropped, falling back toward the
    /// reference kernel, when its preconditions fail: skipping the standalone rows needs a pass
    /// with rows and no prefill rows; any variant needs an even head size whose sections cover
    /// its pairs; the shared variant also needs the pairs to fit [`MAX_SHARED_PAIRS`].
    pub fn plan(&self, shape: &RopeKvShape) -> RopeKvPlan {
        let pairs = shape.head_dim / 2;
        let shaped = shape.head_dim > 0
            && shape.head_dim.is_multiple_of(2)
            && shape.sections.iter().sum::<usize>() == pairs;
        if !shaped {
            return RopeKvPlan::default();
        }
        RopeKvPlan {
            shared_trig: self.shared_trig && pairs <= MAX_SHARED_PAIRS,
            skip_kv_scratch: self.skip_kv_scratch && shape.rows > 0 && shape.prefill_rows == 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: RopeKvSelect = RopeKvSelect {
        shared_trig: true,
        skip_kv_scratch: true,
    };

    fn shape(rows: usize, prefill_rows: usize) -> RopeKvShape {
        RopeKvShape {
            rows,
            prefill_rows,
            head_dim: 128,
            sections: [16, 24, 24],
        }
    }

    #[test]
    fn default_selects_the_reference() {
        let plan = RopeKvSelect::default().plan(&shape(8, 0));
        assert!(plan.is_reference());
    }

    #[test]
    fn pure_decode_takes_both_variants() {
        let plan = ALL.plan(&shape(8, 0));
        assert!(plan.shared_trig && plan.skip_kv_scratch);
        assert_eq!(plan.shared_bytes(128), 512);
    }

    #[test]
    fn mixed_and_prefill_keep_the_standalone_rows() {
        for prefill in [1, 5, 8] {
            let plan = ALL.plan(&shape(8, prefill));
            assert!(plan.shared_trig, "prefill {prefill}");
            assert!(!plan.skip_kv_scratch, "prefill {prefill}");
        }
    }

    #[test]
    fn variants_resolve_independently() {
        let a = RopeKvSelect {
            shared_trig: true,
            skip_kv_scratch: false,
        };
        let b = RopeKvSelect {
            shared_trig: false,
            skip_kv_scratch: true,
        };
        assert_eq!(
            a.plan(&shape(4, 0)),
            RopeKvPlan {
                shared_trig: true,
                skip_kv_scratch: false
            }
        );
        assert_eq!(
            b.plan(&shape(4, 0)),
            RopeKvPlan {
                shared_trig: false,
                skip_kv_scratch: true
            }
        );
        assert!(b.plan(&shape(4, 2)).is_reference());
        assert_eq!(a.plan(&shape(4, 2)).shared_bytes(128), 512);
        assert_eq!(b.plan(&shape(4, 0)).shared_bytes(128), 0);
    }

    #[test]
    fn empty_pass_never_skips() {
        assert!(!ALL.plan(&shape(0, 0)).skip_kv_scratch);
    }

    #[test]
    fn unsupported_head_shapes_use_the_reference() {
        let odd = RopeKvShape {
            head_dim: 127,
            sections: [21, 21, 21],
            ..shape(8, 0)
        };
        assert!(ALL.plan(&odd).is_reference());
        let zero = RopeKvShape {
            head_dim: 0,
            sections: [0; 3],
            ..shape(8, 0)
        };
        assert!(ALL.plan(&zero).is_reference());
        let mismatched = RopeKvShape {
            sections: [16, 24, 23],
            ..shape(8, 0)
        };
        assert!(ALL.plan(&mismatched).is_reference());
    }

    #[test]
    fn shared_variant_is_bounded_by_the_pair_limit() {
        let wide = |head_dim: usize| RopeKvShape {
            head_dim,
            sections: [head_dim / 2, 0, 0],
            ..shape(8, 0)
        };
        assert!(ALL.plan(&wide(2 * MAX_SHARED_PAIRS)).shared_trig);
        let over = ALL.plan(&wide(2 * MAX_SHARED_PAIRS + 2));
        assert!(!over.shared_trig && over.skip_kv_scratch);
    }

    #[test]
    fn a_skipping_plan_never_serves_prefill() {
        let decode = ALL.plan(&shape(8, 0));
        assert!(decode.serves(0));
        assert!(!decode.serves(1));
        assert!(!decode.serves(8));
        let reference = RopeKvPlan::default();
        assert!(reference.serves(0) && reference.serves(3));
        let shared_only = RopeKvPlan {
            shared_trig: true,
            skip_kv_scratch: false,
        };
        assert!(shared_only.serves(3));
    }

    #[test]
    fn plan_follows_the_pass_each_time() {
        // One selection, alternating decode and prefill passes: the skip never carries over.
        for (rows, prefill, skips) in [(8, 0, true), (8, 3, false), (4, 0, true), (6, 6, false)] {
            assert_eq!(ALL.plan(&shape(rows, prefill)).skip_kv_scratch, skips);
        }
    }
}
