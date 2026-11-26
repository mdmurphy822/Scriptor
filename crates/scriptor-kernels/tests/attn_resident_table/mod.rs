//! Case table of the resident-block equality tests: the single source of every
//! (batch, length kind, query length, block size, seed, kernel variant, context bound,
//! SM count) combination the GPU tests run, and the sequence lengths each one launches.
#![allow(dead_code)]

/// Decode batch sizes, one GPU test group each.
pub const BATCHES: [usize; 6] = [1, 8, 16, 32, 64, 128];
/// KV block sizes.
pub const BLOCKS: [usize; 2] = [16, 32];
/// SM counts passed to the split planner.
pub const SM_COUNTS: [usize; 3] = [48, 12, 8];
/// Context-bound multipliers over the longest live sequence.
pub const CTX_MULTIPLIERS: [usize; 2] = [1, 3];
/// Kernel variants: unordered, then ordered.
pub const ORDERED: [bool; 2] = [false, true];
/// Seed sets: (rng salt, scene salt).
pub const SEED_SETS: [(u64, u64); 2] = [(0x1357_9bdf, 0xabcd), (0x2468_ace0, 0x5eed)];
/// Longest sequence any case holds.
pub const MAX_LEN: usize = 2048;

/// Shape of the sequence lengths of one scene.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LenKind {
    /// Every third slot idle, the others `q_len + 1 + below(700)`.
    Mixed,
    /// One sequence at [`MAX_LEN`], the rest `q_len + 1 + below(300)`.
    Tail,
    /// Every slot at 273.
    Uniform,
}

pub const LEN_KINDS: [LenKind; 3] = [LenKind::Mixed, LenKind::Tail, LenKind::Uniform];

/// Query lengths run at `batch`: 1, 2 and 4, plus 8 (the 16-row edge) at batch 8.
pub fn q_lens(batch: usize) -> Vec<usize> {
    let mut q = vec![1, 2, 4];
    if batch == 8 {
        q.push(8);
    }
    q
}

/// One launch pair: the same scene run at two and at three resident blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Case {
    pub batch: usize,
    pub kind: LenKind,
    pub q_len: usize,
    pub block: usize,
    pub seed: usize,
    pub ordered: bool,
    pub ctx_multiplier: usize,
    pub sm_count: usize,
}

/// Cases of one (batch, length kind) test.
pub fn partition(batch: usize, kind: LenKind) -> Vec<Case> {
    let mut cases = Vec::new();
    for q_len in q_lens(batch) {
        for block in BLOCKS {
            for seed in 0..SEED_SETS.len() {
                for ordered in ORDERED {
                    for ctx_multiplier in CTX_MULTIPLIERS {
                        for sm_count in SM_COUNTS {
                            cases.push(Case {
                                batch,
                                kind,
                                q_len,
                                block,
                                seed,
                                ordered,
                                ctx_multiplier,
                                sm_count,
                            });
                        }
                    }
                }
            }
        }
    }
    cases
}

/// Every (batch, length kind) test of the GPU binary.
pub fn tests() -> Vec<(usize, LenKind)> {
    BATCHES
        .iter()
        .flat_map(|&b| LEN_KINDS.iter().map(move |&k| (b, k)))
        .collect()
}

/// Every case, enumerated directly over the axes.
pub fn full() -> Vec<Case> {
    let mut cases = Vec::new();
    for batch in BATCHES {
        for q_len in q_lens(batch) {
            for kind in LEN_KINDS {
                for block in BLOCKS {
                    for seed in 0..SEED_SETS.len() {
                        for ordered in ORDERED {
                            for ctx_multiplier in CTX_MULTIPLIERS {
                                for sm_count in SM_COUNTS {
                                    cases.push(Case {
                                        batch,
                                        kind,
                                        q_len,
                                        block,
                                        seed,
                                        ordered,
                                        ctx_multiplier,
                                        sm_count,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    cases
}

pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    pub fn below(&mut self, n: usize) -> usize {
        self.next().to_bits() as usize % n.max(1)
    }
}

/// Rng seed of a scene: the seed set's salt mixed with the batch and query length.
pub fn rng_seed(seed: usize, batch: usize, q_len: usize) -> u64 {
    SEED_SETS[seed].0 ^ (batch * 31 + q_len) as u64
}

/// Scene seed: the seed set's salt mixed with the batch.
pub fn scene_seed(seed: usize, batch: usize) -> u64 {
    SEED_SETS[seed].1 ^ batch as u64
}

/// Sequence lengths of `batch` slots of shape `kind`; each holds at least its `q_len` new rows.
pub fn lengths(rng: &mut Rng, kind: LenKind, batch: usize, q_len: usize) -> Vec<usize> {
    match kind {
        LenKind::Mixed => (0..batch)
            .map(|s| {
                if s % 3 == 2 {
                    q_len
                } else {
                    q_len + 1 + rng.below(700)
                }
            })
            .collect(),
        LenKind::Tail => {
            let mut lens: Vec<usize> = (0..batch).map(|_| q_len + 1 + rng.below(300)).collect();
            lens[batch / 2] = MAX_LEN;
            lens
        }
        LenKind::Uniform => vec![273; batch],
    }
}
