//! Hoisted pair order for paged decode attention: the order kernel against its CPU reference,
//! bitwise equality of the ordered attention kernel with the default one, and graph replay.
//! Run with `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-kernels --features cuda --test attn_pair_order_gpu -- --ignored`.
#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use scriptor_cuda::{CapturedGraph, CudaError, Device};
use scriptor_kernels::attention::{
    DecodeCapacity, DecodeGeometry, DecodeWorkspace, PagedDecodeAttention, PairOrderKernel,
    PairOrderMode, SeqTables, pair_order_reference,
};

const SM_COUNT: usize = 48;
const BLOCK: usize = 16;
const HD: usize = 128;

/// Split-K workspace for one `(batch, splits, q_len)` launch whose pair order is allocated at
/// every pair count.
fn ordered_workspace(
    stream: &Arc<CudaStream>,
    batch: usize,
    splits: usize,
    q_len: usize,
    q_heads: usize,
) -> DecodeWorkspace {
    let capacity = DecodeCapacity::covering(
        [(batch, splits, q_len)],
        q_heads,
        PairOrderMode::Hoisted { min_pairs: 1 },
    );
    DecodeWorkspace::with_capacity(stream, capacity).unwrap()
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn below(&mut self, n: usize) -> usize {
        self.next().to_bits() as usize % n.max(1)
    }
}

fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn gpu_enabled() -> bool {
    std::env::var("SCRIPTOR_GPU_TESTS").as_deref() == Ok("1")
}

fn random_bf16(rng: &mut Rng, n: usize, amp: f32) -> Vec<u16> {
    (0..n).map(|_| bf16_bits(rng.next() * amp)).collect()
}

/// Sequence lengths of `batch` slots: varied contexts, with every third slot idle (holding only
/// its `q_len` new rows) like a padded graph bucket.
fn lengths(rng: &mut Rng, batch: usize, q_len: usize) -> Vec<usize> {
    (0..batch)
        .map(|s| {
            if s % 3 == 2 {
                q_len
            } else {
                q_len + 1 + rng.below(700)
            }
        })
        .collect()
}

struct Scene {
    stream: Arc<CudaStream>,
    geo: DecodeGeometry,
    batch: usize,
    q_len: usize,
    max_blocks: usize,
    kc: CudaSlice<u16>,
    vc: CudaSlice<u16>,
    q: CudaSlice<u16>,
    tables: CudaSlice<i32>,
    lens: CudaSlice<i32>,
}

impl Scene {
    fn new(stream: &Arc<CudaStream>, seed: u64, lens: &[usize], q_len: usize) -> Self {
        let geo = DecodeGeometry::text(BLOCK);
        let batch = lens.len();
        let max_len = *lens.iter().max().unwrap();
        let max_blocks = max_len.div_ceil(BLOCK);
        let num_blocks: usize = lens.iter().map(|l| l.div_ceil(BLOCK)).sum::<usize>() + 1;
        let mut tables = vec![0i32; batch * max_blocks];
        let mut next = 0;
        for (s, l) in lens.iter().enumerate() {
            for b in 0..l.div_ceil(BLOCK) {
                tables[s * max_blocks + b] = next;
                next += 1;
            }
        }
        let mut rng = Rng(seed);
        let cache = num_blocks * BLOCK * geo.kv_heads * HD;
        let kc = random_bf16(&mut rng, cache, 1.0);
        let vc = random_bf16(&mut rng, cache, 2.0);
        let q = random_bf16(&mut rng, batch * q_len * geo.q_heads * HD, 1.0);
        Self {
            stream: stream.clone(),
            geo,
            batch,
            q_len,
            max_blocks,
            kc: stream.clone_htod(&kc).unwrap(),
            vc: stream.clone_htod(&vc).unwrap(),
            q: stream.clone_htod(&q).unwrap(),
            tables: stream.clone_htod(&tables).unwrap(),
            lens: stream
                .clone_htod(&lens.iter().map(|&l| l as i32).collect::<Vec<_>>())
                .unwrap(),
        }
    }

    fn plan(&self, max_ctx: usize) -> (usize, usize, DecodeWorkspace) {
        let (splits, chunk) =
            PagedDecodeAttention::plan_splits(&self.geo, self.batch, max_ctx, SM_COUNT);
        let ws = ordered_workspace(
            &self.stream,
            self.batch,
            splits,
            self.q_len,
            self.geo.q_heads,
        );
        (splits, chunk, ws)
    }

    fn run(
        &self,
        kernel: &PagedDecodeAttention,
        ws: &mut DecodeWorkspace,
        out: &mut CudaSlice<u16>,
        splits: usize,
        chunk: usize,
    ) -> Result<(), CudaError> {
        let tables = SeqTables {
            block_tables: &self.tables,
            max_blocks: self.max_blocks,
        };
        kernel.run(
            &self.stream,
            &self.geo,
            &self.kc,
            &self.vc,
            &tables,
            &self.lens,
            &self.q,
            out,
            ws,
            self.batch,
            self.q_len,
            splits,
            chunk,
            (HD as f32).powf(-0.5),
        )
    }
}

fn host_lens(stream: &Arc<CudaStream>, d: &CudaSlice<i32>) -> Vec<i32> {
    stream.clone_dtoh(d).unwrap()
}

#[test]
#[ignore = "needs GPU"]
fn order_kernel_matches_cpu_reference() {
    if !gpu_enabled() {
        return;
    }
    let dev = Device::open().expect("gpu");
    let stream = dev.new_stream().expect("stream");
    let builder = PairOrderKernel::load(dev.context()).expect("kernel");
    let mut rng = Rng(0xfeed_f00d_1234_5678);
    let shapes: [(usize, usize, usize); 8] = [
        (1, 1, 64),
        (1, 12, 128),
        (3, 4, 64),
        (32, 2, 192),
        (64, 3, 64),
        (128, 1, 640),
        (128, 8, 128),
        (512, 8, 64),
    ];
    for (batch, splits, chunk) in shapes {
        let mut lens: Vec<i32> = (0..batch)
            .map(|_| rng.below(splits * chunk + 40) as i32)
            .collect();
        for l in lens.iter_mut().step_by(3) {
            *l = 0;
        }
        let d_lens = stream.clone_htod(&lens).unwrap();
        let mut ws = ordered_workspace(&stream, batch, splits, 1, 16);
        builder
            .build(&stream, &d_lens, &mut ws, batch, splits, chunk)
            .unwrap();
        stream.synchronize().unwrap();
        let got = ws.read_pair_order(&stream, batch * splits).unwrap();
        assert_eq!(
            got,
            pair_order_reference(&lens, splits, chunk),
            "batch {batch} splits {splits} chunk {chunk}"
        );
        assert!(ws.pair_order_matches(batch, splits, chunk));
        assert!(!ws.pair_order_matches(batch, splits + 1, chunk));
    }
}

#[test]
#[ignore = "needs GPU"]
fn ordered_launch_requires_a_matching_order() {
    if !gpu_enabled() {
        return;
    }
    let dev = Device::open().expect("gpu");
    let stream = dev.new_stream().expect("stream");
    let ordered = PagedDecodeAttention::load_ordered(dev.context()).expect("kernel");
    let builder = PairOrderKernel::load(dev.context()).expect("builder");
    let mut rng = Rng(77);
    let lens = lengths(&mut rng, 4, 1);
    let scene = Scene::new(&stream, 5, &lens, 1);
    let (splits, chunk, mut ws) = scene.plan(*lens.iter().max().unwrap());
    let mut out = stream
        .alloc_zeros::<u16>(4 * scene.geo.q_heads * HD)
        .unwrap();
    assert!(
        scene
            .run(&ordered, &mut ws, &mut out, splits, chunk)
            .is_err()
    );
    builder
        .build(&stream, &scene.lens, &mut ws, 4, splits, chunk)
        .unwrap();
    scene
        .run(&ordered, &mut ws, &mut out, splits, chunk)
        .unwrap();
    ws.invalidate_pair_order();
    assert!(
        scene
            .run(&ordered, &mut ws, &mut out, splits, chunk)
            .is_err()
    );
}

fn check_bitwise(batch: usize, q_len: usize) {
    let dev = Device::open().expect("gpu");
    let stream = dev.new_stream().expect("stream");
    let default = PagedDecodeAttention::load(dev.context()).expect("default");
    let ordered = PagedDecodeAttention::load_ordered(dev.context()).expect("ordered");
    let builder = PairOrderKernel::load(dev.context()).expect("builder");
    let mut rng = Rng(0x1357_9bdf ^ (batch * 31 + q_len) as u64);
    let lens = lengths(&mut rng, batch, q_len);
    let scene = Scene::new(&stream, 0xabcd ^ batch as u64, &lens, q_len);
    let max_len = *lens.iter().max().unwrap();
    let n = batch * q_len * scene.geo.q_heads * HD;
    // The graph plans split-K for the configured context, not the live one, so also exercise a
    // larger bound than the longest sequence.
    for max_ctx in [max_len, max_len * 3] {
        let (splits, chunk, mut ws_a) = scene.plan(max_ctx);
        let (_, _, mut ws_b) = scene.plan(max_ctx);
        let mut want = stream.alloc_zeros::<u16>(n).unwrap();
        let mut got = stream.alloc_zeros::<u16>(n).unwrap();
        scene
            .run(&default, &mut ws_a, &mut want, splits, chunk)
            .unwrap();
        builder
            .build(&stream, &scene.lens, &mut ws_b, batch, splits, chunk)
            .unwrap();
        scene
            .run(&ordered, &mut ws_b, &mut got, splits, chunk)
            .unwrap();
        stream.synchronize().unwrap();
        let (w, g): (Vec<u16>, Vec<u16>) = (
            stream.clone_dtoh(&want).unwrap(),
            stream.clone_dtoh(&got).unwrap(),
        );
        assert!(
            w == g,
            "batch {batch} q_len {q_len} ctx bound {max_ctx} splits {splits}: outputs differ"
        );
    }
}

macro_rules! bitwise_cases {
    ($($name:ident: $batch:expr;)*) => {$(
        #[test]
        #[ignore = "needs GPU"]
        fn $name() {
            if !gpu_enabled() {
                return;
            }
            check_bitwise($batch, 1);
            check_bitwise($batch, 4);
        }
    )*};
}

bitwise_cases! {
    ordered_matches_default_c1: 1;
    ordered_matches_default_c8: 8;
    ordered_matches_default_c16: 16;
    ordered_matches_default_c32: 32;
    ordered_matches_default_c64: 64;
    ordered_matches_default_c128: 128;
}

#[test]
#[ignore = "needs GPU"]
fn graph_replay_rebuilds_the_order_from_the_replayed_lengths() {
    if !gpu_enabled() {
        return;
    }
    let dev = Device::open().expect("gpu");
    // SAFETY: every buffer of this test is allocated, written and read on the one stream
    // below, so stream order alone serialises them, as in the executor; without tracking no
    // event recorded before the capture becomes a dependency of the captured work.
    unsafe { dev.context().disable_event_tracking() };
    let stream = dev.new_stream().expect("stream");
    let default = PagedDecodeAttention::load(dev.context()).expect("default");
    let ordered = PagedDecodeAttention::load_ordered(dev.context()).expect("ordered");
    let builder = PairOrderKernel::load(dev.context()).expect("builder");
    let (batch, q_len) = (64, 1);
    let mut rng = Rng(0x2468_ace0);
    let first = lengths(&mut rng, batch, q_len);
    let mut scene = Scene::new(&stream, 99, &first, q_len);
    let max_len = *first.iter().max().unwrap();
    let (splits, chunk, mut ws_graph) = scene.plan(max_len);
    let (_, _, mut ws_ref) = scene.plan(max_len);
    let n = batch * q_len * scene.geo.q_heads * HD;
    let mut got = stream.alloc_zeros::<u16>(n).unwrap();
    let mut want = stream.alloc_zeros::<u16>(n).unwrap();
    let graph = CapturedGraph::capture(&stream, |_| {
        builder.build(&stream, &scene.lens, &mut ws_graph, batch, splits, chunk)?;
        scene.run(&ordered, &mut ws_graph, &mut got, splits, chunk)?;
        scene.run(&ordered, &mut ws_graph, &mut got, splits, chunk)
    })
    .expect("capture");
    for round in 0..4 {
        // Lengths within the captured context bound; the order must follow them.
        let next: Vec<usize> = first
            .iter()
            .map(|&l| (1 + (l * (round + 2) + 13 * round) % max_len).max(q_len))
            .collect();
        let host: Vec<i32> = next.iter().map(|&l| l as i32).collect();
        stream.memcpy_htod(&host, &mut scene.lens).unwrap();
        graph.launch().unwrap();
        scene
            .run(&default, &mut ws_ref, &mut want, splits, chunk)
            .unwrap();
        stream.synchronize().unwrap();
        let order = ws_graph.read_pair_order(&stream, batch * splits).unwrap();
        assert_eq!(
            order,
            pair_order_reference(&host_lens(&stream, &scene.lens), splits, chunk),
            "round {round}: replay used a stale order"
        );
        let (w, g): (Vec<u16>, Vec<u16>) = (
            stream.clone_dtoh(&want).unwrap(),
            stream.clone_dtoh(&got).unwrap(),
        );
        assert!(w == g, "round {round}: replay output differs");
    }
}
