//! Resident-block variants of paged decode attention: the driver-reported residency of the
//! two and three block kernels, and bitwise equality of the three-block kernels with the
//! shipped two-block kernels over the case table of `attn_resident_table`.
//! Run with `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-kernels --features cuda --test attn_resident_gpu -- --ignored`.
#![cfg(feature = "cuda")]

#[path = "attn_resident_table/mod.rs"]
mod table;

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaSlice, CudaStream};
use scriptor_cuda::{CapturedGraph, CudaError, Device};
use scriptor_kernels::attention::{
    AttnResidency, DecodeCapacity, DecodeGeometry, DecodeWorkspace, PagedDecodeAttention,
    PairOrderKernel, PairOrderMode, SeqTables,
};
use table::{Case, LenKind, Rng};

const HD: usize = 128;
const SM_COUNT: usize = 48;

fn gpu_enabled() -> bool {
    std::env::var("SCRIPTOR_GPU_TESTS").as_deref() == Ok("1")
}

fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn random_bf16(rng: &mut Rng, n: usize, amp: f32) -> Vec<u16> {
    (0..n).map(|_| bf16_bits(rng.next() * amp)).collect()
}

fn phase(name: &str, start: &Instant) {
    eprintln!(
        "phase {name} elapsed_s={:.2}",
        start.elapsed().as_secs_f64()
    );
}

/// Split-K workspace for one `(batch, splits, q_len)` launch whose pair order is allocated at
/// every pair count.
fn workspace(
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
    fn new(
        stream: &Arc<CudaStream>,
        seed: u64,
        lens: &[usize],
        q_len: usize,
        block: usize,
    ) -> Self {
        let geo = DecodeGeometry::text(block);
        let batch = lens.len();
        let max_len = *lens.iter().max().unwrap();
        let max_blocks = max_len.div_ceil(block);
        let num_blocks: usize = lens.iter().map(|l| l.div_ceil(block)).sum::<usize>() + 1;
        let mut tables = vec![0i32; batch * max_blocks];
        let mut next = 0;
        for (s, l) in lens.iter().enumerate() {
            for b in 0..l.div_ceil(block) {
                tables[s * max_blocks + b] = next;
                next += 1;
            }
        }
        let mut rng = Rng(seed);
        let cache = num_blocks * block * geo.kv_heads * HD;
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

    fn out_len(&self) -> usize {
        self.batch * self.q_len * self.geo.q_heads * HD
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

    /// One launch of `kernel` into a fresh output, building the pair order first when the
    /// kernel is ordered; returns the raw bf16 bits.
    fn launch(
        &self,
        kernel: &PagedDecodeAttention,
        builder: Option<&PairOrderKernel>,
        splits: usize,
        chunk: usize,
        before: impl FnOnce(),
    ) -> Vec<u16> {
        let mut ws = workspace(
            &self.stream,
            self.batch,
            splits,
            self.q_len,
            self.geo.q_heads,
        );
        let mut out = self.stream.alloc_zeros::<u16>(self.out_len()).unwrap();
        if let Some(builder) = builder {
            builder
                .build(&self.stream, &self.lens, &mut ws, self.batch, splits, chunk)
                .unwrap();
        }
        before();
        self.run(kernel, &mut ws, &mut out, splits, chunk).unwrap();
        self.stream.synchronize().unwrap();
        self.stream.clone_dtoh(&out).unwrap()
    }
}

/// Runs every case of one (batch, length kind) partition: for each scene the two-block and
/// three-block kernels of both variants launch on identical inputs and must agree on every bit.
fn check_partition(batch: usize, kind: LenKind) {
    let start = Instant::now();
    let dev = Device::open().expect("gpu");
    let stream = dev.new_stream().expect("stream");
    let builder = PairOrderKernel::load(dev.context()).expect("builder");
    let mut kernels = Vec::new();
    for ordered in table::ORDERED {
        kernels.push((
            ordered,
            PagedDecodeAttention::load_with(dev.context(), AttnResidency::Two, ordered)
                .expect("two"),
            PagedDecodeAttention::load_with(dev.context(), AttnResidency::Three, ordered)
                .expect("three"),
        ));
    }
    let cases = table::partition(batch, kind);
    let mut index = 0;
    while index < cases.len() {
        let head = cases[index];
        let mut rng = Rng(table::rng_seed(head.seed, batch, head.q_len));
        let lens = table::lengths(&mut rng, kind, batch, head.q_len);
        let scene = Scene::new(
            &stream,
            table::scene_seed(head.seed, batch),
            &lens,
            head.q_len,
            head.block,
        );
        let max_len = *lens.iter().max().unwrap();
        let same_scene =
            |c: &Case| c.q_len == head.q_len && c.block == head.block && c.seed == head.seed;
        while index < cases.len() && same_scene(&cases[index]) {
            let c = cases[index];
            let (_, two, three) = kernels.iter().find(|k| k.0 == c.ordered).unwrap();
            let max_ctx = max_len * c.ctx_multiplier;
            let (splits, chunk) =
                PagedDecodeAttention::plan_splits(&scene.geo, batch, max_ctx, c.sm_count);
            let order = c.ordered.then_some(&builder);
            let want = scene.launch(two, order, splits, chunk, || {});
            let got = scene.launch(three, order, splits, chunk, || {});
            assert!(
                want == got,
                "{c:?} splits {splits} chunk {chunk}: three-block output differs"
            );
            index += 1;
        }
        phase(
            &format!(
                "b{batch}_{kind:?}_q{}_bs{}_s{}",
                head.q_len, head.block, head.seed
            ),
            &start,
        );
    }
}

macro_rules! partition_tests {
    ($($name:ident: $batch:expr, $kind:expr;)*) => {$(
        #[test]
        #[ignore = "needs GPU"]
        fn $name() {
            if !gpu_enabled() {
                return;
            }
            check_partition($batch, $kind);
        }
    )*};
}

partition_tests! {
    resident3_matches_resident2_c1_mixed: 1, LenKind::Mixed;
    resident3_matches_resident2_c1_tail: 1, LenKind::Tail;
    resident3_matches_resident2_c1_uniform: 1, LenKind::Uniform;
    resident3_matches_resident2_c8_mixed: 8, LenKind::Mixed;
    resident3_matches_resident2_c8_tail: 8, LenKind::Tail;
    resident3_matches_resident2_c8_uniform: 8, LenKind::Uniform;
    resident3_matches_resident2_c16_mixed: 16, LenKind::Mixed;
    resident3_matches_resident2_c16_tail: 16, LenKind::Tail;
    resident3_matches_resident2_c16_uniform: 16, LenKind::Uniform;
    resident3_matches_resident2_c32_mixed: 32, LenKind::Mixed;
    resident3_matches_resident2_c32_tail: 32, LenKind::Tail;
    resident3_matches_resident2_c32_uniform: 32, LenKind::Uniform;
    resident3_matches_resident2_c64_mixed: 64, LenKind::Mixed;
    resident3_matches_resident2_c64_tail: 64, LenKind::Tail;
    resident3_matches_resident2_c64_uniform: 64, LenKind::Uniform;
    resident3_matches_resident2_c128_mixed: 128, LenKind::Mixed;
    resident3_matches_resident2_c128_tail: 128, LenKind::Tail;
    resident3_matches_resident2_c128_uniform: 128, LenKind::Uniform;
}

#[test]
#[ignore = "needs GPU"]
fn r3_reports_three_resident_blocks() {
    if !gpu_enabled() {
        return;
    }
    let dev = Device::open().expect("gpu");
    for ordered in table::ORDERED {
        let kernel = PagedDecodeAttention::load_with(dev.context(), AttnResidency::Three, ordered)
            .expect("three");
        let blocks = kernel.resident_blocks_per_sm().expect("occupancy");
        assert!(
            blocks >= 3,
            "ordered {ordered}: {blocks} resident blocks per SM"
        );
    }
}

#[test]
#[ignore = "needs GPU"]
fn r2_reports_at_least_two_resident_blocks() {
    if !gpu_enabled() {
        return;
    }
    let dev = Device::open().expect("gpu");
    for ordered in table::ORDERED {
        let kernel = PagedDecodeAttention::load_with(dev.context(), AttnResidency::Two, ordered)
            .expect("two");
        let blocks = kernel.resident_blocks_per_sm().expect("occupancy");
        assert!(
            blocks >= 2,
            "ordered {ordered}: {blocks} resident blocks per SM"
        );
    }
}

#[test]
#[ignore = "needs GPU"]
fn resident3_equals_resident2_with_poisoned_shared_memory() {
    if !gpu_enabled() {
        return;
    }
    let dev = Device::open().expect("gpu");
    let stream = dev.new_stream().expect("stream");
    let ctx = dev.context().clone();
    let builder = PairOrderKernel::load(&ctx).expect("builder");
    for ordered in table::ORDERED {
        let two = PagedDecodeAttention::load_with(&ctx, AttnResidency::Two, ordered).expect("two");
        let three =
            PagedDecodeAttention::load_with(&ctx, AttnResidency::Three, ordered).expect("three");
        for (batch, q_len, block) in [(1, 1, 16), (8, 4, 32), (32, 1, 16), (32, 2, 32)] {
            let mut rng = Rng(table::rng_seed(0, batch, q_len));
            let lens = table::lengths(&mut rng, LenKind::Mixed, batch, q_len);
            let scene = Scene::new(&stream, table::scene_seed(0, batch), &lens, q_len, block);
            let max_len = *lens.iter().max().unwrap();
            let (splits, chunk) =
                PagedDecodeAttention::plan_splits(&scene.geo, batch, max_len, SM_COUNT);
            let order = ordered.then_some(&builder);
            let want = scene.launch(&two, order, splits, chunk, || {});
            let got = scene.launch(&three, order, splits, chunk, || {
                PagedDecodeAttention::poison_shared_memory(
                    &ctx,
                    &stream,
                    4 * SM_COUNT as u32 * 8,
                    40_000,
                )
                .unwrap();
            });
            assert!(
                want == got,
                "ordered {ordered} batch {batch} q_len {q_len} bs {block}: poisoned three-block output differs"
            );
        }
    }
}

#[test]
#[ignore = "needs GPU"]
fn resident3_graph_replay_matches_resident2_eager() {
    if !gpu_enabled() {
        return;
    }
    let dev = Device::open().expect("gpu");
    // SAFETY: every buffer of this test is allocated, written and read on the one stream
    // below, so stream order alone serialises them, as in the executor; without tracking no
    // event recorded before the capture becomes a dependency of the captured work.
    unsafe { dev.context().disable_event_tracking() };
    let stream = dev.new_stream().expect("stream");
    let eager =
        PagedDecodeAttention::load_with(dev.context(), AttnResidency::Two, false).expect("two");
    let ordered3 =
        PagedDecodeAttention::load_with(dev.context(), AttnResidency::Three, true).expect("three");
    let builder = PairOrderKernel::load(dev.context()).expect("builder");
    let (batch, q_len, block) = (64, 1, 16);
    let mut rng = Rng(0x2468_ace0);
    let first = table::lengths(&mut rng, LenKind::Mixed, batch, q_len);
    let mut scene = Scene::new(&stream, 99, &first, q_len, block);
    let max_len = *first.iter().max().unwrap();
    let (splits, chunk) = PagedDecodeAttention::plan_splits(&scene.geo, batch, max_len, SM_COUNT);
    let mut ws_graph = workspace(&stream, batch, splits, q_len, scene.geo.q_heads);
    let mut ws_ref = workspace(&stream, batch, splits, q_len, scene.geo.q_heads);
    let mut got = stream.alloc_zeros::<u16>(scene.out_len()).unwrap();
    let mut want = stream.alloc_zeros::<u16>(scene.out_len()).unwrap();
    let graph = CapturedGraph::capture(&stream, |_| {
        builder.build(&stream, &scene.lens, &mut ws_graph, batch, splits, chunk)?;
        scene.run(&ordered3, &mut ws_graph, &mut got, splits, chunk)?;
        scene.run(&ordered3, &mut ws_graph, &mut got, splits, chunk)
    })
    .expect("capture");
    for round in 0..4 {
        let next: Vec<usize> = first
            .iter()
            .map(|&l| (1 + (l * (round + 2) + 13 * round) % max_len).max(q_len))
            .collect();
        let host: Vec<i32> = next.iter().map(|&l| l as i32).collect();
        stream.memcpy_htod(&host, &mut scene.lens).unwrap();
        graph.launch().unwrap();
        scene
            .run(&eager, &mut ws_ref, &mut want, splits, chunk)
            .unwrap();
        stream.synchronize().unwrap();
        let (w, g): (Vec<u16>, Vec<u16>) = (
            stream.clone_dtoh(&want).unwrap(),
            stream.clone_dtoh(&got).unwrap(),
        );
        assert!(
            w == g,
            "round {round}: three-block replay differs from two-block eager"
        );
    }
}
