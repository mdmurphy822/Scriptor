//! Host-side contract of the speculative-draft acceptance kernel in
//! `kernels/mtp_accept.cu`, plus the CPU references of both acceptance rules:
//! exact match for greedy rows and the rejection sampler for sampled rows.

use crate::reference::sampling::argmax_with_margin;

/// Embedded fatbin stem of the acceptance kernel.
pub const KERNEL_FILE: &str = "mtp_accept";
/// Acceptance kernel: one thread per sequence.
pub const KERNEL_MTP_ACCEPT: &str = "scriptor_mtp_accept";
/// Padding value of unused output slots.
pub const NO_TOKEN: u32 = u32::MAX;
/// Threads per block of the acceptance kernel.
pub const ACCEPT_BLOCK_THREADS: usize = 128;

/// Tokens emitted by one verification of a draft chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceptOutcome {
    /// Accepted drafts followed by the verifier's token at the first
    /// rejection, or the bonus token when every draft was accepted.
    pub tokens: Vec<u32>,
    /// Drafts accepted, equal to `tokens.len() - 1`.
    pub accepted: usize,
}

impl AcceptOutcome {
    fn from_prefix(draft: &[u32], accepted: usize, last: u32) -> Self {
        let mut tokens = draft[..accepted].to_vec();
        tokens.push(last);
        Self { tokens, accepted }
    }
}

/// Exact-match acceptance: the longest prefix of `draft` equal to `target`,
/// followed by `target` at the first mismatch. `target` holds the verifier's
/// token at each of the `draft.len() + 1` verified positions.
pub fn accept_greedy(draft: &[u32], target: &[u32]) -> AcceptOutcome {
    assert_eq!(target.len(), draft.len() + 1);
    let accepted = draft.iter().zip(target).take_while(|(d, t)| d == t).count();
    AcceptOutcome::from_prefix(draft, accepted, target[accepted])
}

/// Probability distribution a sampled row draws from: temperature scaling,
/// then the `top_k` largest and the `top_p` nucleus (ties kept), renormalised.
/// `logits` are already penalised. A non-positive temperature yields the
/// one-hot distribution of the argmax.
pub fn target_distribution(logits: &[f32], temperature: f32, top_k: i32, top_p: f32) -> Vec<f32> {
    let vocab = logits.len();
    let mut dist = vec![0.0f32; vocab];
    if temperature <= 0.0 {
        dist[argmax_with_margin(logits).token as usize] = 1.0;
        return dist;
    }
    let mut threshold = f32::NEG_INFINITY;
    if top_k > 0 && (top_k as usize) < vocab {
        let mut sorted = logits.to_vec();
        sorted.sort_by(|a, b| b.total_cmp(a));
        threshold = sorted[top_k as usize - 1];
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weight = |v: f32| f64::from(((v - max) * (1.0 / temperature)).exp());
    if top_p < 1.0 {
        let mut kept: Vec<f32> = logits.iter().copied().filter(|&v| v >= threshold).collect();
        kept.sort_by(|a, b| b.total_cmp(a));
        let z: f64 = kept.iter().map(|&v| weight(v)).sum();
        let want = f64::from(top_p) * z;
        let mut acc = 0.0f64;
        let mut cut = kept[kept.len() - 1];
        let mut i = 0;
        while i < kept.len() {
            let v = kept[i];
            while i < kept.len() && kept[i] == v {
                acc += weight(kept[i]);
                i += 1;
            }
            if acc >= want {
                cut = v;
                break;
            }
        }
        threshold = cut;
    }
    let total: f64 = logits
        .iter()
        .filter(|&&v| v >= threshold)
        .map(|&v| weight(v))
        .sum();
    for (d, &v) in dist.iter_mut().zip(logits) {
        if v >= threshold {
            *d = (weight(v) / total) as f32;
        }
    }
    dist
}

/// Draws an id from `dist` by inverse CDF in id order at uniform `u`.
pub fn sample_distribution(dist: &[f32], u: f32) -> u32 {
    let total: f64 = dist.iter().map(|&p| f64::from(p)).sum();
    let target = f64::from(u) * total;
    let mut acc = 0.0f64;
    let mut last = 0u32;
    for (i, &p) in dist.iter().enumerate() {
        if p > 0.0 {
            acc += f64::from(p);
            last = i as u32;
            if acc > target {
                return i as u32;
            }
        }
    }
    last
}

/// Draws from `dist` with `rejected` removed and the rest renormalised: the
/// correction distribution after a one-hot draft is rejected. A distribution
/// with no mass outside `rejected` returns `rejected`.
pub fn sample_residual(dist: &[f32], rejected: u32, u: f32) -> u32 {
    let mut residual = dist.to_vec();
    residual[rejected as usize] = 0.0;
    if residual.iter().all(|&p| p <= 0.0) {
        return rejected;
    }
    sample_distribution(&residual, u)
}

/// Rejection-sampling acceptance for greedy (one-hot) drafts. Draft `j` is
/// accepted when `accept_u[j] < dists[j][draft[j]]`; the first rejection emits
/// a draw from the residual of `dists[j]` at `resample_u[j]`; when every draft
/// is accepted a draw from `dists[k]` at `resample_u[k]` is the bonus token.
/// `dists` holds `draft.len() + 1` distributions; the emitted stream has the
/// same distribution as sampling each position from `dists` directly.
pub fn accept_rejection(
    draft: &[u32],
    dists: &[Vec<f32>],
    accept_u: &[f32],
    resample_u: &[f32],
) -> AcceptOutcome {
    let k = draft.len();
    assert_eq!(dists.len(), k + 1);
    assert!(accept_u.len() >= k && resample_u.len() > k);
    for j in 0..k {
        if accept_u[j] >= dists[j][draft[j] as usize] {
            let fix = sample_residual(&dists[j], draft[j], resample_u[j]);
            return AcceptOutcome::from_prefix(draft, j, fix);
        }
    }
    AcceptOutcome::from_prefix(draft, k, sample_distribution(&dists[k], resample_u[k]))
}

/// Launch geometry of the acceptance kernel for `rows` sequences.
pub fn accept_launch(rows: usize) -> crate::launch::LaunchDims {
    crate::launch::elementwise_launch(rows.max(1), ACCEPT_BLOCK_THREADS)
}

#[cfg(feature = "cuda")]
pub use device::MtpAcceptKernel;

#[cfg(feature = "cuda")]
mod device {
    use std::sync::Arc;

    use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, PushKernelArg};
    use scriptor_cuda::Result;

    use super::{KERNEL_FILE, KERNEL_MTP_ACCEPT, accept_launch};
    use crate::norm::{load_module, to_config};

    /// Device launcher of the draft-acceptance kernel.
    pub struct MtpAcceptKernel {
        accept: CudaFunction,
    }

    impl MtpAcceptKernel {
        /// Loads the kernel into `ctx`.
        pub fn load(ctx: &Arc<CudaContext>) -> Result<Self> {
            let module = load_module(ctx, KERNEL_FILE)?;
            Ok(Self {
                accept: module.kernel(KERNEL_MTP_ACCEPT)?,
            })
        }

        /// Accepts drafts for `rows` sequences. `draft` is `[rows, k]`,
        /// `target` is `[rows, k + 1]`, `draft_counts` holds the drafts
        /// proposed per row. With `sampled = Some((accept_prob, uniforms))`
        /// (both `[rows, k]`) draft `j` is accepted when
        /// `uniforms[j] < accept_prob[j]`; otherwise by exact match with
        /// `target`. `out_tokens` is `[rows, k + 1]` padded with
        /// [`super::NO_TOKEN`]; `out_counts` receives the emitted counts.
        #[allow(clippy::too_many_arguments)]
        pub fn accept(
            &self,
            stream: &Arc<CudaStream>,
            draft: &CudaSlice<u32>,
            target: &CudaSlice<u32>,
            draft_counts: &CudaSlice<i32>,
            sampled: Option<(&CudaSlice<f32>, &CudaSlice<f32>)>,
            rows: usize,
            k: usize,
            out_tokens: &mut CudaSlice<u32>,
            out_counts: &mut CudaSlice<u32>,
        ) -> Result<()> {
            if rows == 0 {
                return Ok(());
            }
            let (rows_arg, k_arg) = (rows as i32, k as i32);
            let null: u64 = 0;
            let mut launch = stream.launch_builder(&self.accept);
            launch.arg(draft).arg(target).arg(draft_counts);
            match sampled {
                Some((prob, uniforms)) => launch.arg(prob).arg(uniforms),
                None => launch.arg(&null).arg(&null),
            };
            // SAFETY: argument order and types match the kernel signature;
            // the caller supplies `[rows, k]` / `[rows, k + 1]` buffers and
            // `rows` entries of `draft_counts` / `out_counts`; the null
            // pointers select exact-match acceptance.
            unsafe {
                launch
                    .arg(&rows_arg)
                    .arg(&k_arg)
                    .arg(out_tokens)
                    .arg(out_counts)
                    .launch(to_config(accept_launch(rows)))?;
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::uniform;

    #[test]
    fn greedy_matches_reference_rule() {
        assert_eq!(accept_greedy(&[5, 6], &[5, 6, 7]).tokens, vec![5, 6, 7]);
        let o = accept_greedy(&[5, 9, 7], &[5, 6, 7, 8]);
        assert_eq!((o.tokens, o.accepted), (vec![5, 6], 1));
        assert_eq!(accept_greedy(&[], &[4]).tokens, vec![4]);
        assert_eq!(
            accept_greedy(&[5, 9], &[5, 6, 7]).tokens,
            crate::reference::scatter::mtp_accept(&[5, 9], &[5, 6, 7])
        );
    }

    #[test]
    fn zero_temperature_distribution_is_one_hot() {
        let d = target_distribution(&[0.1, 2.0, 1.0], 0.0, 0, 1.0);
        assert_eq!(d, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn distribution_sums_to_one_and_honours_top_k() {
        let logits = [0.0, 1.0, 2.0, 3.0, 4.0];
        let d = target_distribution(&logits, 1.0, 2, 1.0);
        assert_eq!(d[0], 0.0);
        assert_eq!(d[1], 0.0);
        assert_eq!(d[2], 0.0);
        assert!((d.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(d[4] > d[3]);
    }

    #[test]
    fn zero_temperature_rejection_equals_greedy() {
        let l0 = [0.0, 3.0, 1.0];
        let l1 = [2.0, 0.0, 1.0];
        let dists = vec![
            target_distribution(&l0, 0.0, 0, 1.0),
            target_distribution(&l1, 0.0, 0, 1.0),
        ];
        let hit = accept_rejection(&[1], &dists, &[0.999], &[0.5, 0.5]);
        assert_eq!(hit.tokens, vec![1, 0]);
        let miss = accept_rejection(&[2], &dists, &[0.0], &[0.5, 0.5]);
        assert_eq!(miss.tokens, vec![1]);
    }

    #[test]
    fn rejection_preserves_the_target_marginal() {
        let dist = vec![0.5, 0.3, 0.15, 0.05];
        let dists = vec![dist.clone(), dist.clone()];
        let draft = 0u32;
        let n = 40_000u32;
        let mut first = [0u32; 4];
        for i in 0..n {
            let a = [uniform(11, i)];
            let r = [uniform(22, i), uniform(33, i)];
            first[accept_rejection(&[draft], &dists, &a, &r).tokens[0] as usize] += 1;
        }
        for (t, &p) in dist.iter().enumerate() {
            let freq = first[t] as f32 / n as f32;
            assert!((freq - p).abs() < 0.01, "token {t}: {freq} vs {p}");
        }
    }

    #[test]
    fn residual_never_returns_the_rejected_token() {
        let dist = [0.6, 0.4];
        for i in 0..100 {
            assert_eq!(sample_residual(&dist, 0, uniform(5, i)), 1);
        }
        assert_eq!(sample_residual(&[1.0, 0.0], 0, 0.3), 0);
    }

    #[cfg(feature = "cuda")]
    mod gpu {
        use super::*;
        use crate::verify::MtpAcceptKernel;
        use scriptor_cuda::Device;

        #[test]
        #[ignore = "needs GPU"]
        fn device_accept_matches_reference() {
            if std::env::var("SCRIPTOR_GPU_TESTS").as_deref() != Ok("1") {
                return;
            }
            let dev = Device::open().expect("gpu expected");
            let s = dev.new_stream().expect("stream");
            let kernel = MtpAcceptKernel::load(dev.context()).expect("load");
            let (rows, k) = (5usize, 3usize);
            let draft: Vec<u32> = (0..rows * k).map(|i| (i % 4) as u32).collect();
            let target: Vec<u32> = (0..rows * (k + 1)).map(|i| ((i / 2) % 4) as u32).collect();
            let counts = vec![3i32, 3, 2, 0, 1];
            let d_draft = s.clone_htod(&draft).unwrap();
            let d_target = s.clone_htod(&target).unwrap();
            let d_counts = s.clone_htod(&counts).unwrap();
            let mut out = s.clone_htod(&vec![0u32; rows * (k + 1)]).unwrap();
            let mut n = s.clone_htod(&vec![0u32; rows]).unwrap();
            kernel
                .accept(
                    &s, &d_draft, &d_target, &d_counts, None, rows, k, &mut out, &mut n,
                )
                .unwrap();
            let got = s.clone_dtoh(&out).unwrap();
            let got_n = s.clone_dtoh(&n).unwrap();
            for r in 0..rows {
                let c = counts[r] as usize;
                let want = accept_greedy(
                    &draft[r * k..r * k + c],
                    &target[r * (k + 1)..r * (k + 1) + c + 1],
                );
                assert_eq!(got_n[r] as usize, want.tokens.len());
                assert_eq!(
                    &got[r * (k + 1)..r * (k + 1) + want.tokens.len()],
                    &want.tokens[..]
                );
            }
        }
    }
}
