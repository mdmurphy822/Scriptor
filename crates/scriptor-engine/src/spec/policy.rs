//! Adaptive draft length: online cost models of the verification and draft
//! passes, running per-position acceptance, and the choice of the draft length
//! that maximises expected tokens per second at the current batch.

/// Tunables of the draft-length policy.
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyConfig {
    /// Largest draft length considered; the default leaves the caller's per-step limit as the
    /// only bound, so the measured throughput model alone picks the length.
    pub k_max: usize,
    /// Per-step multiplier applied to the accumulated cost and acceptance
    /// observations; values below one weight recent steps more.
    pub decay: f64,
    /// Cumulative acceptance assumed for position `j` before observations:
    /// `prior_acceptance^(j + 1)`.
    pub prior_acceptance: f64,
    /// Observation weight of the prior.
    pub prior_weight: f64,
    /// Draft pass cost as a fraction of the verification pass cost at the same
    /// batch, used until a draft pass has been measured.
    pub initial_draft_ratio: f64,
    /// Relative throughput gain over `k = 0` a draft length must show.
    pub min_gain: f64,
    /// Steps spent at `k = 0` before a one-draft probe re-measures acceptance.
    pub reprobe_after: u64,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            k_max: usize::MAX,
            decay: 0.995,
            prior_acceptance: 0.6,
            prior_weight: 8.0,
            initial_draft_ratio: 0.25,
            min_gain: 0.02,
            reprobe_after: 64,
        }
    }
}

/// Decayed least-squares fit of `seconds = base + per_row * rows`.
#[derive(Debug, Clone, Default)]
pub struct AffineCost {
    n: f64,
    sx: f64,
    sy: f64,
    sxx: f64,
    sxy: f64,
}

impl AffineCost {
    /// Records one measurement after scaling earlier ones by `decay`.
    pub fn observe(&mut self, rows: usize, seconds: f64, decay: f64) {
        let x = rows as f64;
        self.n = self.n * decay + 1.0;
        self.sx = self.sx * decay + x;
        self.sy = self.sy * decay + seconds;
        self.sxx = self.sxx * decay + x * x;
        self.sxy = self.sxy * decay + x * seconds;
    }

    /// Whether any measurement has been recorded.
    pub fn is_empty(&self) -> bool {
        self.n <= 0.0
    }

    /// Predicted seconds for `rows`; the fit's intercept and slope are
    /// clamped non-negative, and a single observed row count predicts the
    /// mean measured time. `None` before the first measurement.
    pub fn predict(&self, rows: usize) -> Option<f64> {
        if self.is_empty() {
            return None;
        }
        let mean_x = self.sx / self.n;
        let mean_y = self.sy / self.n;
        let var_x = self.sxx / self.n - mean_x * mean_x;
        if var_x <= 1e-9 * mean_x.max(1.0).powi(2) {
            return Some(mean_y);
        }
        let slope = ((self.sxy / self.n - mean_x * mean_y) / var_x).max(0.0);
        let base = (mean_y - slope * mean_x).max(0.0);
        Some(base + slope * rows as f64)
    }
}

/// Decayed per-position draft acceptance. Position `j` counts steps whose
/// draft chain reached `j` and how often drafts `0..=j` were all accepted.
#[derive(Debug, Clone, Default)]
pub struct PositionAcceptance {
    proposed: Vec<f64>,
    accepted: Vec<f64>,
}

impl PositionAcceptance {
    /// Records one sequence-step that proposed `proposed` drafts of which the
    /// first `accepted` were accepted.
    pub fn observe(&mut self, proposed: usize, accepted: usize) {
        if self.proposed.len() < proposed {
            self.proposed.resize(proposed, 0.0);
            self.accepted.resize(proposed, 0.0);
        }
        for j in 0..proposed {
            self.proposed[j] += 1.0;
            if j < accepted {
                self.accepted[j] += 1.0;
            }
        }
    }

    /// Applies one step of decay to every counter.
    pub fn decay(&mut self, decay: f64) {
        for v in self.proposed.iter_mut().chain(&mut self.accepted) {
            *v *= decay;
        }
    }

    /// Cumulative acceptance estimate for positions `0..k`: the probability
    /// that drafts `0..=j` are all accepted, shrunk toward the prior and made
    /// non-increasing in `j`.
    pub fn cumulative(&self, k: usize, cfg: &PolicyConfig) -> Vec<f64> {
        let mut out = Vec::with_capacity(k);
        let mut ceiling = 1.0f64;
        for j in 0..k {
            let prior = cfg.prior_acceptance.powi(j as i32 + 1);
            let seen = self.proposed.get(j).copied().unwrap_or(0.0);
            let hit = self.accepted.get(j).copied().unwrap_or(0.0);
            let est = (hit + prior * cfg.prior_weight) / (seen + cfg.prior_weight);
            ceiling = ceiling.min(est);
            out.push(ceiling);
        }
        out
    }

    /// Expected tokens per sequence-step at draft length `k`.
    pub fn expected_tokens(&self, k: usize, cfg: &PolicyConfig) -> f64 {
        1.0 + self.cumulative(k, cfg).iter().sum::<f64>()
    }
}

/// Chooses the draft length of each step from measured costs and acceptance.
#[derive(Debug, Clone)]
pub struct DraftPolicy {
    cfg: PolicyConfig,
    main: AffineCost,
    draft: AffineCost,
    acceptance: PositionAcceptance,
    steps_at_zero: u64,
}

impl DraftPolicy {
    /// A policy with no observations.
    pub fn new(cfg: PolicyConfig) -> Self {
        Self {
            cfg,
            main: AffineCost::default(),
            draft: AffineCost::default(),
            acceptance: PositionAcceptance::default(),
            steps_at_zero: 0,
        }
    }

    /// The tunables.
    pub fn config(&self) -> &PolicyConfig {
        &self.cfg
    }

    /// Acceptance observations.
    pub fn acceptance(&self) -> &PositionAcceptance {
        &self.acceptance
    }

    /// Records a verification pass over `rows` total rows.
    pub fn observe_main(&mut self, rows: usize, seconds: f64) {
        self.main.observe(rows, seconds, self.cfg.decay);
    }

    /// Records one draft pass over `batch` sequences.
    pub fn observe_draft(&mut self, batch: usize, seconds: f64) {
        self.draft.observe(batch, seconds, self.cfg.decay);
    }

    /// Records the outcome of one sequence-step: `proposed` drafts verified,
    /// `accepted` of them accepted.
    pub fn observe_acceptance(&mut self, proposed: usize, accepted: usize) {
        self.acceptance.observe(proposed, accepted);
    }

    /// Ends a step: decays acceptance observations.
    pub fn end_step(&mut self) {
        self.acceptance.decay(self.cfg.decay);
    }

    /// Estimated seconds of a step that verifies `batch * (k + 1)` rows and
    /// runs `k` draft passes. `None` until a verification pass was measured.
    pub fn step_cost(&self, batch: usize, k: usize) -> Option<f64> {
        let main = self.main.predict(batch * (k + 1))?;
        let per_draft = self.draft.predict(batch).unwrap_or_else(|| {
            let one = self.main.predict(batch).unwrap_or(main);
            one * self.cfg.initial_draft_ratio
        });
        Some(main + per_draft * k as f64)
    }

    /// Expected tokens per second across the batch at draft length `k`.
    pub fn throughput(&self, batch: usize, k: usize) -> Option<f64> {
        let cost = self.step_cost(batch, k)?;
        if cost <= 0.0 {
            return None;
        }
        Some(batch as f64 * self.acceptance.expected_tokens(k, &self.cfg) / cost)
    }

    /// Draft length for the next step at `batch` sequences, at most `limit`.
    /// Returns zero when no positive length beats `k = 0` by `min_gain`;
    /// after `reprobe_after` such steps a single draft is tried again so the
    /// acceptance estimate keeps tracking. Before any cost measurement the
    /// largest permitted length is used to gather them.
    pub fn choose_k(&mut self, batch: usize, limit: usize) -> usize {
        let cap = limit.min(self.cfg.k_max);
        if cap == 0 || batch == 0 {
            return 0;
        }
        let Some(base) = self.throughput(batch, 0) else {
            self.steps_at_zero = 0;
            return cap;
        };
        let mut best = (0usize, base);
        for k in 1..=cap {
            if let Some(t) = self.throughput(batch, k)
                && t > best.1.max(base * (1.0 + self.cfg.min_gain))
            {
                best = (k, t);
            }
        }
        if best.0 == 0 {
            self.steps_at_zero += 1;
            if self.steps_at_zero >= self.cfg.reprobe_after {
                self.steps_at_zero = 0;
                return 1;
            }
            return 0;
        }
        self.steps_at_zero = 0;
        best.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fed(main_per_row: f64, main_base: f64, draft_per_batch: f64) -> DraftPolicy {
        let mut p = DraftPolicy::new(PolicyConfig::default());
        for rows in [8usize, 16, 32, 64, 8, 16, 32, 64] {
            p.observe_main(rows, main_base + main_per_row * rows as f64);
            p.observe_draft(rows, draft_per_batch * rows as f64);
        }
        p
    }

    #[test]
    fn affine_fit_recovers_line() {
        let mut c = AffineCost::default();
        for rows in [1usize, 5, 9, 20] {
            c.observe(rows, 2.0 + 0.5 * rows as f64, 1.0);
        }
        assert!((c.predict(100).unwrap() - 52.0).abs() < 1e-6);
    }

    #[test]
    fn single_row_count_predicts_mean() {
        let mut c = AffineCost::default();
        c.observe(8, 0.01, 1.0);
        c.observe(8, 0.03, 1.0);
        assert!((c.predict(40).unwrap() - 0.02).abs() < 1e-12);
    }

    #[test]
    fn explores_with_largest_k_before_measurements() {
        let mut p = DraftPolicy::new(PolicyConfig::default());
        assert_eq!(p.choose_k(16, 3), 3);
        assert_eq!(p.choose_k(16, 1), 1);
        assert_eq!(p.choose_k(16, 0), 0);
    }

    #[test]
    fn expensive_drafts_choose_zero() {
        let mut p = fed(0.0, 0.01, 0.01);
        for _ in 0..200 {
            p.observe_acceptance(3, 1);
        }
        assert_eq!(p.choose_k(32, 3), 0);
    }

    #[test]
    fn cheap_accurate_drafts_choose_positive_k() {
        let mut p = fed(0.0, 0.01, 0.0001);
        for _ in 0..200 {
            p.observe_acceptance(3, 3);
        }
        assert_eq!(p.choose_k(32, 3), 3);
    }

    #[test]
    fn default_config_is_bounded_only_by_the_caller_limit() {
        let mut p = fed(0.0, 0.01, 0.0001);
        for _ in 0..200 {
            p.observe_acceptance(7, 7);
        }
        assert_eq!(p.choose_k(32, 5), 5);
        assert_eq!(p.choose_k(32, 7), 7);
    }

    #[test]
    fn compute_bound_verification_discourages_drafting() {
        // Verification cost grows linearly with rows: k extra rows cost k*b rows.
        let mut p = fed(0.001, 0.0, 0.0002);
        for _ in 0..200 {
            p.observe_acceptance(3, 1);
        }
        assert_eq!(p.choose_k(64, 3), 0);
    }

    #[test]
    fn disabled_policy_reprobes_one_draft() {
        let mut p = fed(0.0, 0.01, 0.01);
        for _ in 0..200 {
            p.observe_acceptance(3, 0);
        }
        let reprobe = p.config().reprobe_after;
        let ks: Vec<usize> = (0..reprobe).map(|_| p.choose_k(32, 3)).collect();
        assert!(ks[..ks.len() - 1].iter().all(|&k| k == 0));
        assert_eq!(*ks.last().unwrap(), 1);
    }

    #[test]
    fn cumulative_acceptance_is_non_increasing() {
        let mut a = PositionAcceptance::default();
        a.observe(3, 3);
        a.observe(3, 0);
        let cfg = PolicyConfig::default();
        let c = a.cumulative(5, &cfg);
        assert!(c.windows(2).all(|w| w[0] >= w[1]));
        assert!(a.expected_tokens(3, &cfg) > 1.0);
    }
}
