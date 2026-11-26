use scriptor_cache::{BlockPool, CacheError, KvLayout, ReserveOutcome};
use std::collections::HashSet;

fn layout(blocks: usize) -> KvLayout {
    KvLayout {
        num_blocks: blocks,
        block_tokens: 16,
        kv_bytes_per_block: 16 * 65536,
        mtp_kv_bytes_per_block: 16 * 4096,
    }
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

fn check_accounting(pool: &BlockPool, ids: &[u64]) {
    let mut seen = HashSet::new();
    let mut used = 0;
    for &id in ids {
        if !pool.contains(id) {
            continue;
        }
        for &b in pool.block_table(id).unwrap() {
            assert!(seen.insert(b), "block {b} owned twice");
            assert!((b as usize) < pool.layout().num_blocks);
            used += 1;
        }
    }
    let s = pool.stats();
    assert_eq!(s.used_blocks, used);
    assert_eq!(s.free_blocks + used, s.total_blocks);
    assert_eq!(s.allocations - s.releases, used as u64);
}

#[test]
fn layout_bytes_and_blocks() {
    let l = layout(10);
    assert_eq!(l.bytes_per_block(), 16 * (65536 + 4096));
    assert_eq!(l.blocks_for_tokens(1), 1);
    assert_eq!(l.blocks_for_tokens(16), 1);
    assert_eq!(l.blocks_for_tokens(17), 2);
}

#[test]
fn grow_is_on_demand_and_all_or_nothing() {
    let mut p = BlockPool::new(layout(4));
    p.admit(1).unwrap();
    assert_eq!(p.grow_to(1, 17).unwrap().len(), 2);
    assert_eq!(p.grow_to(1, 20).unwrap().len(), 0);
    assert_eq!(p.grow_to(1, 33).unwrap().len(), 1);
    assert_eq!(
        p.grow_to(1, 100),
        Err(CacheError::OutOfBlocks { needed: 4, free: 1 })
    );
    assert_eq!(p.block_table(1).unwrap().len(), 3);
    assert_eq!(p.release(1).unwrap(), 3);
    assert_eq!(p.free_blocks(), 4);
    assert_eq!(p.admit(2), Ok(()));
    assert_eq!(p.admit(2), Err(CacheError::DuplicateSequence(2)));
}

#[test]
fn preemption_frees_only_the_youngest() {
    let mut p = BlockPool::new(layout(6));
    for id in 1..=3 {
        p.admit(id).unwrap();
        p.grow_to(id, 32).unwrap();
    }
    let old_tables: Vec<Vec<u32>> = (1..=2)
        .map(|i| p.block_table(i).unwrap().to_vec())
        .collect();
    assert_eq!(p.free_blocks(), 0);
    match p.reserve(1, 48).unwrap() {
        ReserveOutcome::Granted(r) => {
            assert_eq!(r.preempted, vec![3]);
            assert_eq!(r.new_blocks.len(), 1);
        }
        other => panic!("{other:?}"),
    }
    assert!(!p.contains(3));
    assert!(p.contains(2));
    assert_eq!(p.block_table(2).unwrap(), old_tables[1].as_slice());
    assert_eq!(&p.block_table(1).unwrap()[..2], old_tables[0].as_slice());
    assert_eq!(p.stats().preemptions, 1);
}

#[test]
fn youngest_requester_preempts_itself() {
    let mut p = BlockPool::new(layout(4));
    p.admit(1).unwrap();
    p.grow_to(1, 32).unwrap();
    p.admit(2).unwrap();
    p.grow_to(2, 32).unwrap();
    match p.reserve(2, 64).unwrap() {
        ReserveOutcome::SelfPreempted { preempted } => assert_eq!(preempted, vec![2]),
        other => panic!("{other:?}"),
    }
    assert!(p.contains(1) && !p.contains(2));
    assert_eq!(p.block_table(1).unwrap().len(), 2);
    p.admit(2).unwrap();
    assert_eq!(p.youngest(), Some(2));
}

#[test]
fn accounting_exact_under_random_admission_and_finish() {
    for seed in 0..32u64 {
        let mut rng = Lcg(seed * 7919 + 1);
        let mut p = BlockPool::new(layout(24));
        let ids: Vec<u64> = (0..12).collect();
        for _ in 0..2000 {
            let id = rng.next() % 12;
            match rng.next() % 4 {
                0 => {
                    let _ = p.admit(id);
                }
                1 | 2 => {
                    if p.contains(id) {
                        let tokens = p.tokens(id).unwrap() + (rng.next() % 40) as usize;
                        let before = p.sequences_by_age();
                        match p.reserve(id, tokens).unwrap() {
                            ReserveOutcome::Granted(r) => {
                                let pos = before.iter().position(|s| *s == id).unwrap();
                                for v in &r.preempted {
                                    assert!(before.iter().position(|s| s == v).unwrap() > pos);
                                }
                            }
                            ReserveOutcome::SelfPreempted { preempted } => {
                                assert_eq!(preempted.last(), Some(&id));
                            }
                        }
                    }
                }
                _ => {
                    let _ = p.release(id);
                }
            }
            check_accounting(&p, &ids);
        }
        for id in &ids {
            let _ = p.release(*id);
        }
        assert_eq!(p.free_blocks(), 24);
    }
}

fn runs_in(table: &[u32]) -> usize {
    1 + table.windows(2).filter(|w| w[1] != w[0] + 1).count()
}

#[test]
fn fresh_pool_is_one_run_and_tables_are_adjacent() {
    let mut p = BlockPool::new(layout(64));
    assert_eq!(p.free_runs(), 1);
    assert_eq!(p.largest_free_run(), 64);
    p.admit(1).unwrap();
    p.grow_to(1, 16 * 5).unwrap();
    assert_eq!(p.block_table(1).unwrap(), &[0, 1, 2, 3, 4]);
    assert_eq!(p.free_runs(), 1);
}

#[test]
fn growth_extends_the_run_after_the_last_block() {
    let mut p = BlockPool::new(layout(64));
    p.admit(1).unwrap();
    for t in 1..=10 {
        p.grow_to(1, t * 16).unwrap();
    }
    assert_eq!(runs_in(p.block_table(1).unwrap()), 1);
}

#[test]
fn release_coalesces_neighbours() {
    let mut p = BlockPool::new(layout(12));
    for id in 1..=3 {
        p.admit(id).unwrap();
        p.grow_to(id, 16 * 4).unwrap();
    }
    assert_eq!(p.free_runs(), 0);
    p.release(1).unwrap();
    p.release(3).unwrap();
    assert_eq!(p.free_runs(), 2);
    p.release(2).unwrap();
    assert_eq!(p.free_runs(), 1);
    assert_eq!(p.largest_free_run(), 12);
}

#[test]
fn request_prefers_a_single_fitting_run() {
    let mut p = BlockPool::new(layout(16));
    for id in 1..=4 {
        p.admit(id).unwrap();
        p.grow_to(id, 16 * 4).unwrap();
    }
    p.release(1).unwrap();
    p.release(3).unwrap();
    p.admit(9).unwrap();
    p.grow_to(9, 16 * 3).unwrap();
    assert_eq!(runs_in(p.block_table(9).unwrap()), 1);
    p.admit(10).unwrap();
    p.grow_to(10, 16 * 5).unwrap();
    assert_eq!(p.free_blocks(), 0);
}

#[test]
fn exhaustion_and_capacity_match_block_count() {
    let mut p = BlockPool::new(layout(10));
    p.admit(1).unwrap();
    p.admit(2).unwrap();
    p.grow_to(1, 16 * 3).unwrap();
    p.grow_to(2, 16 * 3).unwrap();
    p.grow_to(1, 16 * 6).unwrap();
    assert_eq!(
        p.grow_to(2, 16 * 9),
        Err(CacheError::OutOfBlocks { needed: 6, free: 1 })
    );
    p.grow_to(2, 16 * 4).unwrap();
    assert_eq!(p.free_blocks(), 0);
    let used = p.block_table(1).unwrap().len() + p.block_table(2).unwrap().len();
    assert_eq!(used, 10);
}

#[test]
fn mixed_churn_keeps_accounting_and_limits_fragmentation() {
    let mut p = BlockPool::new(layout(512));
    let mut rng = Lcg(7);
    let ids: Vec<u64> = (1..=24).collect();
    let mut tokens = vec![0usize; ids.len()];
    for step in 0..4000 {
        let i = (rng.next() as usize) % ids.len();
        let id = ids[i];
        if !p.contains(id) {
            p.admit(id).unwrap();
            tokens[i] = 0;
        }
        if rng.next().is_multiple_of(5) {
            p.release(id).unwrap();
        } else {
            tokens[i] += 1 + (rng.next() as usize) % 40;
            let _ = p.grow_to(id, tokens[i]);
        }
        if step % 100 == 0 {
            check_accounting(&p, &ids);
        }
    }
    check_accounting(&p, &ids);
    for &id in &ids {
        if p.contains(id) {
            p.release(id).unwrap();
        }
    }
    assert_eq!(p.free_runs(), 1);
    assert_eq!(p.largest_free_run(), 512);
}

#[test]
fn explicit_keys_order_age_and_eviction() {
    let mut p = BlockPool::new(layout(6));
    p.admit_with_key(10, 5).unwrap();
    p.admit_with_key(11, 2).unwrap();
    p.admit_with_key(12, 9).unwrap();
    for id in [10, 11, 12] {
        p.grow_to(id, 32).unwrap();
    }
    assert_eq!(p.sequences_by_age(), vec![11, 10, 12]);
    assert_eq!(p.youngest(), Some(12));
    match p.reserve(11, 48).unwrap() {
        ReserveOutcome::Granted(r) => assert_eq!(r.preempted, vec![12]),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_readmitted_sequence_with_its_key_is_evicted_before_lower_keys() {
    let mut p = BlockPool::new(layout(6));
    p.admit_with_key(1, 0).unwrap();
    p.admit_with_key(2, 1).unwrap();
    p.admit_with_key(3, 2).unwrap();
    for id in [1, 2, 3] {
        p.grow_to(id, 32).unwrap();
    }
    p.release(1).unwrap();
    p.admit_with_key(1, 0).unwrap();
    p.grow_to(1, 32).unwrap();
    assert_eq!(p.sequences_by_age(), vec![1, 2, 3]);
    assert_eq!(p.youngest(), Some(3));
    match p.reserve(1, 48).unwrap() {
        ReserveOutcome::Granted(r) => assert_eq!(r.preempted, vec![3]),
        other => panic!("{other:?}"),
    }
}

#[test]
fn keyless_admission_is_younger_than_every_given_key() {
    let mut p = BlockPool::new(layout(4));
    p.admit_with_key(1, 40).unwrap();
    p.admit(2).unwrap();
    assert_eq!(p.sequences_by_age(), vec![1, 2]);
}

#[test]
fn age_listing_into_reused_buffers_matches_the_allocating_listing() {
    let mut p = BlockPool::new(layout(8));
    let (mut keyed, mut out) = (Vec::new(), vec![99, 98, 97, 96]);
    p.sequences_by_age_into(&mut keyed, &mut out);
    assert!(out.is_empty(), "stale contents survive an empty pool");
    p.admit_with_key(10, 5).unwrap();
    p.admit_with_key(11, 2).unwrap();
    p.admit(12).unwrap();
    p.sequences_by_age_into(&mut keyed, &mut out);
    assert_eq!(out, p.sequences_by_age());
    assert_eq!(out, vec![11, 10, 12]);
    p.release(11).unwrap();
    p.sequences_by_age_into(&mut keyed, &mut out);
    assert_eq!(out, vec![10, 12]);
}
