use scriptor_cache::{EmbeddingCache, HostBuffer, ImageGrid, InsertOutcome, embedding_key};

const ROW: usize = 1536 * 2;

fn buf(rows: usize) -> HostBuffer {
    HostBuffer(vec![0u8; rows * ROW])
}

fn key(n: u8) -> scriptor_cache::EmbeddingKey {
    embedding_key(&[n; 12], ImageGrid { t: 1, h: 2, w: 2 })
}

#[test]
fn key_depends_on_pixels_and_grid() {
    let a = embedding_key(&[1, 2, 3, 4], ImageGrid { t: 1, h: 2, w: 2 });
    assert_eq!(
        a,
        embedding_key(&[1, 2, 3, 4], ImageGrid { t: 1, h: 2, w: 2 })
    );
    assert_ne!(
        a,
        embedding_key(&[1, 2, 3, 5], ImageGrid { t: 1, h: 2, w: 2 })
    );
    assert_ne!(
        a,
        embedding_key(&[1, 2, 3, 4], ImageGrid { t: 1, h: 4, w: 1 })
    );
    assert_eq!(a.to_hex().len(), 64);
}

#[test]
fn hit_miss_counted() {
    let mut c = EmbeddingCache::new(10 * ROW, 1536, 2);
    assert!(c.get(&key(1)).is_none());
    assert_eq!(
        c.insert(key(1), 3, buf(3)).unwrap(),
        InsertOutcome::Stored { evicted: 0 }
    );
    let e = c.get(&key(1)).unwrap();
    assert_eq!(e.rows, 3);
    let s = c.stats();
    assert_eq!((s.hits, s.misses, s.entries, s.bytes), (1, 1, 1, 3 * ROW));
}

#[test]
fn lru_eviction_respects_recency() {
    let mut c = EmbeddingCache::new(6 * ROW, 1536, 2);
    c.insert(key(1), 2, buf(2)).unwrap();
    c.insert(key(2), 2, buf(2)).unwrap();
    c.insert(key(3), 2, buf(2)).unwrap();
    assert!(c.get(&key(1)).is_some());
    assert_eq!(
        c.insert(key(4), 2, buf(2)).unwrap(),
        InsertOutcome::Stored { evicted: 1 }
    );
    assert!(c.contains(&key(1)) && !c.contains(&key(2)) && c.contains(&key(3)));
    assert_eq!(c.stats().evictions, 1);
    assert_eq!(
        c.insert(key(5), 5, buf(5)).unwrap(),
        InsertOutcome::Stored { evicted: 3 }
    );
    assert_eq!(c.stats().entries, 1);
}

#[test]
fn oversize_rejected_and_size_checked() {
    let mut c = EmbeddingCache::new(2 * ROW, 1536, 2);
    assert_eq!(
        c.insert(key(1), 3, buf(3)).unwrap(),
        InsertOutcome::Rejected
    );
    assert_eq!(c.stats().rejected, 1);
    assert!(c.insert(key(1), 2, buf(1)).is_err());
}

#[test]
fn evicted_entry_survives_while_held() {
    let mut c = EmbeddingCache::new(2 * ROW, 1536, 2);
    c.insert(key(1), 2, buf(2)).unwrap();
    let held = c.get(&key(1)).unwrap();
    c.insert(key(2), 2, buf(2)).unwrap();
    assert!(!c.contains(&key(1)));
    assert_eq!(held.rows, 2);
}

#[test]
fn reinsert_replaces_without_leaking_bytes() {
    let mut c = EmbeddingCache::new(10 * ROW, 1536, 2);
    c.insert(key(1), 4, buf(4)).unwrap();
    c.insert(key(1), 2, buf(2)).unwrap();
    assert_eq!(c.stats().bytes, 2 * ROW);
    assert!(c.remove(&key(1)));
    assert_eq!(c.stats().bytes, 0);
}

#[test]
fn targeted_eviction_skips_kept_entries_and_stops_at_the_target() {
    let mut c = EmbeddingCache::new(8 * ROW, 1536, 2);
    for n in 1..=4 {
        c.insert(key(n), 2, buf(2)).unwrap();
    }
    let order: Vec<_> = c.iter_lru().map(|(k, _)| *k).collect();
    assert_eq!(order, vec![key(1), key(2), key(3), key(4)]);
    let kept = key(1);
    assert_eq!(c.evict_lru_unless(|k, _| *k == kept, 4 * ROW), 2);
    assert!(c.contains(&key(1)) && !c.contains(&key(2)) && !c.contains(&key(3)));
    assert!(c.contains(&key(4)));
    assert_eq!(c.stats().evictions, 2);
    assert_eq!(c.evict_lru_unless(|k, _| *k == kept, 0), 1);
    assert_eq!(c.stats().entries, 1);
}
