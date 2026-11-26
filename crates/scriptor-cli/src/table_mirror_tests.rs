//! CPU tests of the block-table mirror against a recording fake uploader.

use scriptor_engine::EngineError;

use crate::table_mirror::TableMirror;

/// Records each uploaded payload; fails the next upload when `fail` is set.
#[derive(Default)]
struct Fake {
    payloads: Vec<Vec<i32>>,
    fail: bool,
}

impl Fake {
    fn sync(
        &mut self,
        mirror: &mut TableMirror,
        slot: usize,
        blocks: &[u32],
    ) -> Result<(), EngineError> {
        let (payloads, fail) = (&mut self.payloads, self.fail);
        mirror.sync(slot, blocks, |t| {
            if fail {
                return Err(EngineError::Executor("copy failed".into()));
            }
            payloads.push(t.to_vec());
            Ok(())
        })
    }
}

#[test]
fn release_adopt_then_block_zero_uploads_once() {
    let mut m = TableMirror::new(2, 4);
    let mut f = Fake::default();
    f.sync(&mut m, 0, &[7, 9]).unwrap();
    m.invalidate(0);
    f.sync(&mut m, 0, &[0]).unwrap();
    assert_eq!(f.payloads, vec![vec![7, 9, 0, 0], vec![0, 0, 0, 0]]);
}

#[test]
fn same_table_reused_by_new_owner_uploads_once() {
    let mut m = TableMirror::new(1, 4);
    let mut f = Fake::default();
    f.sync(&mut m, 0, &[7, 9]).unwrap();
    m.invalidate(0);
    f.sync(&mut m, 0, &[7, 9]).unwrap();
    assert_eq!(f.payloads.len(), 2);
    assert_eq!(f.payloads[1], vec![7, 9, 0, 0]);
}

#[test]
fn unchanged_table_within_owner_skips() {
    let mut m = TableMirror::new(1, 4);
    let mut f = Fake::default();
    f.sync(&mut m, 0, &[3, 4]).unwrap();
    f.sync(&mut m, 0, &[3, 4]).unwrap();
    f.sync(&mut m, 0, &[3, 4]).unwrap();
    assert_eq!(f.payloads, vec![vec![3, 4, 0, 0]]);
}

#[test]
fn shrink_and_grow_upload_padded_tables() {
    let mut m = TableMirror::new(1, 4);
    let mut f = Fake::default();
    f.sync(&mut m, 0, &[1, 2, 3]).unwrap();
    f.sync(&mut m, 0, &[1]).unwrap();
    f.sync(&mut m, 0, &[1, 2]).unwrap();
    assert_eq!(
        f.payloads,
        vec![vec![1, 2, 3, 0], vec![1, 0, 0, 0], vec![1, 2, 0, 0]]
    );
}

#[test]
fn failed_copy_then_retry_uploads() {
    let mut m = TableMirror::new(1, 4);
    let mut f = Fake::default();
    f.sync(&mut m, 0, &[5]).unwrap();
    f.fail = true;
    assert!(f.sync(&mut m, 0, &[6]).is_err());
    f.fail = false;
    f.sync(&mut m, 0, &[6]).unwrap();
    assert_eq!(f.payloads, vec![vec![5, 0, 0, 0], vec![6, 0, 0, 0]]);
    f.sync(&mut m, 0, &[6]).unwrap();
    assert_eq!(f.payloads.len(), 2);
}

#[test]
fn failed_first_upload_of_old_table_retries_even_if_equal() {
    let mut m = TableMirror::new(1, 4);
    let mut f = Fake::default();
    f.sync(&mut m, 0, &[0]).unwrap();
    m.invalidate(0);
    f.fail = true;
    assert!(f.sync(&mut m, 0, &[0]).is_err());
    f.fail = false;
    f.sync(&mut m, 0, &[0]).unwrap();
    assert_eq!(f.payloads.len(), 2);
}

#[test]
fn oversized_table_is_rejected_without_upload() {
    let mut m = TableMirror::new(1, 2);
    let mut f = Fake::default();
    assert!(f.sync(&mut m, 0, &[1, 2, 3]).is_err());
    assert!(f.payloads.is_empty());
}
