//! CPU tests of transactional slot adoption against fake setup steps.

use scriptor_engine::{EngineError, RequestId};

use crate::slot_book::SlotBook;

const SLOTS: usize = 4;
/// Setup steps of an adoption: three mask copies and the draft copy.
const STEPS: usize = 4;

fn host_err() -> EngineError {
    EngineError::Executor("copy failed".into())
}

/// Runs the fake setup of `id`, recording each completed step; step `fail_at` errors.
fn adopt(
    book: &mut SlotBook,
    id: RequestId,
    fail_at: Option<usize>,
    done: &mut Vec<usize>,
) -> Result<usize, EngineError> {
    book.adopt(id, |_| {
        for step in 0..STEPS {
            if fail_at == Some(step) {
                return Err(host_err());
            }
            done.push(step);
        }
        Ok(())
    })
}

fn failed_step_leaks_nothing(step: usize, prior: u64) {
    let mut book = SlotBook::new(SLOTS);
    let mut done = Vec::new();
    for id in 0..prior {
        adopt(&mut book, id, None, &mut done).unwrap();
    }
    let owned = prior as usize;
    assert!(adopt(&mut book, 99, Some(step), &mut Vec::new()).is_err());
    assert!(book.is_partition(SLOTS));
    assert_eq!(book.counts(), [SLOTS - owned, owned, 0, 0]);
    assert_eq!(book.slot_of(99), None);
    assert_eq!(book.release(99), None);
    for id in 0..prior {
        assert!(book.release(id).is_some());
    }
    assert_eq!(book.counts(), [SLOTS, 0, 0, 0]);
    assert!(book.is_partition(SLOTS));
}

#[test]
fn first_mask_copy_failure_leaks_no_slot() {
    failed_step_leaks_nothing(0, 0);
    failed_step_leaks_nothing(0, 2);
}

#[test]
fn second_mask_copy_failure_leaks_no_slot() {
    failed_step_leaks_nothing(1, 0);
    failed_step_leaks_nothing(1, 2);
}

#[test]
fn count_mask_copy_failure_leaks_no_slot() {
    failed_step_leaks_nothing(2, 0);
    failed_step_leaks_nothing(2, 2);
}

#[test]
fn draft_copy_failure_leaks_no_slot() {
    failed_step_leaks_nothing(3, 0);
    failed_step_leaks_nothing(3, 2);
}

#[test]
fn batch_failure_releases_adopted_requests_and_frees_all_slots() {
    let mut book = SlotBook::new(SLOTS);
    let mut done = Vec::new();
    let ids = [10, 11, 12];
    let mut failed = false;
    for (n, &id) in ids.iter().enumerate() {
        let fail = (n == 2).then_some(3);
        failed = adopt(&mut book, id, fail, &mut done).is_err();
    }
    assert!(failed);
    assert_eq!(book.counts(), [SLOTS - 2, 2, 0, 0]);
    for id in ids {
        book.release(id);
    }
    assert_eq!(book.counts(), [SLOTS, 0, 0, 0]);
    assert!(book.is_partition(SLOTS));
}

#[test]
fn recoverable_failure_then_retry_succeeds_with_full_capacity() {
    let mut book = SlotBook::new(1);
    assert!(adopt(&mut book, 1, Some(1), &mut Vec::new()).is_err());
    let slot = adopt(&mut book, 1, None, &mut Vec::new()).unwrap();
    assert_eq!(slot, 0);
    assert_eq!(book.slot_of(1), Some(0));
    assert!(book.is_partition(1));
}

#[test]
fn fatal_failure_retires_the_slot() {
    let mut book = SlotBook::new(SLOTS);
    let err = book
        .adopt(1, |_| Err(EngineError::fatal(700, "illegal address")))
        .unwrap_err();
    assert!(err.is_fatal());
    assert_eq!(book.counts(), [SLOTS - 1, 0, 0, 1]);
    assert!(book.is_partition(SLOTS));
}

#[test]
fn exhausted_book_rejects_without_state_change() {
    let mut book = SlotBook::new(1);
    adopt(&mut book, 1, None, &mut Vec::new()).unwrap();
    assert!(adopt(&mut book, 2, None, &mut Vec::new()).is_err());
    assert_eq!(book.counts(), [0, 1, 0, 0]);
}

mod classification {
    use cudarc::driver::{DriverError, sys::CUresult};
    use scriptor_cuda::CudaError;
    use scriptor_model::gpu::GpuTextError;

    use super::*;
    use crate::gpu_executor::exec_err;

    const ILLEGAL_ADDRESS: i32 = 700;
    const CONTEXT_DESTROYED: i32 = 709;
    const OUT_OF_MEMORY: i32 = 2;

    fn raw(code: i32) -> DriverError {
        DriverError(match code {
            ILLEGAL_ADDRESS => CUresult::CUDA_ERROR_ILLEGAL_ADDRESS,
            CONTEXT_DESTROYED => CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED,
            OUT_OF_MEMORY => CUresult::CUDA_ERROR_OUT_OF_MEMORY,
            other => panic!("unmapped test code {other}"),
        })
    }

    /// Adopts one slot whose setup fails with `exec_err(error)`; returns the mapped error and
    /// the book's counts.
    fn adopt_failing<E: std::fmt::Display + 'static>(error: E) -> (EngineError, [usize; 4]) {
        let mut book = SlotBook::new(SLOTS);
        let err = book.adopt(1, |_| Err(exec_err(error))).unwrap_err();
        assert!(book.is_partition(SLOTS));
        (err, book.counts())
    }

    fn code_of(err: &EngineError) -> Option<i32> {
        match err {
            EngineError::Fatal { code, .. } => Some(*code),
            _ => None,
        }
    }

    fn assert_fatal(err: &EngineError, counts: [usize; 4], code: i32) {
        assert!(err.is_fatal(), "{code}: {err}");
        assert_eq!(code_of(err), Some(code));
        assert_eq!(counts, [SLOTS - 1, 0, 0, 1]);
    }

    fn assert_recoverable(err: &EngineError, counts: [usize; 4]) {
        assert!(!err.is_fatal(), "{err}");
        assert_eq!(counts, [SLOTS, 0, 0, 0]);
    }

    #[test]
    fn raw_illegal_address_retires_the_slot() {
        let (err, counts) = adopt_failing(raw(ILLEGAL_ADDRESS));
        assert_fatal(&err, counts, ILLEGAL_ADDRESS);
    }

    #[test]
    fn raw_context_destroyed_retires_the_slot() {
        let (err, counts) = adopt_failing(raw(CONTEXT_DESTROYED));
        assert_fatal(&err, counts, CONTEXT_DESTROYED);
    }

    #[test]
    fn raw_out_of_memory_frees_the_slot() {
        let (err, counts) = adopt_failing(raw(OUT_OF_MEMORY));
        assert_recoverable(&err, counts);
    }

    #[test]
    fn wrapped_forms_match_the_raw_classification() {
        for code in [ILLEGAL_ADDRESS, CONTEXT_DESTROYED, OUT_OF_MEMORY] {
            let (from_raw, raw_counts) = adopt_failing(raw(code));
            let (from_cuda, cuda_counts) = adopt_failing(CudaError::from(raw(code)));
            let (from_text, text_counts) =
                adopt_failing(GpuTextError::from(CudaError::from(raw(code))));
            let (from_text_raw, text_raw_counts) = adopt_failing(GpuTextError::from(raw(code)));
            assert_eq!(code_of(&from_raw), code_of(&from_cuda), "{code}");
            assert_eq!(code_of(&from_raw), code_of(&from_text), "{code}");
            assert_eq!(code_of(&from_raw), code_of(&from_text_raw), "{code}");
            assert_eq!(raw_counts, cuda_counts, "{code}");
            assert_eq!(raw_counts, text_counts, "{code}");
            assert_eq!(raw_counts, text_raw_counts, "{code}");
            assert_eq!(from_raw.is_fatal(), code != OUT_OF_MEMORY);
        }
    }
}
