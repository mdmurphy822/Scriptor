//! Panic containment for worker threads.

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;
use std::sync::mpsc::Receiver;

/// Text of a caught panic payload.
pub fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_owned()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-text panic payload".to_owned()
    }
}

/// Runs jobs received on `jobs` against a worker state made by `build` until the channel
/// disconnects. A job whose `run` panics is dropped, so any reply handle it owns disconnects,
/// and the state is rebuilt with `build` (the previous state is kept when the rebuild fails),
/// so one bad request never shrinks the worker pool. Returns the number of panics survived, or
/// `None` when the initial state cannot be built.
pub fn serve_jobs<S, J>(
    jobs: &Mutex<Receiver<J>>,
    build: impl Fn() -> Option<S>,
    run: impl Fn(&S, J),
) -> Option<u64> {
    let mut state = build()?;
    let mut panics = 0;
    loop {
        let job = jobs.lock().unwrap_or_else(|p| p.into_inner()).recv();
        let Ok(job) = job else { return Some(panics) };
        if catch_unwind(AssertUnwindSafe(|| run(&state, job))).is_err() {
            panics += 1;
            if let Some(fresh) = build() {
                state = fresh;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc::{Sender, channel};

    #[test]
    fn a_panicking_job_does_not_stop_the_worker() {
        let (tx, rx) = channel::<(u32, Sender<u32>)>();
        let rx = Mutex::new(rx);
        let builds = AtomicUsize::new(0);
        let worker = std::thread::scope(|scope| {
            let handle = scope.spawn(|| {
                serve_jobs(
                    &rx,
                    || {
                        builds.fetch_add(1, Ordering::SeqCst);
                        Some(10u32)
                    },
                    |base, (n, reply)| {
                        assert!(n != 0, "injected panic");
                        let _ = reply.send(base + n);
                    },
                )
            });
            let (r1, a1) = channel();
            tx.send((0, r1)).unwrap();
            assert!(a1.recv().is_err(), "the panicked job's reply disconnects");
            let (r2, a2) = channel();
            tx.send((5, r2)).unwrap();
            assert_eq!(a2.recv().unwrap(), 15);
            drop(tx);
            handle.join().unwrap()
        });
        assert_eq!(worker, Some(1));
        assert_eq!(
            builds.load(Ordering::SeqCst),
            2,
            "state rebuilt after the panic"
        );
    }

    #[test]
    fn failed_rebuild_keeps_the_previous_state() {
        let (tx, rx) = channel::<(u32, Sender<u32>)>();
        let rx = Mutex::new(rx);
        let builds = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            let handle = scope.spawn(|| {
                serve_jobs(
                    &rx,
                    || (builds.fetch_add(1, Ordering::SeqCst) == 0).then_some(7u32),
                    |base, (n, reply)| {
                        assert!(n != 0, "injected panic");
                        let _ = reply.send(base + n);
                    },
                )
            });
            let (r1, _a1) = channel();
            tx.send((0, r1)).unwrap();
            let (r2, a2) = channel();
            tx.send((1, r2)).unwrap();
            assert_eq!(a2.recv().unwrap(), 8);
            drop(tx);
            assert_eq!(handle.join().unwrap(), Some(1));
        });
    }

    #[test]
    fn unbuildable_state_ends_the_worker() {
        let (_tx, rx) = channel::<u32>();
        let rx = Mutex::new(rx);
        assert_eq!(serve_jobs(&rx, || None::<u8>, |_, _| {}), None);
    }

    #[test]
    fn panic_text_is_recovered() {
        let p = catch_unwind(|| panic!("boom {}", 1)).unwrap_err();
        assert_eq!(panic_message(p.as_ref()), "boom 1");
        let p = catch_unwind(|| panic!("static")).unwrap_err();
        assert_eq!(panic_message(p.as_ref()), "static");
    }
}
