//! Shutdown drain sequencing: wait for in-flight work until a deadline, then cancel it.

use std::future::Future;
use std::time::Duration;

use tokio::sync::mpsc::UnboundedReceiver;

/// How a drain ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainEnd {
    /// The server finished on its own within the deadline.
    Completed,
    /// The deadline passed, the work was cancelled, and the server then finished.
    DeadlineCancelled,
    /// A second signal cancelled the work and the server then finished.
    SignalCancelled,
    /// A further signal arrived after the cancellation; the server is abandoned.
    Abandoned,
}

/// Waits for `server` to finish. When `deadline` passes or a signal arrives on `signals`,
/// `cancel` runs once and the wait continues; one more signal then gives up.
pub async fn drain(
    server: &mut (impl Future + Unpin),
    signals: &mut UnboundedReceiver<()>,
    deadline: Duration,
    cancel: impl FnOnce(),
) -> DrainEnd {
    let cancelled_by = tokio::select! {
        _ = &mut *server => return DrainEnd::Completed,
        () = tokio::time::sleep(deadline) => DrainEnd::DeadlineCancelled,
        _ = signals.recv() => DrainEnd::SignalCancelled,
    };
    cancel();
    tokio::select! {
        _ = &mut *server => cancelled_by,
        _ = signals.recv() => DrainEnd::Abandoned,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use tokio::sync::mpsc::unbounded_channel;

    fn finishing_after(ms: u64) -> std::pin::Pin<Box<impl Future<Output = ()>>> {
        Box::pin(tokio::time::sleep(Duration::from_millis(ms)))
    }

    #[tokio::test]
    async fn work_finishing_before_the_deadline_is_not_cancelled() {
        let (_tx, mut rx) = unbounded_channel();
        let cancelled = Cell::new(false);
        let end = drain(
            &mut finishing_after(10),
            &mut rx,
            Duration::from_secs(30),
            || cancelled.set(true),
        )
        .await;
        assert_eq!(end, DrainEnd::Completed);
        assert!(!cancelled.get());
    }

    #[tokio::test]
    async fn the_deadline_cancels_then_waits_for_the_server() {
        let (_tx, mut rx) = unbounded_channel();
        let cancelled = Cell::new(false);
        let end = drain(
            &mut finishing_after(150),
            &mut rx,
            Duration::from_millis(20),
            || cancelled.set(true),
        )
        .await;
        assert_eq!(end, DrainEnd::DeadlineCancelled);
        assert!(cancelled.get());
    }

    #[tokio::test]
    async fn a_second_signal_cancels_at_once() {
        let (tx, mut rx) = unbounded_channel();
        tx.send(()).unwrap();
        let cancelled = Cell::new(false);
        let end = drain(
            &mut finishing_after(100),
            &mut rx,
            Duration::from_secs(30),
            || cancelled.set(true),
        )
        .await;
        assert_eq!(end, DrainEnd::SignalCancelled);
        assert!(cancelled.get());
    }

    #[tokio::test]
    async fn a_third_signal_abandons_a_stuck_server() {
        let (tx, mut rx) = unbounded_channel();
        tx.send(()).unwrap();
        tx.send(()).unwrap();
        let mut stuck = std::pin::pin!(std::future::pending::<()>());
        let end = drain(&mut stuck, &mut rx, Duration::from_secs(30), || {}).await;
        assert_eq!(end, DrainEnd::Abandoned);
    }

    #[tokio::test]
    async fn zero_deadline_cancels_immediately() {
        let (_tx, mut rx) = unbounded_channel();
        let cancelled = Cell::new(false);
        let end = drain(&mut finishing_after(30), &mut rx, Duration::ZERO, || {
            cancelled.set(true)
        })
        .await;
        assert_eq!(end, DrainEnd::DeadlineCancelled);
        assert!(cancelled.get());
    }
}
