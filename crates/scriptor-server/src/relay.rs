//! Waiting on an engine event stream while watching for the consumer going away.

use tokio::sync::mpsc;

/// What a relay task observed.
#[derive(Debug, PartialEq, Eq)]
pub enum Next<T> {
    /// The engine produced an event.
    Event(T),
    /// The engine side closed its stream.
    Ended,
    /// The consumer dropped its receiver, so the request should be cancelled at once.
    ConsumerGone,
}

/// Waits for the next engine event or for `consumer`'s receiver to be dropped, whichever
/// comes first, so a client disconnect is noticed without waiting for another token.
pub async fn next_event<T, U>(
    events: &mut mpsc::UnboundedReceiver<T>,
    consumer: &mpsc::Sender<U>,
) -> Next<T> {
    tokio::select! {
        biased;
        () = consumer.closed() => Next::ConsumerGone,
        event = events.recv() => match event {
            Some(e) => Next::Event(e),
            None => Next::Ended,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn a_dropped_consumer_is_noticed_while_the_engine_is_silent() {
        let (_engine_tx, mut events) = mpsc::unbounded_channel::<u32>();
        let (tx, rx) = mpsc::channel::<u32>(1);
        let waiter = tokio::spawn(async move { next_event(&mut events, &tx).await });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(
            !waiter.is_finished(),
            "silent engine and live consumer keep waiting"
        );
        drop(rx);
        assert_eq!(waiter.await.unwrap(), Next::ConsumerGone);
    }

    #[tokio::test]
    async fn events_and_stream_end_are_reported() {
        let (engine_tx, mut events) = mpsc::unbounded_channel::<u32>();
        let (tx, _rx) = mpsc::channel::<u32>(1);
        engine_tx.send(7).unwrap();
        assert_eq!(next_event(&mut events, &tx).await, Next::Event(7));
        drop(engine_tx);
        assert_eq!(next_event(&mut events, &tx).await, Next::Ended);
    }
}
