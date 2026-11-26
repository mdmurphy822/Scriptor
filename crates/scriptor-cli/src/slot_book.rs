//! Ownership bookkeeping of the device step's sequence slots.

use std::collections::{BTreeSet, HashMap};

use scriptor_engine::{EngineError, RequestId};

/// Every slot is in exactly one of: free, owned by a request, pending inside an adoption, or
/// retired after a fatal device error.
pub struct SlotBook {
    free: BTreeSet<usize>,
    owned: HashMap<RequestId, usize>,
    pending: BTreeSet<usize>,
    retired: BTreeSet<usize>,
}

impl SlotBook {
    /// A book over slots `0..slots`, all free.
    pub fn new(slots: usize) -> Self {
        Self {
            free: (0..slots).collect(),
            owned: HashMap::new(),
            pending: BTreeSet::new(),
            retired: BTreeSet::new(),
        }
    }

    /// The slot owned by `id`.
    pub fn slot_of(&self, id: RequestId) -> Option<usize> {
        self.owned.get(&id).copied()
    }

    /// Reserves a free slot, runs `setup` on it, and gives it to `id` only when `setup`
    /// succeeds. A failed setup returns the slot to the free set, except after a fatal error,
    /// which retires the slot because the device state is uncertain.
    pub fn adopt(
        &mut self,
        id: RequestId,
        setup: impl FnOnce(usize) -> Result<(), EngineError>,
    ) -> Result<usize, EngineError> {
        let slot = self
            .free
            .pop_first()
            .ok_or_else(|| EngineError::Executor("no free device slot for the sequence".into()))?;
        self.pending.insert(slot);
        let result = setup(slot);
        self.pending.remove(&slot);
        match result {
            Ok(()) => {
                if let Some(old) = self.owned.insert(id, slot) {
                    self.free.insert(old);
                }
                Ok(slot)
            }
            Err(e) => {
                if e.is_fatal() {
                    self.retired.insert(slot);
                } else {
                    self.free.insert(slot);
                }
                Err(e)
            }
        }
    }

    /// Frees the slot owned by `id` and returns it.
    pub fn release(&mut self, id: RequestId) -> Option<usize> {
        let slot = self.owned.remove(&id)?;
        self.free.insert(slot);
        Some(slot)
    }

    /// Slots in each state: free, owned, pending, retired.
    #[cfg(test)]
    pub fn counts(&self) -> [usize; 4] {
        [
            self.free.len(),
            self.owned.len(),
            self.pending.len(),
            self.retired.len(),
        ]
    }

    /// Whether every slot of `0..slots` is in exactly one state.
    #[cfg(test)]
    pub fn is_partition(&self, slots: usize) -> bool {
        let mut all: Vec<usize> = self.free.iter().copied().collect();
        all.extend(self.owned.values());
        all.extend(&self.pending);
        all.extend(&self.retired);
        all.sort_unstable();
        all == (0..slots).collect::<Vec<_>>()
    }
}
