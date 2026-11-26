//! Host mirror of the per-slot block tables held in device memory.

use scriptor_cache::BlockId;
use scriptor_engine::EngineError;

/// Host copy of each slot's device block table, with a validity flag per slot. A slot's copy
/// describes device memory only while it is valid; an invalid slot uploads on its next sync.
pub struct TableMirror {
    tables: Vec<Vec<i32>>,
    valid: Vec<bool>,
}

impl TableMirror {
    /// A mirror of `slots` tables of `stride` entries each, all invalid.
    pub fn new(slots: usize, stride: usize) -> Self {
        Self {
            tables: vec![vec![0; stride]; slots],
            valid: vec![false; slots],
        }
    }

    /// Marks `slot`'s device table as unknown so the next sync uploads it.
    pub fn invalidate(&mut self, slot: usize) {
        self.valid[slot] = false;
    }

    /// Makes the device table of `slot` hold `blocks` zero-padded to the stride. `upload` receives
    /// the padded table and is called only when the device contents may differ; the mirror is
    /// committed only after `upload` returns `Ok`.
    pub fn sync(
        &mut self,
        slot: usize,
        blocks: &[BlockId],
        upload: impl FnOnce(&[i32]) -> Result<(), EngineError>,
    ) -> Result<(), EngineError> {
        let stride = self.tables[slot].len();
        if blocks.len() > stride {
            return Err(EngineError::Executor(format!(
                "a sequence holds {} blocks, the step supports {}",
                blocks.len(),
                stride
            )));
        }
        let mirror = &self.tables[slot];
        if self.valid[slot]
            && blocks.iter().zip(mirror).all(|(&b, &m)| b as i32 == m)
            && mirror[blocks.len()..].iter().all(|&m| m == 0)
        {
            return Ok(());
        }
        let mut next = vec![0_i32; stride];
        for (n, &b) in next.iter_mut().zip(blocks) {
            *n = b as i32;
        }
        self.valid[slot] = false;
        upload(&next)?;
        self.tables[slot] = next;
        self.valid[slot] = true;
        Ok(())
    }
}
