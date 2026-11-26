//! The logits fixture: bf16 verification-row logits dumped from a real checkpoint together with
//! the sampling state each row was chosen under.
//!
//! All integers and floats are little-endian. The file is
//!
//! | field        | type          | meaning                                              |
//! |--------------|---------------|------------------------------------------------------|
//! | magic        | 8 bytes       | `SCRLGT01`                                           |
//! | vocab        | `u32`         | row width, at least 1                                |
//! | q            | `u32`         | rows per group (`k + 1`), at least 1                 |
//! | groups       | `u32`         | number of groups, at least 1                         |
//! | provenance   | `u32` + bytes | UTF-8 `key=value` lines naming the source            |
//! | group * N    |               | see below                                            |
//!
//! Each group is one sequence's verification step: `repetition`, `presence`, `frequency`
//! (`f32` each), `n_prompt`, `n_output`, `n_drafts` (`u32` each, `n_drafts == q - 1`), the
//! prompt token ids, the output token ids emitted before the step, the draft token ids, and
//! `q * vocab` bf16 bit patterns (`u16`), row `j` being the logits that verify draft `j`.
//!
//! The provenance must carry non-empty `model`, `checkpoint`, `tool` and `prompt` keys. The
//! file must end exactly after the last group.

const MAGIC: &[u8; 8] = b"SCRLGT01";
const REQUIRED_KEYS: [&str; 4] = ["model", "checkpoint", "tool", "prompt"];

/// One sequence's verification step.
#[derive(Debug, Clone, PartialEq)]
pub struct FixtureGroup {
    /// Repetition penalty the step ran under.
    pub repetition: f32,
    /// Presence penalty the step ran under.
    pub presence: f32,
    /// Frequency penalty the step ran under.
    pub frequency: f32,
    /// Prompt token ids.
    pub prompt: Vec<u32>,
    /// Output token ids emitted before the step.
    pub output: Vec<u32>,
    /// The `q - 1` drafts the step verified.
    pub drafts: Vec<u32>,
    /// `q * vocab` bf16 bit patterns.
    pub logits: Vec<u16>,
}

/// A parsed or to-be-written logits fixture.
#[derive(Debug, Clone, PartialEq)]
pub struct LogitsFixture {
    /// `key=value` provenance entries in file order.
    pub provenance: Vec<(String, String)>,
    /// Row width.
    pub vocab: usize,
    /// Rows per group.
    pub q: usize,
    /// The groups.
    pub groups: Vec<FixtureGroup>,
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize, what: &str) -> Result<&[u8], String> {
        let end = self
            .at
            .checked_add(n)
            .filter(|&e| e <= self.bytes.len())
            .ok_or_else(|| format!("truncated while reading {what}"))?;
        let out = &self.bytes[self.at..end];
        self.at = end;
        Ok(out)
    }

    fn u32(&mut self, what: &str) -> Result<u32, String> {
        let b = self.take(4, what)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn f32(&mut self, what: &str) -> Result<f32, String> {
        self.u32(what).map(f32::from_bits)
    }

    fn u32s(&mut self, n: usize, what: &str) -> Result<Vec<u32>, String> {
        let bytes = n
            .checked_mul(4)
            .ok_or_else(|| format!("{what} length overflows"))?;
        Ok(self
            .take(bytes, what)?
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
}

impl LogitsFixture {
    /// The value of provenance `key`.
    pub fn provenance_value(&self, key: &str) -> Option<&str> {
        self.provenance
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Checks every structural rule of the format.
    pub fn validate(&self) -> Result<(), String> {
        if self.vocab == 0 {
            return Err("vocab is zero".into());
        }
        if self.q == 0 {
            return Err("rows per group is zero".into());
        }
        if self.groups.is_empty() {
            return Err("the fixture holds no groups".into());
        }
        for key in REQUIRED_KEYS {
            if self
                .provenance_value(key)
                .is_none_or(|v| v.trim().is_empty())
            {
                return Err(format!("provenance lacks a non-empty `{key}`"));
            }
        }
        for (k, v) in &self.provenance {
            if k.is_empty() || k.contains(['=', '\n']) || v.contains('\n') {
                return Err(format!(
                    "provenance entry `{k}` is not a single key=value line"
                ));
            }
        }
        for (i, g) in self.groups.iter().enumerate() {
            if g.logits.len() != self.q * self.vocab {
                return Err(format!(
                    "group {i} holds {} logits, expected {}",
                    g.logits.len(),
                    self.q * self.vocab
                ));
            }
            if g.drafts.len() + 1 != self.q {
                return Err(format!(
                    "group {i} holds {} drafts for q {}",
                    g.drafts.len(),
                    self.q
                ));
            }
            let bad = g
                .prompt
                .iter()
                .chain(&g.output)
                .chain(&g.drafts)
                .find(|&&t| t as usize >= self.vocab);
            if let Some(t) = bad {
                return Err(format!("group {i} holds token {t} outside the vocabulary"));
            }
        }
        Ok(())
    }

    /// The file bytes, after validation.
    pub fn encode(&self) -> Result<Vec<u8>, String> {
        self.validate()?;
        let mut out = MAGIC.to_vec();
        let mut put = |v: u32| out.extend_from_slice(&v.to_le_bytes());
        put(self.vocab as u32);
        put(self.q as u32);
        put(self.groups.len() as u32);
        let text: String = self
            .provenance
            .iter()
            .map(|(k, v)| format!("{k}={v}\n"))
            .collect();
        out.extend_from_slice(&(text.len() as u32).to_le_bytes());
        out.extend_from_slice(text.as_bytes());
        for g in &self.groups {
            for v in [g.repetition, g.presence, g.frequency] {
                out.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            for n in [g.prompt.len(), g.output.len(), g.drafts.len()] {
                out.extend_from_slice(&(n as u32).to_le_bytes());
            }
            for t in g.prompt.iter().chain(&g.output).chain(&g.drafts) {
                out.extend_from_slice(&t.to_le_bytes());
            }
            for b in &g.logits {
                out.extend_from_slice(&b.to_le_bytes());
            }
        }
        Ok(out)
    }

    /// Parses and validates a fixture; any malformation, truncation or trailing byte is an
    /// error naming what was wrong.
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        let mut r = Reader { bytes, at: 0 };
        if r.take(8, "magic")? != MAGIC {
            return Err("bad magic: not a logits fixture".into());
        }
        let vocab = r.u32("vocab")? as usize;
        let q = r.u32("q")? as usize;
        let groups = r.u32("group count")? as usize;
        if vocab == 0 || q == 0 || groups == 0 {
            return Err(format!(
                "vocab {vocab}, q {q} and groups {groups} must be nonzero"
            ));
        }
        let len = r.u32("provenance length")? as usize;
        let text = std::str::from_utf8(r.take(len, "provenance")?)
            .map_err(|_| "provenance is not UTF-8".to_string())?;
        let provenance = text
            .lines()
            .map(|l| {
                l.split_once('=')
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .ok_or_else(|| format!("provenance line `{l}` has no `=`"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let row_bytes = q
            .checked_mul(vocab)
            .and_then(|n| n.checked_mul(2))
            .ok_or("row size overflows")?;
        let mut out = Vec::new();
        for i in 0..groups {
            let repetition = r.f32("repetition")?;
            let presence = r.f32("presence")?;
            let frequency = r.f32("frequency")?;
            let n_prompt = r.u32("prompt length")? as usize;
            let n_output = r.u32("output length")? as usize;
            let n_drafts = r.u32("draft count")? as usize;
            let prompt = r.u32s(n_prompt, "prompt")?;
            let output = r.u32s(n_output, "output")?;
            let drafts = r.u32s(n_drafts, "drafts")?;
            let raw = r.take(row_bytes, &format!("group {i} logits"))?;
            let logits = raw
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            out.push(FixtureGroup {
                repetition,
                presence,
                frequency,
                prompt,
                output,
                drafts,
                logits,
            });
        }
        if r.at != bytes.len() {
            return Err(format!(
                "{} trailing bytes after the last group",
                bytes.len() - r.at
            ));
        }
        let fixture = Self {
            provenance,
            vocab,
            q,
            groups: out,
        };
        fixture.validate()?;
        Ok(fixture)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(q: usize, vocab: usize, groups: usize) -> LogitsFixture {
        let provenance = ["model", "checkpoint", "tool", "prompt"]
            .iter()
            .map(|k| ((*k).to_string(), format!("{k}-value")))
            .collect();
        let groups = (0..groups)
            .map(|g| FixtureGroup {
                repetition: 1.1,
                presence: 0.0,
                frequency: 0.25,
                prompt: vec![1, 2, (g % vocab) as u32],
                output: vec![3],
                drafts: (0..q - 1).map(|d| (d % vocab) as u32).collect(),
                logits: (0..q * vocab).map(|i| (i * 7 + g) as u16).collect(),
            })
            .collect();
        LogitsFixture {
            provenance,
            vocab,
            q,
            groups,
        }
    }

    #[test]
    fn a_fixture_round_trips_through_its_bytes() {
        for (q, vocab, groups) in [(1, 8, 8), (4, 16, 3), (1, 33, 1)] {
            let f = sample(q, vocab, groups);
            let bytes = f.encode().expect("encode");
            assert_eq!(LogitsFixture::decode(&bytes).expect("decode"), f);
        }
    }

    #[test]
    fn truncation_at_every_length_is_rejected() {
        let bytes = sample(2, 8, 2).encode().expect("encode");
        for cut in 0..bytes.len() {
            assert!(LogitsFixture::decode(&bytes[..cut]).is_err(), "cut {cut}");
        }
    }

    #[test]
    fn an_odd_byte_tail_or_trailing_bytes_are_rejected() {
        let mut bytes = sample(2, 8, 2).encode().expect("encode");
        bytes.push(0);
        assert!(
            LogitsFixture::decode(&bytes)
                .expect_err("odd tail")
                .contains("trailing")
        );
        bytes.push(0);
        assert!(LogitsFixture::decode(&bytes).is_err());
    }

    #[test]
    fn zero_dimensions_and_bad_magic_are_rejected() {
        let good = sample(1, 8, 1).encode().expect("encode");
        let mut zero_vocab = good.clone();
        zero_vocab[8..12].copy_from_slice(&0u32.to_le_bytes());
        assert!(LogitsFixture::decode(&zero_vocab).is_err());
        let mut magic = good;
        magic[0] ^= 1;
        assert!(
            LogitsFixture::decode(&magic)
                .expect_err("magic")
                .contains("magic")
        );
    }

    #[test]
    fn missing_provenance_and_out_of_range_tokens_are_rejected() {
        let mut f = sample(1, 8, 1);
        f.provenance.retain(|(k, _)| k != "checkpoint");
        assert!(f.encode().expect_err("provenance").contains("checkpoint"));
        let mut f = sample(1, 8, 1);
        f.groups[0].prompt.push(8);
        assert!(f.encode().expect_err("token").contains("outside"));
        let mut f = sample(2, 8, 1);
        f.groups[0].drafts.clear();
        assert!(f.encode().is_err());
    }
}
