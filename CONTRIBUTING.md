# Contributing to Scriptor

- Commit subjects use Conventional Commit style (`fix: ...`, `feat: ...`).
- Gate before review: `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo test --workspace --locked`.
- Tests use synthetic fixtures only; GPU tests are `#[ignore = "needs GPU"]`.
- Committed files carry no machine paths, credentials or private data.
- Code comments state what an item does, not its history.

## Contributor License Agreement

Scriptor is dual-licensed under AGPL-3.0-only or a commercial license. Outside
contributions are accepted only under a signed Contributor License Agreement
(CLA) that grants the copyright holder the right to license the contribution
under both AGPL-3.0-only and commercial terms. The CLA text is available from
contact@chironcourseworks.com.
