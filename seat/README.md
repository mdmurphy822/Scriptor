# scriptor seat template

`scriptor.start.sh` and `scriptor.stop.sh` are seat scripts in the
shape Semantik renders: `{{name}}` placeholders only, values limited to
`[A-Za-z0-9._:/@+=,-]`, the start script returns after launching, and the stop
script ends the seat by its pid file. `params.toml` lists every parameter with
its default.

## Contract

- Launch: the engine runs detached, its pid in `<state_dir>/scriptor.pid`
  and its output in `<state_dir>/scriptor.log`.
- Readiness: `GET http://127.0.0.1:<port>/v1/models` answers 200 and lists the
  served name.
- Drain: SIGTERM stops admitting requests, finishes those in flight, and exits.
  The stop script sends SIGTERM and waits for the process to exit.
- Exit codes: the exit status tells a supervisor whether restarting can help.

  | Code | Meaning | Restart |
  | ---- | ------- | ------- |
  | 0 | Clean stop after a drain | no |
  | 78 | Refused configuration: the memory plan does not fit, or the checkpoint, tokenizer or cache directory is unusable | never; `RestartPreventExitStatus=78` |
  | 75 | Transient failure: a boot stage failed, or the device entered a sticky fault (illegal address, launch failure, ECC, destroyed context) | yes |
  | 70 | Internal failure: a lost thread or runtime that could not start | yes |

  On a sticky device fault the engine fails every in-flight request with a message naming
  the CUDA result code, answers `GET /health` with 503 and
  `{"state":"dead","cuda":"<code>"}`, stops listening and exits 75 without draining, since
  no further CUDA call can be trusted. Out-of-memory and plan-exceeded failures are
  per-request errors and do not stop the engine.
- Stop deadline: set `TimeoutStopSec` to the drain deadline (longest request's
  `max_model_len` / measured decode rate x margin + one prefill pass), never `infinity`; the
  unit example shows the derivation.
- Fault injection (testing only; off unless set, logged when armed): `SCRIPTOR_FAULT` takes
  comma-separated rules `alloc:<site>:<nth>[:oom|fatal]`, `sync:<site>:hang` and
  `panic:<stage>`; sites include `arena`, `kv_pool` and `prefill_work`.
  `SCRIPTOR_FAKE_MEMINFO` and `SCRIPTOR_FAKE_PSI` name files read in place of `/proc/meminfo` and
  `/proc/pressure/memory`; `SCRIPTOR_FAKE_DEVICE_TOTAL` replaces the detected device total in bytes.
- Engine file: an empty `engine_file` converts the checkpoint into
  `<cache_dir>/glm-ocr.engine` on first start; later starts reuse it.
- Cache: an empty `cache_dir` uses
  `$XDG_CACHE_HOME/semantik/seat-cache/scriptor`.

## Semantik registration

Semantik registers the seat with a two-line `TEMPLATES` entry pointing at these
scripts and `params.toml` defaults (a Semantik-side change, a user decision):

```rust
start: include_str!("scriptor.start.sh"),
stop: include_str!("scriptor.stop.sh"),
```
