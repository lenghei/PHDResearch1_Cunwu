# Package validation (2026-11-01)

- CPU regression suite: 19 tests, passed.
- Independent structural FM_P enumeration: 46,368 valid configurations.
- Synthetic annotated validation ZIP: 548 image/annotation pairs.
- Synthetic full-run generation, audit, deterministic replay, and schema-2 snapshot export: passed.
- Tail-comma annotations, reject-on-tampering, effect-seed determinism, feature constraints, and 25 per-leaf renderer registrations: covered.
- Runtime FlamaPy parser/SAT enumeration: requires the dedicated `.venv_flamapy` environment on the user's Windows machine; not run in this packaging environment.
- Real VisDrone ZIP generation: requires the original local 548-image archive; not run in this packaging environment.
- No GPU inference, detector training, or AP calculation is performed by this task.
