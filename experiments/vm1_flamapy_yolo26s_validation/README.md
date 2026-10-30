# VM1: FlamaPy–YOLO26s validation

Primary checkpoint: ck01, YOLO26s fine-tuned on VisDrone, native10, input 1280.
The original VM09 UVLs and acceptance artifact are preserved here with their historical identity.

| Mode | Work | Output |
|---|---|---|
| `--check` (default) | CPU data/checkpoint checks, FlamaPy, tests, historical ck05 metric regeneration | Admission, regression logs and status HTML |
| `--demo` | First five sorted validation images, CLEAN and one seeded valid degradation | Predictions, GT, AP/AR, transformed PNGs and HTML previews |
| `--full-clean` | All 548 validation images, CLEAN | Predictions, GT, AP/AR and HTML preview |
| `--regenerate <run>` | Recalculate saved COCO metrics on CPU | Metric comparison in a new run |
| `--refresh-index` | Refresh the experiment's run history | Updated index only; no new Run or inference |

Run `python -B experiments/vm1_flamapy_yolo26s_validation/run.py --check`
from the repository root. Set `flamapy_python` in the private configuration to an existing
FlamaPy interpreter. The JSON bridge records selected features, numerical parameters,
UVL hashes and seeds. With seed 20261027, the demo selects I0B1N2 at R1280:
normal illumination, kernel 7 motion blur and Gaussian noise variance 30.

Degradation order: illumination → motion blur → noise, before Letterbox.
Source image geometry is preserved. A deterministic image-specific field controls blur angle and noise.
Metrics use the existing native10 evaluator and original-pixel size bins;
previews use a display threshold of 0.25 without changing evaluation predictions.

Each run has a unique directory under the configured external output root.
It contains the command, configuration, code hashes, environment, data manifest,
checkpoint identity, UVL selection and logs. Inference adds per-condition raw JSONL,
COCO predictions, GT, metrics, tensor traces, timing and GPU memory records.
The root `index.html` links the condition previews. Use `--no-open` to suppress browser opening.
The parent experiment index lists all saved runs, newest first, with mode/status
filters and the latest successful result for each mode. Old runs without HTML link to saved metadata.

CPU admission passed. GPU modes remain **PREPARED** until manually executed.
Five-image results are a pipeline demonstration. Validation images were used for
historical checkpoint selection, so evaluation remains development validation.
The preserved ck05 YOLOv8s baseline is separate from new ck01 results.
