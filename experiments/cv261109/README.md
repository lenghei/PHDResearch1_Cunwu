# CV261109 - UAV degradation feature model (FM_P)

Supervisor milestone: 9 November 2026. This experiment models UAV image degradations in UVL, checks valid configurations with FlamaPy, samples a configuration reproducibly, and generates a matching degraded version of the 548-image VisDrone2019-DET validation split.

This directory contains the **week-2 replacement** for the earlier 27-configuration I/B/N demonstration. It does not modify VM1, VM2, CV261102, checkpoints, or previous runs. Generation needs CPU only; no YOLO training or detector inference occurs.

## 1. Model and generation protocol

- Feature model: [`fm_p.uvl`](fm_p.uvl). It contains 25 leaf features and the five cross-tree constraints in the supervisor's proposal (Research Plan, pp. 5-6).
- The expected configuration count is **46,368**. The Python structural enumeration and the installed FlamaPy Python API must independently agree before generation starts.
- The numeric parameters are read from the UVL attributes, not duplicated in a separate table or configuration source.
- The deterministic effect sequence is atmosphere, illumination, blur, rolling shutter, noise, downscale, compression. When both blur kinds or both noise kinds are present, their internal order is motion before defocus, Gaussian before shot.
- The random seed for each applicable effect comes from SHA256 of the master degradation seed, source image stem, and feature name.
- Simulated altitude reduces and restores pixel resolution. The output image retains the original dimensions and unchanged original-image annotation coordinates. This is a resampling proxy, not a simulated camera trajectory.
- Output images use lossless PNG. Preview JPEG files are visual aids only. The original image ZIP and annotation bytes remain unchanged.

Consult [`PROTOCOL.md`](PROTOCOL.md) for the individual transformations, limitations, and constraint rationale. The physical calibration of the proposed levels and constraints remains a distinct scientific validation task.

## 2. Local environment

Open the **outer** project root `E:\ForStudy\code\uavgap` in PyCharm. Use the existing `.venv` interpreter for image generation and `.venv_flamapy` for genuine UVL SAT analysis. Do not upgrade or mix these environments.

The program automatically reads the existing VM1 private config at `main/experiments/vm1_flamapy_yolo26s_validation/config.local.yaml`, if present. It uses the following keys from that config:

- `dataset_zip`: original annotated `VisDrone2019-DET-val.zip`
- `dataset_zip_sha256`: expected source archive SHA256
- `expected_images`: 548
- `flamapy_python`: path to `.venv_flamapy/Scripts/python.exe`
- `degradation_seed`: fixed per-image effect seed

If the VM1 config is absent, copy `config.example.yaml` to `config.local.yaml` **in this directory**, and supply the corresponding environment variables or real local paths. `config.local.yaml` is Git-ignored. You may also pass `--dataset-zip` and `--flamapy-python` explicitly.

Required packages in the main environment: Python 3.11+, `numpy`, `opencv-python`, and `PyYAML`. The FlamaPy interpreter needs `flamapy-fm`, `flamapy-sat`, `flamapy-fw`, and UVL support, as already used by VM1.

## 3. Meeting demonstration, step by step

From the outer `uavgap/` root, run the following commands in order.

**Step A: CPU unit tests.** These tests include a synthetic 548-image archive and require no original data or FlamaPy packages.

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s main/experiments/cv261109/tests -p 'test_*.py' -v
```

**Step B: model and dataset admission.** FlamaPy enumerates the full UVL model and samples a legal configuration using the advisor-provided seed. The program also compares the count with an independent structural enumeration. No output run is created.

```powershell
& .\.venv\Scripts\python.exe -B main/experiments/cv261109/run.py --check --seed 1031
```

Expected message: `Preflight PASS: 548 VisDrone images; 46368 FlamaPy-valid configurations; five constraints` plus a selected configuration ID and feature list. This is an expected outcome, not a claim that it has already been observed on the user's machine.

**Step C: three-image live demonstration.** This demonstrates the chosen legal configuration, numeric parameters, and original-to-generated image alignment.

```powershell
& .\.venv\Scripts\python.exe -B main/experiments/cv261109/run.py --seed 1031 --count 3
```

Open `preview/index.html` in the printed local Run directory. The output has `status=DEMO`, not full validation. The `selection.json` lists the exact features and UVL-derived parameters.

**Step D: verify exact reproduction.** Substitute the RunID from Step C; the audit checks each saved image, the original annotation, original ZIP hash, source bytes, and a fresh deterministic re-render.

```powershell
$run = 'E:\ForStudy\code\uavgap\datasets\local_runs\cv261109\<RunID>'
& .\.venv\Scripts\python.exe -B main/experiments/cv261109/run.py --verify-run $run --replay-count 0
```

**Step E: generate the complete 548-image set.** Use the same seed, but omit `--count`; this generates every validation image under one configuration.

```powershell
& .\.venv\Scripts\python.exe -B main/experiments/cv261109/run.py --seed 1031
```

**Step F: verify the complete run.** The default audits every file and independently re-renders three images. Use `--replay-count 0` to regenerate all 548 in memory for a stronger check.

```powershell
$run = 'E:\ForStudy\code\uavgap\datasets\local_runs\cv261109\<FULL_RUN_ID>'
& .\.venv\Scripts\python.exe -B main/experiments/cv261109/run.py --verify-run $run --replay-count 0
```

**Step G: publish only small verified metadata for GitHub.** The public exporter requires a completed 548-image run and performs its own audit. This produces a `snapshot.json` and a schema-2 `manifest.json` in `main/reports/results/cv261109/<RunID>/`, for the auto-discovery dashboard.

```powershell
& .\.venv\Scripts\python.exe -B main/experiments/cv261109/run.py --publish-run $run
```

**Step H: commit and synchronize from the local authoritative Git tree.** Inspect staged changes before pushing. Never publish full generated images, raw annotations, private paths, or model weights.

```powershell
cd E:\ForStudy\code\uavgap\main
git add experiments/cv261109 reports/results/cv261109
git diff --cached --stat
git commit -m "Add supervisor FM_P generator and verified CV261109 evidence"
git push origin main
```

If online changes made `main` diverge, consult the separate local-first Git guide before an intentional force-with-lease push. The experiment program does **not** commit, pull, rebase, or push automatically.

## 4. What is saved

Each execution gets a unique, never-overwritten RunID under `datasets/local_runs/cv261109/` outside Git:

```text
<RunID>/
  selection.json           Model hash, selected leaves, UVL parameters, source/code hashes
  image_manifest.jsonl     One record per image: original/output/annotation hashes, effect seeds
  result.json              Actual generation count, real duration, source annotation counts
  images/                  Lossless full-size generated PNGs
  annotations/             Byte-for-byte copies of original VisDrone annotation files
  preview/index.html       Side-by-side source and generated images for live presentation
  preview/*.jpg            A small, clearly identified visual sample
  failure.json             Only if generation fails
```

You can reproduce a specific configuration without remembering the random seed by using `--configuration-id p_<hex>`; the selection is still validated against the complete FlamaPy model. Use the same model/code/data files and `--degradation-seed` to reproduce exact pixels. OpenCV/NumPy versions also matter for exact pixel-level reproducibility across machines.

## 5. Scientific boundaries

A `PASS` here means that the generator produced and verified 548 paired images for one valid FM_P configuration. It does **not** measure YOLO detection accuracy, prove photorealism, estimate static or empirical gaps, or validate the physical accuracy of each proposed constraint. The `CV261116` FM_D/static-gap stage is separate.

Original validation annotations and original-image size bins are preserved, including under the downscale proxy. Data and generated images are intentionally local; GitHub receives only scripts, the UVL, and an optional verified metadata snapshot.

Reference: J. A. Galindo, *Research plan: the gap between UAV image degradations and YOLO configurations*, 22 October 2026, pp. 5-6 and 15-16.
