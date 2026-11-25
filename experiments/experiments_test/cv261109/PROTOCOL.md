# FM_P transformation protocol and research notes

## Scope

The model follows the supervisor's October 2026 draft (pages 5-6). The constraints and numeric levels are **research hypotheses**, not universally valid camera-physics laws. This baseline should subsequently be calibrated with real UAV imagery; the constraints may be revised, with every revision recorded and re-counted.

## Feature groups and implementation

| Group | Selected UVL level | Operation |
|---|---|---|
| Illumination | Normal / overexposed / underexposed / low light | `clip(gain * image + bias, 0, 255)` |
| Blur | Motion_k7/k15/k21 | Rotate a normalized line PSF using UVL angle bounds and a per-image seed; `filter2D` |
| Blur | Defocus_r3/r6 | Circular normalized disk PSF; `filter2D` |
| Noise | Gauss_s5/s15/s30 | Zero-mean pixelwise Gaussian noise with `sigma` from UVL |
| Noise | Shot_low/high | Poisson counts at expected photon level from UVL, reconstructed to intensity |
| Compression | JPEG_q75/q50/q25 | OpenCV JPEG encode/decode at the UVL quality level |
| Atmosphere | Haze_light/dense | Fixed-depth atmospheric blending with transmission `exp(-beta)` and UVL airlight |
| Atmosphere | Rain | Seeded synthetic line streaks; count, length, opacity from UVL |
| Altitude | Downscale_x1_5/x2 | Downsample using INTER_AREA and resize to original pixels using INTER_LINEAR |
| Sensor | RollingShutter | Row-wise periodic horizontal displacement with amplitude and waves from UVL |

All numeric operation levels come from `fm_p.uvl`. The extra implementation attributes `Motion.angle_min/angle_max`, `Haze.airlight`, `Rain.stroke_length`, `Rain.opacity`, `RollingShutter.max_shift_fraction`, and `RollingShutter.waves` concretize two under-specified effects from the supervisor's draft. Their values are **provisional**; they are not claimed as empirically calibrated.

## Cross-tree constraints

1. `LowLight => ShotNoise`: follows the draft's association of strong low-light effects with sensor photon noise. It should be tested against actual exposure/gain conditions; low light does not always imply visibly dominant shot noise.
2. `Rain => !Overexposed_strong & !Overexposed_mild`: narrows the sampled scenario space as in the draft; not a universal physical law because rain and high camera exposure can co-occur.
3. `Haze_dense => !Overexposed_strong`: a study design exclusion, not a universal atmospheric constraint.
4. `RollingShutter => MotionBlur`: a simplified shared-motion scenario assumption; a rolling-shutter effect can exist without visible motion blur in real sensors.
5. `Shot_high => LowLight | Underexposed_strong`: a restriction on severe shot-noise samples for this study; physical noise characteristics depend on photon statistics and camera processing.

No external paper is presented as proving these five constraints. They are reproduced from the supervisor's October 2026 proposal for an auditable first implementation. Physical calibration and literature support must be checked before treating them as final experimental constraints.

## Representation and repeatability

- FlamaPy enumerates legal UVL configurations. A separate Python structural enumerator must agree on the count of **46,368**.
- Different seeds give deterministic selection from a lexicographically sorted collection of leaf sets. Exact selection is recorded in `selection.json`.
- Each image/effect uses SHA256-derived RNG seed from the master seed, filename stem, and feature name. The full selected feature list and each effect's random parameters are written to `image_manifest.jsonl`.
- Simulated altitude **does not** change the output dimensions or annotation coordinate system; it changes sampled visual detail and then restores resolution. It is not an elevation/scale-corrected 3D projection.
- A clean `Normal` selection is valid in the formal feature space. The generator records it rather than quietly re-sampling a degradation.
- All generated images are stored as lossless PNG. Any JPEG preview is unrelated to evaluation input.
- Source image and annotation bytes are never rewritten inside the source ZIP. Annotation copies are checksum-checked.
- File SHA256 plus deterministic pixel replay tests the exact released effects; reproducibility can depend on pinned OpenCV/NumPy and CPU implementations.

## Next study (FM_D / static gap)

For the 16 November milestone, map each degradation feature to evidenced, *actual* training augmentations of a specific detector/checkpoint, and measure per-configuration resource attributes on the RTX 4060. Do not equate syntactic FM_P coverage with observed detection robustness. The static gap and the empirical gap require independent evidence.
