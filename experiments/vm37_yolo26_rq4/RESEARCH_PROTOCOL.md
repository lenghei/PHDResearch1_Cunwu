# YOLO26-first matched RQ4 study

Primary: YOLO26s. Comparison: YOLOv8s. Preserve the supervisor RQ1-RQ4 and original VM36 feature models.

First run native10 FP32 training at 1280, batch=2, workers=0 and seed=42 with the fixed 64/16 engineering membership. Then measure a complete 6471-image epoch and 548-image development validation under the candidate 100-epoch schedule. Stop the cost study after epoch 1. Neither is a completed scientific arm.

Main candidate: two architectures × three seeds (42/43/44) × CONTROL/ALL_FIVE; 100 epochs, 1280, batch=2, nbs=64, AdamW lr0=0.01, cosine, warmup=3, FP32 and workers=0. Each seed-matched pair shares initialization and the final-epoch EMA rule. Historical best checkpoints are not matched controls.

CONTROL retains ordinary HSV/geometry/mosaic/Albumentations. ALL_FIVE adds the policy in `TARGETED_POLICY.json` before the shared geometric pipeline. Each family has an independent probability of 0.5. Order: haze, exposure, motion blur, noise, JPEG. Draws use the seed, image ID, epoch, occurrence and family.

Training ranges are haze beta0.5–1.5 with airlight255; exposure gain0.4–1.3 and bias−50–80; odd motion kernels within7–21 and orientations0–180degrees; Gaussian sigma5.5–30 or Poisson scale20–100 with equal conditional probability; JPEG quality25–75. Exact discrete evaluation strengths are rejected. The actual admissible motion kernels are9,11,13,17,19. Numeric separation does not establish independent image populations.

The probability is a fixed documented prior, not a tuned optimum. White haze airlight follows the DehazeNet convention; applying constant transmission to a full image is an uncalibrated approximation. These continuous training draws represent the five FM_D capabilities, not enumerated FM_P leaf states. Original FM_P constraints still govern evaluation. Engineering fixtures cannot admit ALL_FIVE.

Research evaluation retains native10, FP32, square Letterbox, one-to-many plus class-aware multi-label NMS, conf=0.001, IoU=0.7, max_det=3000, max_nms=30000 and the original ignore-union/COCO implementation. Internal library validation has a separate metric definition. The installed YOLO26 setter selects the inference head while retaining both training branches; the official loss selects E2ELoss when the dual branch exists.

Ultralytics8.4.210 is frozen to official commit `c6427a7816d42ab4022b5f9a0242fe93e40e1ff8`. Its one-epoch Canary has no warmup and closes mosaic at startup. A complete first-epoch cost study uses `epochs=100` and stops after epoch1:3236batches,784 optimizer/EMA updates,9708warmup batches and1 pending microbatch. The100-epoch source rule predicts323600batches,10917updates and25 pending microbatches; these are planned counts, not completed results.

Health checks cover every training loss and gradient batch, model/EMA/optimizer states at every update, BN running variance, gradient norm, learning rate and atomic FP32 save/reload. Grouped checks and read-only decoded-pixel caching reduce synchronization and repeated hashing without removing checks. CPU parity and a separate GPU Canary are required for this implementation. CUDA event intervals are timing intervals, not a pure-kernel or energy benchmark.

Snapshots retain model, EMA, optimizer, scheduler and scaler. They do not preserve all RNG, sampler and pending-gradient state; exact training resume is not validated. Preserve failed runs and recover an evaluation stage separately when training itself passed.

One new 48-hour allocation includes at most 2 hours of initial admission. One GPU worker, no budget reset. A complete 100-epoch pair must fit measured remaining compute/storage before either arm starts. Test-dev remains DATASET_BLOCKED. Validation results are DEVELOPMENT_ONLY.

`COST_ADMISSION_POLICY.json` freezes the runtime margin and21-context evaluation reserve before the optimized complete-epoch measurement. A forecast is not a measured100-epoch duration. The formal100-epoch entry remains closed until a complete pair passes scientific, compute and storage gates.

The cost policy reserves a minimum of 21 contexts per arm at 1280 for one architecture. This is a conditional measured-subset gap, not the full original four-detector population or the full legal UVL space. Additional resolutions and architectures require additional evaluation resources.
