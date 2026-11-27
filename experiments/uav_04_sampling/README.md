#  Configuration sampling

From the outer `uavgap` directory:

```powershell
& ./.venv/Scripts/python.exe -B main/experiments/uav_04_sampling/run.py
```

Rebuilds the frozen 100 P × 5 D design and checks the released tables and original manifest. It also exports two full-space P cohorts and verifies the checkpoint-specific inference candidates against FM_D with FlamaPy. Add `--frozen-only` to replay just the original tables.

## Development evaluation

```powershell
& ./.venv/Scripts/python.exe -B main/experiments/uav_04_sampling/run_development.py
```

The default command checks the complete 548-image validation split, frozen checkpoint and protocol. Run with `--execute --groups 100 --hours 24` only for an explicitly authorized development evaluation. The worker follows the frozen sequence, shares a verified PNG cache across all five settings of each P, repeats the first pair, and preserves completed pairs under the new run. The supervisor enforces the thermal and time limits. Formal test-dev admission is handled in [Week 5](../uav_05_evaluation/README.md).

[Sample tables and coverage](../../reports/results/uav_04_sampling/).

## Full-space sampling design

Install the pinned BDD plugin in the local FlamaPy environment, then build the design on CPU from the outer `uavgap` directory.

```powershell
& ./.venv_flamapy/Scripts/python.exe -B -m pip install -r main/code/requirements-flamapy.txt
& ./.venv/Scripts/python.exe -B main/experiments/uav_04_sampling/design_sampling.py
```

The design draws 100 uniform configurations without replacement from the canonical FlamaPy BDD catalog. An independent 80-configuration sample covers feasible cross-dimension level pairs, including absent optional effects. The union keeps both source labels when a configuration appears in both samples. Twenty active slots remain open until a model can be fitted with eligible PASS/FAIL labels. The uniform cohort provides the prevalence denominator after its oracle outcomes exist.

The detector table binds selected inference settings to actual checkpoint hashes. It includes matched YOLO26s T1280 and YOLOv8s T1280 entries at Infer640 and Infer1280. The seed-45 Train960 control and targeted pair form a separate development comparison. The original FM_D and frozen 500-pair matrix remain separate from this proposal.

The command prints the local output directory. It replays the renderer on one fixed VisDrone validation image per selected P and audits the saved development pairs. A formal test-dev run uses separate admission.

## Analyze saved development cells

```powershell
& ./.venv/Scripts/python.exe -B main/experiments/uav_04_sampling/analyze_development.py $run $resume
```

Set $run to the original local result directory. Omit $resume when reviewing only that run. This checks sealed cells and shared image bootstrap indices, then writes Gap by degradation, threshold sensitivity, paired detector differences and remaining pairs. A Gap requires all five settings to fail. The continuation entry point verifies the original run and keeps its deadline.

Verify a continuation chain by passing each sealed continuation in execution order:

```powershell
& ./.venv/Scripts/python.exe -B main/experiments/uav_04_sampling/resume_development.py $run --previous $resume
```

Add `--execute` to resume approved work before the original deadline.
For a separately authorized continuation, pass the author's approval as `--no-deadline`. Add `--after $active` to wait for an active run to finish its authorized time window. Hardware or worker errors stop the handoff.

After verifying an exact replay, use `--parallel-bootstrap` to process draws with two CPU threads on high-detection cells. Smaller cells and low-memory systems use the serial evaluator. Inference timing is recorded before Bootstrap.

Use `--sparse-bootstrap` for cached single-cap COCO evaluation. It preserves stable score ties, the existing image draws, and exact interpolated precision. Incompatible inputs use the selected reference evaluator.

Use `--sparse-matching` to skip detections below every matching IoU threshold inside the COCO matcher. All detections remain in precision accumulation and saved predictions.

GPU utilization is required at startup. During execution, a utilization query error is recorded separately from the live temperature, memory, and device identity checks.
