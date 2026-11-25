# VM44: Tiny errors and measured selection

A fixed-score Tiny diagnostic uses the accepted VM43 predictions. The separate selector uses VM36 development measurements to choose an interval-qualified detector on 4 of 21 conditions. It abstains when measured evidence cannot support a choice. The feature configuration is supplied by the user.

[Published result summaries](../../../reports/results/vm44_tiny_error_variability_decision)

From the repository root, verify the snapshot hashes and replay the available summary calculations with Python 3.10 or later:

```bash
python -B code/tools/replay_public_results.py vm44_tiny_error_variability_decision
```

The source RunIDs and SHA256 digests are in each manifest. Full regeneration requires the archived run inputs.
