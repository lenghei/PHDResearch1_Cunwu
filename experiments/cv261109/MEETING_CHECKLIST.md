# CV261109 live replication checklist

1. Open `fm_p.uvl` and show the feature tree and five constraints.
2. Run the CPU tests with the current main environment.
3. Execute `run.py --check --seed <supervisor_seed>`; show the FlamaPy count, features, and configuration ID.
4. Execute `run.py --seed <supervisor_seed> --count 3`; open the printed `preview/index.html`.
5. Open `selection.json`: show the exact UVL attributes, model SHA256, effect order, and dataset SHA256.
6. Open `image_manifest.jsonl`: show per-image original, generated, and annotation hashes, and per-effect seeds.
7. Execute `run.py --verify-run <DEMO_RUN> --replay-count 0`; show identical regenerated pixels.
8. Execute the same seed without `--count` to generate all 548 images.
9. Audit the full run, optionally regenerate all 548 images in memory, and publish the reviewed metadata snapshot.
10. Show the `reports/results/cv261109/<RunID>/` Git-ready files. Keep the original and generated images local.

Interpretation: generated images and a verified formal configuration are evidence for the week-2 generator. They are not detector AP results or a physical realism validation.
