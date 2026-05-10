# Week 8 Analysis Script
# Perform statistical analysis: compute relative degradation, rank conditions, compare model robustness
# Models: YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the FULL experimental dataset (no slicing, no partial data)
df = pd.read_csv("week6_results.csv")

# Show basic dataset info to confirm full loading
print("=== Full Dataset Loaded Successfully ===")
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print("Unique models:", df["model"].unique())
print("Unique conditions:", df["condition"].unique())
print("\n")

# Get baseline mAP50 for each model (S1T1 is the normal baseline condition)
baseline = {}
for model in df["model"].unique():
    val = df[(df["model"] == model) & (df["condition"] == "S1T1")]["mAP50"].values[0]
    baseline[model] = val

print("=== Baseline mAP50 (S1T1) for Each Model ===")
for m, v in baseline.items():
    print(f"{model}: {v:.4f}")
print("\n")

# Calculate relative degradation percentage for every sample
def compute_degradation(row):
    base = baseline[row["model"]]
    current = row["mAP50"]
    return ((base - current) / base) * 100

df["relative_degradation"] = df.apply(compute_degradation, axis=1)

# Rank conditions by average degradation (most impact → least impact)
condition_ranking = df.groupby("condition")["relative_degradation"].mean().sort_values(ascending=False)

print("=== Condition Impact Ranking (Higher = Worse Performance Drop) ===")
print(condition_ranking.round(2))
print("\n")

# Compare model robustness (lower mean degradation = more robust)
model_robustness = df.groupby("model")["relative_degradation"].mean().sort_values()

print("=== Model Robustness (Lower = More Robust) ===")
print(model_robustness.round(2))
print("\n")

# --------------------------
# Heatmap: Relative Degradation
# --------------------------
heatmap_data = df.pivot(index="condition", columns="model", values="relative_degradation")

plt.figure(figsize=(9, 6))
sns.heatmap(heatmap_data, annot=True, cmap="Reds", fmt=".1f", linewidths=0.5)
plt.title("Relative Degradation Heatmap")
plt.tight_layout()
plt.savefig("degradation_heatmap.png")
plt.close()

# --------------------------
# Bar charts for each model
# --------------------------
models = df["model"].unique()
for model in models:
    data = df[df["model"] == model].sort_values("condition")
    plt.figure(figsize=(11, 5))
    plt.bar(data["condition"], data["mAP50"], color="steelblue")
    plt.title(f"{model} mAP50 Across Conditions")
    plt.xlabel("Condition")
    plt.ylabel("mAP50")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"bar_{model}.png")
    plt.close()

# --------------------------
# Save analysis notebook for submission
# --------------------------
with open("analysis_notebook.md", "w") as f:
    f.write("# Week 8 Analysis Notebook\n\n")
    f.write("## Task\n")
    f.write("Perform statistical analysis: compute relative degradation per condition, rank conditions by impact, compare models on robustness (not just peak accuracy).\n\n")

    f.write("## Condition Impact Ranking (High to Low)\n")
    f.write(condition_ranking.to_string())
    f.write("\n\n")

    f.write("## Model Robustness (Lower = Better)\n")
    f.write(model_robustness.to_string())
    f.write("\n\n")

    f.write("## Output Files\n")
    f.write("- degradation_heatmap.png\n")
    f.write("- bar_YOLOv8n.png\n")
    f.write("- bar_YOLOv8s.png\n")
    f.write("- bar_YOLOv8m.png\n")
    f.write("- bar_YOLOv8l.png\n")

print("=== Week 8 Analysis Finished ===")
print("All plots and analysis notebook have been saved.")