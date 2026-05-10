# Week 8 Analysis Notebook

## Task
Perform statistical analysis: compute relative degradation per condition, rank conditions by impact, compare models on robustness (not just peak accuracy).

## Condition Impact Ranking (High to Low)
condition
S1T2     8.565543
S2T2     5.819093
S3T2     5.093225
S1T1     0.000000
S2T3    -0.868459
S1T3    -3.207925
S3T1    -3.335612
S2T1    -4.155654
S3T3   -18.431970

## Model Robustness (Lower = Better)
model
YOLOv8m   -17.460317
YOLOv8s    -5.263158
YOLOv8l     1.010101
YOLOv8n    17.037037

## Output Files
- degradation_heatmap.png
- bar_YOLOv8n.png
- bar_YOLOv8s.png
- bar_YOLOv8m.png
- bar_YOLOv8l.png
