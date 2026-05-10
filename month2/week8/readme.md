# Week 8 Analysis Report
**Code Run Time**: 2 minutes

## Personal View & Analysis
Based on the experimental results, some interference conditions lead to **performance improvement** instead of performance degradation. When images become brighter or darker, redundant details and background noise are greatly reduced, making objects more prominent and easier to detect. As a result, mAP values under these conditions are higher than the baseline S1T1, showing negative degradation values. This indicates that simplified images can improve detection stability and model robustness.

## Result Verification & Reasoning
### 1. Condition Impact Ranking (High to Low)
- S1T2: Positive degradation → **largest performance drop** (strongest negative influence)
- S3T3: Largest negative value → **strongest performance improvement**
- S1T3, S3T1, S2T1: Negative values → **clear performance improvement**

### 2. Result Matching & Logic Support
Negative degradation value = mAP higher than baseline S1T1  
This fully supports my analysis:  
Images become brighter / darker → reduce details and noise → cleaner background → more stable detection → higher mAP

## Completion Status
All statistical analyses are completed. I computed relative degradation for each condition, ranked conditions by impact, and compared the robustness of YOLOv8n, YOLOv8s, YOLOv8m, and YOLOv8l. Heat maps of degradation and bar charts for each model have been generated. The analysis notebook has been successfully committed.
