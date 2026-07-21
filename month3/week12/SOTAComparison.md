### SOTA Comparative Analysis Report: VisDrone Dataset Performance

**Student:** Cunwu

**Dataset:** VisDrone-DET 2019

**Focus:** Zero-Shot Baseline ($S_1T_1$) vs. 9 Degradation Conditions across YOLOv8 Variants

---

#### 1. Performance Overview & SOTA Comparison

Here is the quick comparison between published fine-tuned SOTA models (e.g., TPH-YOLOv5, Drone-DETR) and our zero-shot baseline evaluations on VisDrone:

| Metric / Category | Published SOTA | Our Best Baseline ($S_1T_1$) | Our 9-Condition Average | Top Model Scale |
| --- | --- | --- | --- | --- |
| **`Car` Class $\text{mAP}_{50}$** | **~62.5%** | **41.70%** | **32.75%** | YOLOv8m |
| **`Person` Class $\text{mAP}_{50}$** | **~38.0%** | **22.18%** | **13.07%** | YOLOv8l |
| **Overall Dataset $\text{mAP}_{50}$** | **38.1% – 53.9%** | **13.14%** | **9.06%** | YOLOv8l |

---

#### 2. Key Category Analysis

##### A. `Car` Category (Best Performing)

* **Our Result:** Reached **41.70% $\text{mAP}_{50}$** on YOLOv8m (and **41.53%** on YOLOv8l) under clear baseline conditions ($S_1T_1$).
* **Comparison to SOTA:** Standard SOTA achieves **~62.5% $\text{mAP}_{50}$**. The ~20.8% gap exists because we evaluated zero-shot without VisDrone fine-tuning, and used $640\times640$ resolution instead of $1280\times1280$.
* **Why it works best:** Vehicles have larger bounding box areas and distinct geometric shapes, making COCO pre-trained weights transfer much better to aerial views.
* **Environmental Robustness:** Across all 9 blur and noise conditions, the `car` category maintained a solid average of **32.75% $\text{mAP}_{50}$**, making it our most resilient class.

##### B. `Person` Category (Second Best)

* **Our Result:** Reached **22.18% $\text{mAP}_{50}$** on YOLOv8l and **22.04%** on YOLOv8m under baseline conditions.
* **Comparison to SOTA:** SOTA models reach **~38.0% $\text{mAP}_{50}$**.
* **Why it struggles:** Pedestrians are extremely tiny in UAV images ($<15\times15$ pixels). Without domain fine-tuning or specialized small-object detection heads (P2 layer), standard pre-trained models lose spatial details during downsampling.

---

#### 3. Summary & Next Steps

* **Current Status:** Out-of-the-box pre-trained YOLOv8 models handle medium targets (`car`) reasonably well (**41.70%**), but fail on tiny objects and overall dataset scores without domain training.
* **Next Steps:**
1. Fine-tune YOLO models on the VisDrone training set using our custom `train.py` script to close the ~20.8% gap to SOTA.
2. Increase input resolution ($1024\text{px}$ or $1280\text{px}$) to boost `person` detection.
3. Benchmark newer models (YOLOv10 / YOLOv11) under the same 9 degradation conditions.
