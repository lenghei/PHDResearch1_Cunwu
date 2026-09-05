# Work report for September

I recently refined the research plan with more focus on the **research contribution and novelty**. The paper will now concentrate on four main aspects.

1. **3×3 Compound Degradation Evaluation**  
   Continue using the current 3×3 compound degradation experiment to simulate common UAV inspection conditions, including illumination changes, motion blur, and image noise.

2. **640 vs. 1280 Resolution Comparison**  
   Add a comparison between 640 and 1280 input resolutions, with particular attention to small-object detection. Since tiny and small objects are common in UAV aerial inspection, the experiment will investigate whether the performance improvement at higher resolution is mainly concentrated on tiny/small objects rather than only improving overall mAP.

3. **Zero-Shot vs. VisDrone Fine-Tuned Comparison**  
   Compare zero-shot models with VisDrone fine-tuned models to investigate whether fine-tuning improves not only normal detection accuracy, but also model robustness under difficult conditions.

4. **YOLOv8s vs. YOLO26s Comparison**  
   The main model comparison will focus on YOLOv8s and the newer YOLO26s. YOLO26s includes improvements related to small-object detection, so the experiment will verify whether these improvements are effective for UAV small-object detection and robustness under complex conditions. This also makes the paper more meaningful than simply comparing different model sizes within the same YOLO generation.

Currently, I am completing the updated experimental code, including unified training settings, 640/1280 evaluation, nine degradation conditions, genuine mAP50/mAP50:95 calculation, and small-object size statistics. After the code is completed, I will rerun the full set of experiments.

