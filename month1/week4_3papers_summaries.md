## Paper 1: PGEM-DETR: Physics-guided enhancement mechanism for drone-based object detection in adverse visual environments

### 1. Information

Title: PGEM-DETR: Physics-guided enhancement mechanism for drone-based object detection in adverse visual environments 
Link: https://academic.oup.com/jcde/article/13/1/60/8317245
cite: Siyu Zhang, Xiaohua Cao, Peng Wang, Jiale Li, PGEM-DETR: Physics-guided enhancement mechanism for drone-based object detection in adverse visual environments, Journal of Computational Design and Engineering, Volume 13, Issue 1, January 2026, Pages 60–74, https://doi.org/10.1093/jcde/qwaf121

### 2. Summary

Problem: Drone object detection in adverse visual environments (fog, low light) suffers from degraded image quality and poor robustness. 
Method: Proposed PGEM-DETR integrating physics-guided enhancement module (PGEM) and cross-architecture feature interaction module (CAFIM) with DETR. 
Dataset: AVE-CARPK, AVE-DroneVehicle, AVE-VisDrone (fog/low-light augmented from CARPK, DroneVehicle, VisDrone). 
Results: PGEM-DETR-L achieved 83.1% AP50/58.1% AP50:95 on AVE-CARPK; PGEM-DETR-N reached 183.6 FPS (INT8) on Jetson Orin NX. 
Limitation: Limited to fog/night conditions; needs optimization for more edge platforms; small object detection under extreme degradation is challenging.

## Paper 2: OLO-RAW: Advancing UAV Detection With Robustness to Adverse Weather Conditions

### 1. Information

Title: YOLO-RAW: Advancing UAV Detection With Robustness to Adverse Weather Conditions 
Link: https://ieeexplore.ieee.org/abstract/document/10976436
cite: A. Munir, A. J. Siddiqui, M. S. Hossain and A. El-Maleh, "YOLO-RAW: Advancing UAV Detection With Robustness to Adverse Weather Conditions," in IEEE Transactions on Intelligent Transportation Systems, vol. 26, no. 6, pp. 7857-7873, June 2025, doi: 10.1109/TITS.2025.3560792.




### 2. Summary

Problem: UAV detection suffers from poor robustness to adverse weather (rain, noise, motion blur), scale variations, and complex backgrounds. 
Method: Proposed YOLO-RAW with SPP-Extended, Scale-blended Feature Aggregation, and parameter-free SimAM attention module based on YOLOv5. 
Dataset: Complex Background Dataset (CBD) + 4 test sets (RTS, ATS, MBTS, MWTS) with adverse weather effects. 
Results: Achieved 71.1% mAP50 on MWTS, outperforming YOLOv5m by 5.4% mAP50 and 7.7% recall; runs at 15 FPS. 
Limitation: Slight drop in tiny UAV detection; relies on synthetic adverse weather data; inference speed is slightly lower than YOLOv5.



## Paper 3: Aerial Autonomy Under Adversity: Advances in Obstacle and Aircraft Detection Techniques for Unmanned Aerial Vehicles

### 1. Information

Title: Aerial Autonomy Under Adversity: Advances in Obstacle and Aircraft Detection Techniques for Unmanned Aerial Vehicles
Link: https://www.mdpi.com/2504-446X/9/8/549
cite: Randieri, C.; Ganesh, S.V.; Raj, R.D.A.; Yanamala, R.M.R.; Pallakonda, A.; Napoli, C. Aerial Autonomy Under Adversity: Advances in Obstacle and Aircraft Detection Techniques for Unmanned Aerial Vehicles. Drones 2025, 9, 549. https://doi.org/10.3390/drones9080549



### 2. Summary

Problem: UAV obstacle/aircraft detection faces challenges from adverse environments, thin objects, motion blur, and limited onboard computing power. 
Method: Reviewed classical (optical flow, stereo vision) and deep learning (YOLO, DETR) methods, plus multi-sensor fusion (RGB/LiDAR/radar) for harsh condition adaptation. 
Dataset: Analyzed benchmark datasets (UAVid, VisDrone, DOTA, Foggy Cityscapes) covering urban/adverse weather scenarios. 
Results: Sensor fusion and lightweight deep learning models balance accuracy/efficiency; radar performs best in harsh weather, LiDAR excels at 3D mapping. 
Limitation: Current methods lack generalization in extreme weather; sensor fusion has calibration/synchronization issues; lightweight models trade accuracy for speed.

