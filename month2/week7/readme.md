# Week 7 Summary
- All result data has been fully recorded and result files have been successfully uploaded.
- The project plan is being executed step-by-step as scheduled.
- Following Professor Galindo’s suggestion: *foresee any other mechanisms to introduce artefacts within the videos (flying birds, camera distortions, etc??)*, I am currently researching and considering adding realistic interference factors such as flying birds and blur caused by camera focus failure.

# Personal View
- Next week, I will analyze and compare the generated images under the 9 conditions to evaluate whether they represent common real-world scenarios, making the research more practical.
- Major factors affecting image quality include weather (fog, rain, snow), lighting conditions (sunlight, exposure, backlight, night), and drone hardware issues (focus failure, water on lens, dust blur).
- Since potential interference factors are too numerous to simulate entirely, the purpose of this study is to identify key interference conditions, analyze their impact, and provide objective references for improving image utilization and object detection performance.

# Problems & Solutions (Week 6)
**Problem Encountered**:
The annotation files were in the original VisDrone format (comma-separated), while YOLO requires space-separated and normalized coordinates, causing the error `could not convert string to float` and resulting in evaluation failure.

**Solution**:
Run the format conversion script `convert_visdrone_to_yolo.py` to convert VisDrone labels to the standard YOLO format.
