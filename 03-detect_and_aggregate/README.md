# 03 - Detection & Post-Processing (WIP)
## Detecting sprites with [YOLOv11*](https://github.com/ultralytics/ultralytics) and aggregating the output into useable data

_Detects sprites on screen in each frame of a video using the weights trained in the previous step. The raw output will be text files for each frame one or more objects were detected in the video. The scripts in this step should take that frame-by-frame output and convert it into a dataset of "actions" with context that can be analyzed. **(This is the only step necessary if you just want to use the weights I created.)**_


### Process
1. Download `new_detect.py`, `rd_aggregate.py` and the relevant character `weights` you need. Some weights might be mandatory like the "effects" and "roster" weights. Place everything in the same folder together
2. Open `new_detect.py` and edit the filepaths for the variables `video_filepath`, `roster_filepath`, `character1`, `model1_filepath`, `character2`, `model2_filepath` as neeeded
3. Run `new_detect.py` (This may take a while depending on the length of your video)
4. Several Excel files will be generated when the script is completed but the final output for analysis will be in `30-results.xlsx`

___

**Script** | **Description** |
--- | --- | 
`new_detect.py` | Main workspace used to run sprite detection and the functions in `rd_aggregate.py`| 
`rd_aggregate.py` | Variety of functions used in `new_detect.py` that aggregates the YOLO text file output into data | 
`labelmaps` | Key that translates the numeric labels in the text files to the actual move/character names. Shouldn't need to edit these if using my weights. | 
`weights` | Models trained for given fighting game characters and also some contextual models for effects, HUD, roster, etc. | 

