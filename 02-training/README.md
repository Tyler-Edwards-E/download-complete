# 02 - Training
## Training a detection model in a Kaggle Notebook using [YOLOv8](https://github.com/ultralytics/ultralytics)

### Process
1. Log into Kaggle and upload the zipped folder from [[01]-prepreoccessing](https://github.com/Tyler-Edwards-E/raging-download/tree/main/%5B01%5D-preprocessing) (`ken-example.zip`)
2. Open `rd-yolov8-training.ipynb` in Kaggle by either copying and editing the [file from Kaggle](https://www.kaggle.com/code/tyedwardse/rd-yolov8-training-example-copy/notebook) or downloading a copy from GitHub.
3. Change the `run-name` and `character` variables in cell 3 if necessary and
4. Turn on the `GPU P100` GPU in the Kaggle Notebook `Settings` -> `Accelerators` in the top left or in `Session options` on the right
5. (Optional) Create a [wandb](https://wandb.ai/site) account and import the key at `Add-Ons` -> `Secrets` in the top left as "wandb-key" (For tracking model progress and performance)
6. Once everything is setup, run the notebook by clicking _**"Save Version"**_ in the top right (Runs the notebook without needing to have the page open)
7. When the notebook finishes running, your model weights can be found at `/kaggle/working/runs/detect/{RUN NAME}/weights/last.pt` in the notebook output


___

**Script** | **Description** |
--- | --- | 
rd-yolov8-training.ipynb | Notebook that uses [YOLOv8](https://github.com/ultralytics/ultralytics) to train sprite detection models. Can be found here or on [Kaggle](https://www.kaggle.com/code/tyedwardse/rd-yolov8-training-example-copy/notebook) | 
