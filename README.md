# Wafer Defect Map Classificaiton

This is a demo project built for personal use using the [MixedWM38 dataset](https://github.com/Junliangwangdhu/WaferMap). Note that there is an issue with the dataset as pointed out in [this issue](https://github.com/Junliangwangdhu/WaferMap/issues/2), which was corrected for the results shared here.

## Wafer map patterns
![image](/Wafer%20Map.png)

# Model

Uses the Ultralytics YOLOv8-Large classification model, with standard pretrained weights. The training was run for a short 10 epochs as this was only as a demo project. 

# Results

## Confusion Matrix Result
![image](/wafer_defects/EXP00002/result_confusion_matrix.png)

## Loss and Accuracy Plots
![image](/wafer_defects/EXP00002/results.png)

Overall results from `EXP0002` which was a full GPU training with validation experiment, see the `args.yaml` file to view configuration. Additional metrics were computed using [val_and_results.py](/val_and_results.py). This result should not be considered complete, as model should be trained for additional epochs and additional hyperparameters explored. It is merely a demonstration of implementing a classifier model on wafer defect maps.