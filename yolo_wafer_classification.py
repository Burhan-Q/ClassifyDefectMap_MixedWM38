from ultralytics import YOLO
from pathlib import Path
import numpy as np
# print('go')
# Load a model
model = YOLO('yolov8l-cls.yaml')  # build a new model from YAML
data_dir = "Q:/ML_data/MixedWM38/2023-05-02_MixedWM38_Kaggle/data"

model.to('cuda') if model.device.type == 'cpu' else None

# Workaround
model.trainer.build_dataset(img_path=data_dir) # loads data, but misses 'val_dir'
model.trainer.data['val'] = Path("Q:/ML_data/MixedWM38/2023-05-02_MixedWM38_Kaggle/data/val")

# Train the model
model.train(data=data_dir,
            epochs=10,
            batch=16,
            imgsz=64,
            device=0,
            workers=12,
            project='wafer_defects',
            name='EXP000',
            seed=17,
            deterministic=True,
            val=False,
            # save_json=True,
            # save_conf=True,
            dropout=0.15,   # default 0.0
            mosaic=0.96,    # default 1.0
            # flipud=0.0,     # default 0.0
            # fliplr=0.5,     # default 0.5
            # scale=0.5,      # default 0.5
            # translate=0.1,  # default 0.1
            degrees=50,     # default 0.0
            # mixup=0.0,      # default 0.0
            # copy_paste=0.0  # default 0.0
            )

cls_count = model.metrics.confusion_matrix.nc
cls_names = model.names
matrix = model.metrics.confusion_matrix.matrix # rows are predictions, columns are ground-truth

tp, fp = model.metrics.confusion_matrix.tp_fp() # true positive and false positive (total of true class incorrect)
fn = np.array([(matrix[:,c:c+1].sum() - matrix[c,c]) for c in range(cls_count)]) # false negative, (total of predicted class incorrect)

class_metrics = dict()
all_p, all_r, all_f1 = [], [], []
for c in range(cls_count):
    recall = tp[c] / (tp[c] + fn[c])
    precision = tp[c] / (tp[c] + fp[c])
    f1_score = (2 * precision * recall) / (precision + recall)
    class_metrics[cls_names[c]] = np.array([precision, recall, f1_score])
    all_p.append(precision)
    all_r.append(recall)
    all_f1.append(f1_score)

np.mean(all_p)
np.mean(all_r)
np.mean(all_f1)

# Compare to results here https://arxiv.org/pdf/2303.13827.pdf