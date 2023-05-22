from ultralytics import YOLO
import numpy as np
# print('go')
weights = "Q:/ML_data/MixedWM38/2023-05-02_MixedWM38_Kaggle/wafer_defects/EXP00011/weights/best.pt"
model = YOLO(weights)  # build a new model from YAML
data_dir = "Q:/ML_data/MixedWM38/2023-05-02_MixedWM38_Kaggle/data"

model.val(data=data_dir,device='cpu')

cls_count = model.metrics.confusion_matrix.nc
cls_names = model.names
matrix = model.metrics.confusion_matrix.matrix 

tp, fp = model.metrics.confusion_matrix.tp_fp() # true positive and false positive (total of true class incorrect)
fn = np.array([(matrix[:,c:c+1].sum() - matrix[c,c]) for c in range(cls_count)]) # false negative, (total of predicted class incorrect)

class_metrics = dict()
all_p, all_r, all_f1, all_acc = [], [], [], []
for c in range(cls_count):
    recall = tp[c] / (tp[c] + fn[c])
    precision = tp[c] / (tp[c] + fp[c])
    accuracy = (tp[c] + 0) / (tp[c] + 0 + fp[c] + fn[c]) # zeros are True-Negatives, but there are none for this dataset
    f1_score = (2 * precision * recall) / (precision + recall)
    class_metrics[cls_names[c]] = np.array([precision, recall, f1_score, accuracy])
    all_p.append(precision)
    all_r.append(recall)
    all_f1.append(f1_score)
    all_acc.append(accuracy)

matrix_file = "Q:/ML_data/MixedWM38/2023-05-02_MixedWM38_Kaggle/wafer_defects/EXP00011/confusion.csv"
metrics_file = "Q:/ML_data/MixedWM38/2023-05-02_MixedWM38_Kaggle/wafer_defects/EXP00011/metrics.yaml"
names_file = "Q:/ML_data/MixedWM38/2023-05-02_MixedWM38_Kaggle/wafer_defects/EXP00011/class_indices.yaml"
tp_fp_fn_file = "Q:/ML_data/MixedWM38/2023-05-02_MixedWM38_Kaggle/wafer_defects/EXP00011/tp_fp_fn_counts.yaml"

# Sace confusion matrix CSV file
import pandas as pd
_ = pd.DataFrame(matrix).to_csv(matrix_file)

with open(metrics_file,'w') as met:
    met.writelines([(k,v) for k,v in class_metrics.items()])

# Save YAML files
import yaml

metrics_export = {k:{'precision':float(v[0]),'recall':float(v[1]),'f1-score':float(v[2]),'accuracy':float(v[3])} for k,v in class_metrics.items()}
metrics_export.update({'Average':{'precision':float(np.mean(all_p)),
                                  'recall':float(np.mean(all_r)),
                                  'f1-score':float(np.mean(all_f1)),
                                  'accuracy':float(np.mean(all_acc))}})
with open(metrics_file,'w') as met:
    yaml.safe_dump(metrics_export,met)

with open(names_file,'w') as nf:
    yaml.safe_dump(cls_names, nf)

tp_fp_fn_counts = dict()
for i in range(cls_count):
    tp_fp_fn_counts[int(i)] = {'tp':int(tp[i]),'fp':int(fp[i]),'fn':int(fn[i])}

with open(tp_fp_fn_file,'w') as cnt_f:
    yaml.safe_dump(tp_fp_fn_counts,cnt_f)



# Generate Tensorboard Visualizations
## NOTE tested and worked 2023-05-16

# model = YOLO(weights) # reload here if needed

## reference https://github.com/ultralytics/yolov5/blob/master/utils/loggers/__init__.py
## https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/utils/callbacks/tensorboard.py
## https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html#inspect-the-model-using-tensorboard
from ultralytics.yolo.utils.torch_utils import de_parallel
import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("Q:/ML_data/MixedWM38/2023-05-02_MixedWM38_Kaggle/wafer_defects/EXP00011")
img_size = (64,64)
im = torch.zeros((1, 3, *img_size)) # loads to CPU, model should also be on CPU

## Generate Model Inspection graph 
### This command might generate an error, but the graph did still save
writer.add_graph(torch.jit.trace(de_parallel(model),im,strict=False),[])
writer.close()

## Tensorboard Projector
### NOTE tested and worked 2023-05-16
### https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html#adding-a-projector-to-tensorboard
### https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_embedding

from pathlib import Path
import numpy as np
from collections import Counter

n_imgs = 400 # divisible evenly by len(classes)
classes = [
    "Center",
    "Donut",
    "Edge_Ring",
    "Loc",
    "Near_Full",
    "Normal",
    "Random",
    "Scratch"] # use base classes

im_path = "Q:/ML_data/MixedWM38/2023-05-02_MixedWM38_Kaggle/wafers"
class_imgs = [img for img in Path(im_path).rglob("*.png") if img.parent.name in classes]
# proj_labels = [img.parent.name for img in proj_imgs]
c = Counter()
proj_imgs = list()
proj_labels = list()

# Randomly select even number of examples
perm_imgs = np.random.permutation(np.array(class_imgs))
max_count = n_imgs / len(classes)

while any([c[m] != max_count for m in c]) or len(c) == 0:
    num = np.random.choice(np.arange(len(class_imgs)))
    choice = perm_imgs[num]

    if (c[choice.parent.name] < max_count) and choice is not None:
        proj_imgs.append(choice)
        proj_labels.append(choice.parent.name)
        c[choice.parent.name] += 1
    elif (c[choice.parent.name] == max_count):
        choice = None
        pass
    elif choice is None and all([c[m] == max_count for m in c]):
        print("complete")
        break
    elif choice is None and any([c[m] != max_count for m in c]):
        print("continuing")
        continue
    else:
        print("else")
        break

import cv2 as cv
import torch

im_arr = np.array([cv.imread(img.as_posix()) for img in proj_imgs])
im_shape = im_arr[0].shape
lbl_arr = np.array(proj_labels)

features = im_arr.reshape(-1, np.product(im_shape))

labl_img = torch.from_numpy(np.transpose(im_arr,(0,3,1,2))) # order (N,C,H,W) that add_embeddings() expects, MUST be torch.Tensor

writer.add_embedding(features,
                     metadata=lbl_arr,
                     label_img=labl_img)
writer.close()
