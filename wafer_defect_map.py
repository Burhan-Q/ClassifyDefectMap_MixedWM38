# print('go')

import numpy as np
from pathlib import Path
import yaml
# from PIL import Image
import cv2 as cv
from sklearn.model_selection import train_test_split

# cd "Q:\ML_data\MixedWM38\2023-05-02_MixedWM38_Kaggle"
# YOLO refernce https://docs.ultralytics.com/tasks/classify/#dataset-format

def prev_im(image, string=""):
    cv.imshow(string,image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def gen_gray_img(map_data:np.ndarray):
    """
    Usage
    ---
    Generate gray-scale image of wafer data.

    Parameters
    ---
    map_data : ``numpy.ndarray``
        Wafer data map with expected pixel values of (0, 1, 2); where `0` indicates non-wafer pixels, `1` indicates 'good' wafer pixels, and `2` indicates 'bad' wafer pixels.

    Returns
    ---
    Gray-scale image as ``numpy.ndarray`` with pixel values (0, 128, 255) corresponding to input pixels (0, 1, 2).
    
    """
    assert all([n in np.unique(map_data) for n in [0,1,2]]), f"Map values do not match expected values (0, 1, 2)"
    assert all([n in range(3) for n in np.unique(map_data)]), f"One or more map values outside of expected value range (0, 1, 2)"

    gray = np.copy(map_data)
    gray[gray == 1] += 127 # good location
    gray[gray == 2] += 253 # bad location

    return gray.astype(np.uint8)

def gray_to_color(gray_map:np.ndarray,
                  bad_clr:tuple=(255,255,0),
                  good_clr:tuple=(25,102,255)):
    """
    Usage
    ---
    Convert gray-scale wafer image map to color map using `good_clr` and `bad_clr` BGR colors

    Parameters
    ---
    gray_map : ``numpy.ndarray``
    bad_clr : ``tuple`` optional,
        BGR color values to use for 'bad' pixels, `default=(255,255,0)` "Pumpkin"
    good_clr : ``tuple`` optional,
        BGR color values to use for 'good' pixels, `default=(25,102,255)` "Aqua"

    Returns
    ---
    BGR color image of wafer map, using `good_clr` and `bad_clr` pixel values

    """
    assert all([n in np.unique(gray_map) for n in [0,128,255]]), f"Gray scale values do not match expected values (0, 128, 255)"

    color_map = cv.cvtColor(np.copy(gray_map),cv.COLOR_GRAY2BGR) # B, G, R
    
    for d in range(color_map.shape[-1]):
            color_map[:,:,d][color_map[:,:,d] == 255] = bad_clr[d]
            color_map[:,:,d][color_map[:,:,d] == 128] = good_clr[d]
    
    return color_map

# Load data
file = list(Path(".").glob("*.npz"))[0]
data = np.load(file)

# Seperate map data and labels
imgs = data['arr_0'] # shape (38015, 52, 52)
lbls = data['arr_1'] # shape (38015, 8)

# Get unique labels
unique_lbls = np.unique(lbls,axis=0)

# Load encoding file and create string labels
encodes = list(Path(".").glob("*.yaml"))[0]
with open(encodes,'r') as enc:
    encd = yaml.safe_load(enc)

str_lbls = [str(l) for l in lbls]

# Fix imgs with more than 3 pixel values
# should ONLY have 0,1,2 for values
# see https://github.com/Junliangwangdhu/WaferMap/issues/2
for im in imgs:
    val = np.unique(im)
    if len(val) > 3:
        im[im == 3] = 2

# Train-Validation split
X, y = imgs, [list(encd.keys())[list(encd.values()).index(k.tolist())] for k in lbls]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

# Review counts if needed
train_lbl_counts = {k:v for k,v in zip(*np.unique(y_train,return_counts=True))}
test_lbl_counts = {k:v for k,v in zip(*np.unique(y_test,return_counts=True))}

# Export color images to train or valid directory
train_dir = './data/train'
test_dir = './data/valid'

# Train
for ct, t in enumerate(X_train):
    cls_dir = f"{train_dir}/{y_train[ct]}"
    if not Path(cls_dir).exists():
        _ = Path.mkdir(Path(cls_dir))
    else:
        pass
    filename = f"{cls_dir}/{ct}.png"
    color = gray_to_color(gen_gray_img(t))
    stride_szd = cv.resize(np.copy(color),(64,64),interpolation=cv.INTER_CUBIC)
    _ = cv.imwrite(filename,stride_szd)

# Validation
for cval,val in enumerate(X_test):
    cls_dir = f"{test_dir}/{y_test[cval]}"
    if not Path(cls_dir).exists():
        _ = Path.mkdir(Path(cls_dir))
    else:
        pass
    filename = f"{cls_dir}/{cval}.png"
    color = gray_to_color(gen_gray_img(val))
    stride_szd = cv.resize(np.copy(color),(64,64),interpolation=cv.INTER_CUBIC)
    _ = cv.imwrite(filename,stride_szd)

# Generate color images for ALL wafer maps and save images to labeled directory
for k,v in encd.items():
    # Make directory using 'k'
    _ = Path.mkdir(Path(f'./wafers/{k}')) if not Path(f'./wafers/{k}').exists() else None

    # Get indices of matching groups
    values, *_ = np.where(np.array(str_lbls) == str(v).replace(',',''))

    for idx in values:
        # Create gray scale image
        gray = gen_gray_img(imgs[idx])
    
        # Convert to Color
        color = gray_to_color(gray)
        
        # Resize for YOLO model, (64 x 64); multiple of model stride 32
        stride_szd = cv.resize(np.copy(color),(64,64),interpolation=cv.INTER_CUBIC)
        _ = cv.imwrite(f'./wafers/{k}/' + str(idx)+'.png', stride_szd)

