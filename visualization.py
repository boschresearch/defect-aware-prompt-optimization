import cv2
import os
import numpy as np
from PIL import Image

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)

def visualizer(paths, anomaly_map, img_size, save_path, cls_name):
    for idx, path in enumerate(paths):
        cls = path.split('/')[-2]
        filename = path.split('/')[-1]
        with Image.open(path) as img:
            resized_img =  img.resize((img_size, img_size))
        resized_img = np.array(resized_img)

        vis = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)  # RGB
        mask = normalize(anomaly_map[idx])
        vis = apply_ad_scoremap(vis, mask)
        save_vis = os.path.join(save_path, 'imgs', cls_name[0], cls)
        if not os.path.exists(save_vis):
            os.makedirs(save_vis)
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR   
        cv2.imwrite(os.path.join(save_vis, filename), vis)

def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return ((alpha * np_image) + ((1 - alpha) * scoremap)).astype(np.uint8)

