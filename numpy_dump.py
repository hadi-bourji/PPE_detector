from data_utils.ppe_dataset import PPE_DATA
import numpy as np
from tqdm import tqdm

dataset = PPE_DATA(data_path = "data/data_556_s.txt", apply_transforms = False)
total = np.empty((0, 3, 640, 640))
for i in tqdm(range(len(dataset)), desc="Loading Images"):
    img, _ = dataset[i]
    img = img.numpy()
    total = np.concatenate((total, img[np.newaxis, ...]), axis = 0)
np.save("calib_data.npy", total)
    