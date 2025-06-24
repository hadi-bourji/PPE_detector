import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import cv2
import einops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Set matplotlib to use non-interactive backend to avoid Qt issues
import matplotlib
matplotlib.use('Agg')

class PPE_DATA(Dataset):
    def __init__(self, data_path: str = "./data", mode="train"):
        # read file names from train.txt or validation.txt file
        if mode == "train":
            self.file_names = np.loadtxt(f"{data_path}/train.txt", dtype=str)
        elif mode == "validation":
            self.file_names = np.loadtxt(f"{data_path}/validation.txt", dtype=str)
        else:
            raise Exception("Invalid Mode Entered")

    def load_and_resize_img(self, img, labels,  output_size = 640):

        img = torch.from_numpy(img).float() #/ 255.0 # normalize the image
        img = einops.rearrange(img, "h w c -> c h w")
        height, width = img.shape[1:]
        scale = min(output_size / width, output_size / height)

        new_width, new_height = int(width * scale), int(height * scale)
        img = F.interpolate(img.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False)

        # pad with grey (114, 114, 114), normalized
        pad_top = (output_size - new_height) // 2
        pad_bottom = output_size - new_height - pad_top
        pad_left = (output_size - new_width) // 2
        pad_right = output_size - new_width - pad_left        

        img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), value = 114.0)

        # scale labels up to pixel coords, scale by the same refactoring, add padding, then normalize
        if labels.any():
            labels[..., 1] = (labels[..., 1] * width * scale + pad_left) / output_size   # xc
            labels[..., 2] = (labels[..., 2] * height * scale + pad_top ) / output_size   # yc
            labels[..., 3] = (labels[..., 3] * width * scale) / output_size              # w
            labels[..., 4] = (labels[..., 4] * height * scale) / output_size              # h

        return img.squeeze(0), labels


    def __getitem__(self, idx):
        img_path = self.file_names[idx]
        img = cv2.imread(img_path)

        # opencv uses bgr, so switch to standard rgb
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lbl_path = img_path.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        if os.stat(lbl_path).st_size == 0:
            labels = np.empty((0, 5), dtype=np.float32)  # or shape (0,) if you prefer
        else:
            labels = np.loadtxt(lbl_path, dtype=np.float32)
        
        
        img, labels = self.load_and_resize_img(img, labels,)
        return img, labels

    def __len__(self):
        return len(self.file_names)

    @staticmethod
    def show_img(img, labels, output_file="output.png", rect_coords_centered = True):
        # Currently only accepts 3D pytorch tensors, outputs to img file
        if img.ndim == 4:
            img = img.squeeze(0)
        if img.shape[0] == 3:
            n = einops.rearrange(img, "c h w -> h w c").cpu().numpy()
        else:
            n = img.cpu().numpy()

        fig, ax = plt.subplots()
        # use this to draw different colors for each label
        edge_colors = ['r', 'g', 'b', 'y']
        #TODO does this always work?
        if labels.ndim == 3:
            labels = labels.squeeze(0)
            
        for label in labels:
            if rect_coords_centered:
                c, x, y, w, h = label

                x *= n.shape[1]
                y *= n.shape[0]
                w *= n.shape[1]
                h *= n.shape[0]

                # yolo format gives x and y as center coordinates, convert to top-left corner
                x = int(x - w / 2)
                y = int(y - h / 2)
            else:
                c, x1, y1, x2, y2 = label
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)
                # edge_colors[int(c.item())]
                print(c)
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        plt.imshow(n)
        plt.axis('off')
        plt.savefig(f"output_images/{output_file}", bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    dataset = PPE_DATA()
    dataset.__getitem__(11)