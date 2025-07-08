import cv2
import torch
from yolox.test_weights import load_pretrained_weights, download_weights 
from yolox.model import create_yolox_s, create_yolox_l, create_yolox_m
import einops
from data_utils.metrics import post_process_img
import time
import numpy as np
import torch.nn.functional as F
start = time.time()
print("Starting video capture...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print(f"Video opened successfully after {time.time() - start:.2f} seconds.")

# coat, no-coat, eyewear, no-eyewear
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

num_classes = 4

#PPE 1000 300 ep:
weight_path = "model_checkpoints\\yolox_m_nc4_ep300_bs8_lr1e-04_wd5e-04_07-08_10_ce100.pth"
# weight_path = "model_checkpoints\\yolox_s_nc4_ep300_bs16_lr1e-04_wd5e-04_07-08_00.pth"

# no eyewear 200ep
# weight_path = "model_checkpoints\\yolox_s_nc2_ep200_bs32_lr1e-03_wd5e-04_07-03_12.pth"

# regular, good yolo 200ep
# weight_path = 'model_checkpoints\\yolox_s_ep200_bs32_lr1e-03_wd5e-04_07-02_11.pth'
# weight_path = download_weights('yolox_s.pth')
model = create_yolox_m(num_classes)
model = load_pretrained_weights(model, weight_path, num_classes, remap = False)
model.to(device).eval()

def process_frame(frame, device = 'cuda', output_size = 640):
    # Preprocess the frame for YOLOX
    # opencv reads with a batch dimension?
    img = einops.rearrange(frame, 'h w c -> c h w')  # Change to CHW format
    img = torch.from_numpy(img).float().to(device)

    height, width = img.shape[1:]
    scale = min(output_size / width, output_size / height)

    new_width, new_height = int(width * scale), int(height * scale)
    img = F.interpolate(img.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False)

    # pad with grey (114, 114, 114), not normalized
    pad_top = (output_size - new_height) // 2
    pad_bottom = output_size - new_height - pad_top
    pad_left = (output_size - new_width) // 2
    pad_right = output_size - new_width - pad_left        

    img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), value = 114.0)
    return img

frame_count = 0
start = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    frame_count += 1
    img = process_frame(frame, device)
    with torch.no_grad():
        outputs = model(img)
        outputs = post_process_img(outputs[0], confidence_threshold=0.5, iou_threshold=0.5)
    img = img.squeeze(0)
    n = einops.rearrange(img, "c h w -> h w c").cpu().numpy().copy().astype(np.uint8)
    edge_colors = [(0,255,0),(0,0,255), (255,0,0), (0,255,255)]
    class_names = ["coat", "no-coat", "eyewear", "no-eyewear"]
    outputs = outputs.cpu().numpy()
    for label in outputs:
        c, x1, y1, x2, y2, s = label
        # print(f"Detected: {c}, {x1}, {y1}, {x2}, {y2}, {s}")
        if c == -1:
            continue

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        text = f"{class_names[int(c.item())]} {s:.2f}"
        color = edge_colors[int(c.item())]

        cv2.rectangle(n, (x1, y1), (x2, y2), color, 1)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, thickness=1)
        cv2.rectangle(n, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)   # filled bg
        cv2.putText(n, text, (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        # cv2.imwrite(f'output_image{time.time()}.jpg', n)
         # Process outputs (this part depends on your model's output format)
    # For example, you might want to apply a threshold and draw bounding boxes
    if frame_count % 30 == 0:
        elapsed_time = time.time() - start
        fps = frame_count / elapsed_time
        print(f"Avg. FPS: {fps:.2f}")
    # Display the frame
    cv2.imshow('Frame', n)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break