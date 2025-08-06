import cv2
import torch
from yolox.test_weights import load_pretrained_weights, download_weights 
from yolox.model import create_yolox_s, create_yolox_l, create_yolox_m
import einops
from data_utils.metrics import post_process_img
import time
import numpy as np
import torch.nn.functional as F


def create_yolo(num_classes=6, device='cuda', weight_path = None, use_pretrained_yolo = False, yolo_type='s'):
    if yolo_type == 's':
        model = create_yolox_s(num_classes=num_classes)
    elif yolo_type == 'l':
        model = create_yolox_l(num_classes=num_classes)
    elif yolo_type == 'm':
        model = create_yolox_m(num_classes=num_classes)
    model = load_pretrained_weights(model, weight_path, num_classes, remap=use_pretrained_yolo)
    model.to(device).eval()
    return model

def process_frame(frame, device = 'cuda', output_size = 640):
    # Preprocess the frame for YOLOX
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
    return img, (pad_top, pad_bottom, pad_left, pad_right)


def draw_ppe(n, outputs):
    edge_colors = [(0,255,0),(0,0,255), (255,255,0), (0,255,255), (255, 0, 255), (180, 180, 255)]
    class_names = ["coat", "no-coat", "eyewear", "no-eyewear", "gloves", "no-gloves"]
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
    return n

def draw_reg_yolo(n, outputs):
    coco_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    cell_phone_color = (255, 255, 255)
    for label in outputs:
        c, x1, y1, x2, y2, s = label
        # print(f"Detected: {c}, {x1}, {y1}, {x2}, {y2}, {s}")
        if c != 67:
            continue

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        text = f"{"cell phone"} {s:.2f}"
        color = cell_phone_color

        cv2.rectangle(n, (x1, y1), (x2, y2), color, 1)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, thickness=1)
        cv2.rectangle(n, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)   # filled bg
        cv2.putText(n, text, (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    return n

start = time.time()
print("Starting video capture...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
mid_val = 128/255.0
# cap.set(cv2.CAP_PROP_BRIGHTNESS, mid_val)
# cap.set(cv2.CAP_PROP_CONTRAST, mid_val)
# cap.set(cv2.CAP_PROP_SATURATION, mid_val)
# cap.set(cv2.CAP_PROP_EXPOSURE, -4)  # Adjust exposure for better lighting
cv2.namedWindow('main', cv2.WINDOW_NORMAL)

weight_path = "model_checkpoints\\best_ppe.pth"
# weight_path = "model_checkpoints\\yolox_s_uaTrue_nc6_ep300_bs8_lr1e-04_wd5e-04_07-17_11_ce200.pth"
# BEST MODEL
# weight_path = "model_checkpoints_old\\yolox_m_nc4_ep300_bs8_lr1e-04_wd5e-04_07-08_10_ce100.pth"

# weight_path = "model_checkpoints\\yolox_s_nc4_ep300_bs16_lr1e-04_wd5e-04_07-08_00.pth"

# no eyewear 200ep
# weight_path = "model_checkpoints\\yolox_s_nc2_ep200_bs32_lr1e-03_wd5e-04_07-03_12.pth"

# regular, good yolo 200ep
# weight_path = 'model_checkpoints\\yolox_s_ep200_bs32_lr1e-03_wd5e-04_07-02_11.pth'
# weight_path = download_weights('yolox_s.pth')
# weight_path = "model_checkpoints\\yolox_m_uaTrue_nc6_ep200_bs8_lr1e-04_wd5e-04_07-15_19_ce150.pth"

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print(f"Video opened successfully after {time.time() - start:.2f} seconds.")

# coat, no-coat, eyewear, no-eyewear
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
frame_count = 0
start = time.time()
num_classes = 6
ppe_yolo = create_yolo(num_classes = num_classes, device = device, weight_path = weight_path, 
                       use_pretrained_yolo = False, yolo_type='m')
reg_yolo = create_yolo(num_classes = 80, device = device, weight_path = "yolox\\yolox_m.pth", 
                       use_pretrained_yolo = True, yolo_type='m')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    frame_count += 1
    img, pads = process_frame(frame, device)

    with torch.no_grad():
        outputs1 = ppe_yolo(img)
        outputs1 = post_process_img(outputs1[0], confidence_threshold=0.5, iou_threshold=0.5, use_batched_nms=False)
        outputs2 = reg_yolo(img)
        outputs2 = post_process_img(outputs2[0], confidence_threshold=0.5, iou_threshold=0.5, use_batched_nms=False)

    img = img.squeeze(0)
    n = einops.rearrange(img, "c h w -> h w c").cpu().numpy().copy().astype(np.uint8)
    outputs1 = outputs1.cpu().numpy()
    outputs2 = outputs2.cpu().numpy()

    n = draw_ppe(n, outputs1)
    n = draw_reg_yolo(n, outputs2)

    if frame_count % 30 == 0:
        elapsed_time = time.time() - start
        fps = frame_count / elapsed_time
        print(f"Avg. FPS: {fps:.2f}")
    
    # Display the frame
    n = n[pads[0]:n.shape[0] - pads[1], pads[2]:n.shape[1] - pads[3]]  # Remove padding
    cv2.imshow('main', n)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
