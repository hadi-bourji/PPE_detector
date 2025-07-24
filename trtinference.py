# trt_yolox_rt.py
import os, time, pathlib, cv2, torch, numpy as np, tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  (initialises CUDA)
import einops
import torch.nn.functional as F

ONNX_PATH   = "dummy.onnx"
ENGINE_PATH = ONNX_PATH.replace(".onnx", ".plan")
CONF_TH     = 0.50
IOU_TH      = 0.50
INPUT_SIZE  = 640                       # model expects square 640×640

TRT_LOGGER  = trt.Logger(trt.Logger.WARNING)

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


def post_process_img(output, confidence_threshold = 0.25, iou_threshold = 0.5) -> torch.Tensor:
    ''' This function expects the output to be in pixel values and sigmoid to already be applied
    to obj and class probabilities.'''
    x1 = output[..., 0:1] - output[..., 2:3] / 2
    y1 = output[..., 1:2] - output[..., 3:4] / 2
    x2 = output[..., 0:1] + output[..., 2:3] / 2
    y2 = output[..., 1:2] + output[..., 3:4] / 2

    # boxes: (batch, num_anchors, 4)
    boxes = torch.cat([x1, y1, x2, y2], dim=-1)

    # (batch, num_anchors, 1)
    obj = output[..., 4:5]
    class_probs = output[..., 5:]

    scores = obj * class_probs
    best_scores, best_class = scores.max(dim=-1)

    mask = best_scores > confidence_threshold
    best_scores = best_scores[mask] 
    best_class = best_class[mask] 
    boxes = boxes[mask]
    keep = nms(boxes, best_scores, iou_threshold = iou_threshold)
    final_boxes = boxes[keep]
    final_classes = best_class[keep]
    final_scores = best_scores[keep]
    # final classes and final scores have shape (num_kept,), so unsqueeze to add the dim 1 again
    predictions = torch.cat((final_classes.unsqueeze(1), 
                             final_boxes, 
                             final_scores.unsqueeze(1)), dim=1)
    return predictions
def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Vectorized IoU for two ‑sets‑ of axis‑aligned boxes.
    boxes{1,2}: (N, 4) or (M, 4) in XYXY format (x1, y1, x2, y2)
    Returns:    (N, M) IoU matrix
    """
    # areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(0)

    # pairwise intersections
    lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)
    wh = (rb - lt).clamp(min=0)                             # width‑height
    inter = wh[..., 0] * wh[..., 1]                         # (N, M)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter + 1e-7)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    Pure‑PyTorch Non‑Maximum Suppression mirroring
    torchvision.ops.nms(...).

    Args
    ----
    boxes         (Tensor[N, 4])  – boxes in (x1, y1, x2, y2) format
    scores        (Tensor[N])     – confidence scores
    iou_threshold (float)         – IoU overlap threshold to suppress

    Returns
    -------
    keep (Tensor[K]) – indices of boxes that survive NMS,
                       sorted in descending score order
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # sort by score descending
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0]              # index of current highest score
        keep.append(i.item())

        if order.numel() == 1:    # nothing left to compare
            break

        # IoU of the current box with the rest
        ious = _box_iou(boxes[i].unsqueeze(0), boxes[order[1:]]).squeeze(0)

        # keep boxes with IoU ≤ threshold
        order = order[1:][ious <= iou_threshold]

    return torch.as_tensor(keep, dtype=torch.long, device=boxes.device)


# --------------------------------------------------------------------------- #
# 1.  Build or load TensorRT engine                                           #
# --------------------------------------------------------------------------- #
def build_engine(onnx_path: str, engine_path: str):
    if os.path.exists(engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
            return rt.deserialize_cuda_engine(f.read())
    logger = TRT_LOGGER

    builder = trt.Builder(logger)
    network_flags = (
        (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))      # mandatory for ONNX
    | (1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))          )
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    success = parser.parse_from_file(onnx_path)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))
    
    if not success:
        raise Exception("Onnx parsing failed")

    config = builder.create_builder_config()
    # arbitrary, maybe play with this
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)   # 256 MiB
    config.set_flag(trt.BuilderFlag.FP16)

    print("Building TensorRT engine …")
    if network is None:
        print("network is none")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine build failed")

    # save .plan
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

# load for inference
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    return engine


print("building...")
engine = build_engine(ONNX_PATH, ENGINE_PATH)
context = engine.create_execution_context()

# --------------------------------------------------------------------------- #
# 2.  Allocate CUDA I/O buffers – use torch tensors for convenience           #
# --------------------------------------------------------------------------- #
input_idx  = engine.get_binding_index("input" if "input" in engine else engine[0])
output_idx = 1 - input_idx
input_shape  = tuple(engine.get_binding_shape(input_idx))   # (1,3,640,640)
output_shape = tuple(engine.get_binding_shape(output_idx))  # (1, n, 85) (dynamic n; use profile)

# allocate with torch (pinned host <-> device copies handled by cuda.to_dlpack)
inp_torch  = torch.empty(size=input_shape, dtype=torch.float32, device="cuda")
out_torch  = torch.empty(size=output_shape, dtype=torch.float32, device="cuda")

# PyCUDA device pointer & size
d_in  = int(inp_torch.data_ptr())
d_out = int(out_torch.data_ptr())

context.set_tensor_address("input", d_in)
context.set_tensor_address("output", d_out)
stream = torch.cuda.current_stream().cuda_stream


# --------------------------------------------------------------------------- #
# 3.  Video loop                                                              #
# --------------------------------------------------------------------------- #
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera open failed")

frame_count, t0 = 0, time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame_count += 1
    img, pads = process_frame(frame, device="cpu", output_size=INPUT_SIZE)  # cpu tensor
    inp_torch.copy_(img)                      # H2D via unified memory

    # --- inference --------------------------------------------------------- #
    context.execute_async_v3(stream)
    torch.cuda.current_stream().synchronize()
    preds = out_torch.clone()                 # TensorRT may reuse buffer – clone first
    # ---------------------------------------------------------------------- #

    preds = post_process_img(
        preds[0], confidence_threshold=CONF_TH, iou_threshold=IOU_TH
    ).cpu().numpy()

    vis = draw_ppe(frame.copy(), preds)       # draw on original BGR frame
    cv2.imshow("YOLO‑TensorRT", vis)
    if cv2.waitKey(1) & 0xFF == 27:           # ESC quits
        break

    if frame_count % 30 == 0:
        fps = frame_count / (time.time() - t0)
        print(f"FPS {fps:.2f}")

cap.release()
cv2.destroyAllWindows()
