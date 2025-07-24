# trt_yolox_rt.py
import os, time, pathlib, cv2, torch, numpy as np, tensorrt as trt
import einops
import torch.nn.functional as F

ONNX_PATH_PPE   = os.path.join("onnx", "dummy.onnx")
ENGINE_PATH_PPE =  os.path.join("engines", "dummy.plan") 
ONNX_PATH_REGULAR = os.path.join("onnx", "yolox_s.onnx")
ENGINE_PATH_REGULAR = os.path.join("engines", "yolox_s.plan")
CONF_TH     = 0.50
IOU_TH      = 0.50
INPUT_SIZE  = 640                       # model expects square 640×640
MAX_OUTPUT_BOXES = 30
print(trt.__version__)

TRT_LOGGER  = trt.Logger(trt.Logger.WARNING)
    
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
        text = f"cell phone {s:.2f}"
        color = cell_phone_color

        cv2.rectangle(n, (x1, y1), (x2, y2), color, 1)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, thickness=1)
        cv2.rectangle(n, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)   # filled bg
        cv2.putText(n, text, (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    return n

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
    # THIS CURRENTLY EXPECTS IMAGES TO ALREADY BE UNDER 640x640
    # in order to shrink using this function uncomment the scale = ... lines

    # Preprocess the frame for YOLOX
    img = einops.rearrange(frame, 'h w c -> c h w')  # Change to CHW format
    img = torch.from_numpy(img).float().to(device)

    new_height, new_width = img.shape[1:]
    # scale = min(output_size / width, output_size / height)
    # new_width, new_height = int(width * scale), int(height * scale)
    # img = F.interpolate(img.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False)

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
    Vectorized IoU for two -sets- of axis-aligned boxes.
    boxes{1,2}: (N, 4) or (M, 4) in XYXY format (x1, y1, x2, y2)
    Returns:    (N, M) IoU matrix
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
    Pure-PyTorch Non-Maximum Suppression mirroring
    torchvision.ops.nms(...).

    Args
    ----
    boxes         (Tensor[N,4])  - boxes in (x1, y1, x2, y2) format
    scores        (Tensor[N])     - confidence scores
    iou_threshold (float)         - IoU overlap threshold to suppress

    Returns
    -------
    keep (Tensor[K]) - indices of boxes that survive NMS,
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
def add_nms_trt(network):

    #IMPORTANT
    # This function assumes that sigmoid has already been applied to obj and class_scores, and the network is in the shape (cx,cy,w,h,obj, ...)
    
    strides = trt.Dims([1,1,1])
    starts = trt.dims([0,0,0])
    # There should only be one output for the current network
    output = network.get_output(0)
    network.unmark_output(output)
    bs, num_boxes, temp = output.shape

    # get the boxes from network output
    shapes = trt.Dims([bs, num_boxes, 4])
    boxes = network.add_slice(output, starts, shapes, strides)

    # get the obj from network output
    num_classes = temp - 5
    starts[2] = 4
    shapes[2] = 1
    obj_score = network.add_slice(output, starts, shapes, strides)

    # get the class scores
    starts[2] = 5
    shapes[2] = num_classes
    scores = network.add_slice(output, starts, shapes, strides)

    registry = trt.get_plugin_registry()
    assert(registry)
    creator = registry.get_plugin_creator("EfficientNMS_TRT", "1")
    assert(creator)
    fc = []
    fc.append(trt.PluginField("background_class", np.array([-1],dtype=np.int32), trt.PluginFieldtype.INT32))
    fc.append(trt.PluginField("max_output_boxes", np.array([MAX_OUTPUT_BOXES], dtype=np.int32), trt.PluginFieldType.INT32))
    fc.append(trt.PluginField("score_threshold", np.array([CONF_TH], dtype=np.float32), trt.PluginFieldType.FLOAT32))
    fc.append(trt.PluginField("iou_threshold", np.array([IOU_TH], dtype=np.float32), trt.PluginFieldType.FLOAT32))
    fc.append(trt.PluginField("box_coding", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32))
    fc.append(trt.PluginField("score_activation", np.array([0], dtype=np.int32), trt.PluginFieldType.INT32))
    fc.append(trt.PluginField("class_agnostic", np.array([0], dtype=np.int32), trt.PluginFieldType.INT32))

    fc = trt.PluginFieldCollection(fc)
    nms_layer = creator.create_plugin("nms_layer", fc)
    layer = network.add_plugin_v2([boxes.get_output(0), scores.get_output(0)], nms_layer)
    layer.get_output(0).name = "num"
    layer.get_output(1).name = "boxes"
    layer.get_output(2).name = "scores"
    layer.get_output(3).name = "classes"
    for i in range(4):
        network.mark_output(layer.get_output(i))
    return network

def build_engine(onnx_path: str, engine_path: str):
    if os.path.exists(engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
            return rt.deserialize_cuda_engine(f.read())
    logger = TRT_LOGGER

    builder = trt.Builder(logger)
    network_flags = \
        (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))      # mandatory for ONNX
    #| (1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))          )
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    success = parser.parse_from_file(onnx_path)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))
    
    if not success:
        raise Exception("Onnx parsing failed")

    # network = add_nms_trt(network)

    config = builder.create_builder_config()
    # arbitrary, maybe play with this
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)   # 256 MiB
    # config.set_flag(trt.BuilderFlag.FP16)

    print("Building TensorRT engine …")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine build failed")
    print("Completed build")

    # save .plan
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

# load for inference
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    return engine

def alloc_outputs(engine, context):
    output_buffers, ptrs = {}, {}
    DT = {
        trt.DataType.Float: torch.float32,
        trt.DataType.INT32: torch.int32
    }
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
            continue
        shape = tuple(int(d) for d in context.get_tensor_shape(name))
        t = torch.empty(shape, dtype=DT[engine.get_tensor_dtype(name)], device="cuda")
        output_buffers[name] = t
        ptrs[name] = t.data_ptr()
        assert context.set_tensor_address(name, ptrs[name])
    return output_buffers, ptrs
def main():

    print("building...")
    engine_ppe = build_engine(ONNX_PATH_PPE, ENGINE_PATH_PPE)
    engine_reg = build_engine(ONNX_PATH_REGULAR, ENGINE_PATH_REGULAR)

    context_ppe = engine_ppe.create_execution_context()
    context_reg = engine_reg.create_execution_context()

    # --------------------------------------------------------------------------- #
    # 2.  Allocate CUDA I/O buffers – use torch tensors for convenience           #
    # --------------------------------------------------------------------------- #
    input_shape = (1,3,640,640)
    output_shape_ppe = (1, 8400, 11)
    output_shape_reg = (1, 8400, 85)

    # allocate with torch (pinned host <-> device copies handled by cuda.to_dlpack)
    inp_torch_ppe  = torch.empty(size=input_shape, dtype=torch.float32, device="cuda")
    inp_torch_reg  = torch.empty(size=input_shape, dtype=torch.float32, device="cuda")
    preds_ppe  = torch.empty(size=output_shape_ppe, dtype=torch.float32, device="cuda")
    preds_reg  = torch.empty(size=output_shape_reg, dtype=torch.float32, device="cuda")


    d_in_ppe  = inp_torch_ppe.data_ptr()
    d_out_ppe = preds_ppe.data_ptr()
    d_in_reg  = inp_torch_reg.data_ptr()
    d_out_reg = preds_reg.data_ptr()


    context_ppe.set_tensor_address("input", d_in_ppe)
    context_ppe.set_tensor_address("output", d_out_ppe)
    context_reg.set_tensor_address("input", d_in_reg)
    context_reg.set_tensor_address("output", d_out_reg)
    stream_ppe = torch.cuda.Stream()
    stream_reg = torch.cuda.Stream()

    # --------------------------------------------------------------------------- #
    # 3.  Video loop                                                              #
    # --------------------------------------------------------------------------- #

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera open failed")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv2.namedWindow("YOLO-TensorRT", cv2.WINDOW_NORMAL)
    frame_count, t0 = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_count += 1
        img, _ = process_frame(frame, device="cuda", output_size=INPUT_SIZE)  # cpu tensor
        inp_torch_ppe.copy_(img)                      # H2D via unified memory
        inp_torch_reg.copy_(img)

        # --- inference --------------------------------------------------------- #
        with torch.cuda.stream(stream_ppe):
            context_ppe.execute_async_v3(stream_handle=stream_ppe.cuda_stream)

        with torch.cuda.stream(stream_reg):
            context_reg.execute_async_v3(stream_handle = stream_reg.cuda_stream)

        stream_ppe.synchronize()
        stream_reg.synchronize()
        
        # ---------------------------------------------------------------------- #

        processed_preds_ppe = post_process_img(
            preds_ppe[0], confidence_threshold=CONF_TH, iou_threshold=IOU_TH
        ).cpu().numpy()

        if processed_preds_ppe.any():
            processed_preds_ppe[..., 2] = processed_preds_ppe[..., 2] - 80
            processed_preds_ppe[..., 4] = processed_preds_ppe[..., 4] - 80

        processed_preds_reg = post_process_img(
            preds_reg[0], confidence_threshold=CONF_TH, iou_threshold=IOU_TH
        ).cpu().numpy()
 
        if processed_preds_reg.any():
            processed_preds_reg[..., 2] = processed_preds_reg[..., 2] - 80
            processed_preds_reg[..., 4] = processed_preds_reg[..., 4] - 80

        vis = draw_ppe(frame, processed_preds_ppe)       # draw on original BGR frame
        vis = draw_reg_yolo(frame, processed_preds_reg)

        # vis = vis[pads[0]:vis.shape[0] - pads[1], pads[2]:vis.shape[1] - pads[3]]

        cv2.imshow("YOLO-TensorRT", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):           # q quits
            break

        if frame_count == 60:
            fps = frame_count / (time.time() - t0)
            t0 = time.time()
            frame_count = 0
            print(f"FPS {fps:.2f}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()