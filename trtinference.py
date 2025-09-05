# trt_yolox_rt.py
import os, time, pathlib, cv2, torch, numpy as np, tensorrt as trt
import einops
import torch.nn.functional as F
from data_utils.metrics import post_process_img

ONNX_PATH_PPE   = os.path.join("onnx", "ft6.onnx")
ENGINE_PATH_PPE =  os.path.join("engines", "ft6.plan") 
ONNX_PATH_REGULAR = os.path.join("onnx", "yolox_s.onnx")
ENGINE_PATH_REGULAR = os.path.join("engines", "yolox_m_int8+fp16.plan")
CONF_TH     = 0.50
IOU_TH      = 0.50
INPUT_SIZE  = 640                       # model expects square 640×640
MAX_OUTPUT_BOXES = 30
print(trt.__version__)

TRT_LOGGER  = trt.Logger(trt.Logger.WARNING)
    
def draw_reg_yolo(n, outputs):
    """
    Draw COCO ``cell phone`` (class id 67) detections onto an image.
    """
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
    color = (68,182,235)
    outputs = outputs[outputs[:,0] == 67]

    for label in outputs:
        c, x1, y1, x2, y2, s = label
        # print(f"Detected: {c}, {x1}, {y1}, {x2}, {y2}, {s}")

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        text = f"cell phone {s:.2f}"

        cv2.rectangle(n, (x1, y1), (x2, y2), color, 1)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, thickness=1)
        cv2.rectangle(n, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)   # filled bg
        cv2.putText(n, text, (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA, )
    return n

def draw_ppe(n, outputs):
    """
    Draw PPE detections (custom 6-class head) onto an image.
    """
    edge_colors = [(255,255,255),(0,0,255), (255, 0, 255),(0,255,255), (255,255,0), (180, 180, 255)]
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

    #new_height, new_width = img.shape[1:]
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
    return img.squeeze(0), (pad_top, pad_bottom, pad_left, pad_right)

# 1.  Build or load TensorRT engine                                           #
# --------------------------------------------------------------------------- #
def build_engine(onnx_path: str, engine_path: str, precision = ["fp16"]) -> trt.ICudaEngine:
    """
    Build (or load) a TensorRT engine from an ONNX model.
    """
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

    config = builder.create_builder_config()
    # arbitrary, maybe play with this
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)   # 256 MiB
    if "fp16" in precision:
        print("building with FP16")
        config.set_flag(trt.BuilderFlag.FP16)
    if "int8" in precision:
        print("building with int8")
        config.set_flag(trt.BuilderFlag.INT8)
    if "fp8" in precision:
        print("building with fp8")
        config.set_flag(trt.BuilderFlag.FP8)

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

def main():

    print("building...")
    engine_ppe = build_engine(ONNX_PATH_PPE, ENGINE_PATH_PPE, precision=["fp16"])
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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow("trt", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("trt",
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN)

    frame_count, t0 = 0, time.time()

    run=0
    count=0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        count+=1

        frame_count += 1
        frame = cv2.flip(frame, 0)
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
            # print("HELLO")
            processed_preds_ppe[..., 2] = processed_preds_ppe[..., 2] - 140
            processed_preds_ppe[..., 4] = processed_preds_ppe[..., 4] - 140
            processed_preds_ppe[..., 1:5] *= 2

        processed_preds_reg = post_process_img(
            preds_reg[0], confidence_threshold=0.25, iou_threshold=IOU_TH
        ).cpu().numpy()
 
        if processed_preds_reg.any():
            processed_preds_reg[..., 2] = processed_preds_reg[..., 2] - 140
            processed_preds_reg[..., 4] = processed_preds_reg[..., 4] - 140
            processed_preds_reg[..., 1:5] *= 2

        # if count % 60 == 0:
        #    cv2.imwrite(f"imgs4/{count}_img.jpg", frame)
        vis = draw_ppe(frame, processed_preds_ppe)       # draw on original BGR frame
        vis = draw_reg_yolo(frame, processed_preds_reg)

        cv2.imshow("trt", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):           # q quits
            break

        if frame_count == 600:
            fps = frame_count / (time.time() - t0)
            t0 = time.time()
            frame_count = 0
            print(f"FPS {fps:.2f}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
