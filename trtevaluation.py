import tensorrt as trt
import torch
from trtinference import build_engine
from data_utils.ppe_dataset import PPE_DATA
from data_utils.metrics import calculate_mAP
from yolox.model import create_yolox_m
from yolox.test_weights import load_pretrained_weights
from tqdm import tqdm

def evaluate_trt(engine_path=None):
    onnx_path_ppe = "./onnx/dummy.onnx"
    if engine_path is None:
        engine_path = "./engines/dummy.plan"
    device = "cuda"
    max_gt = 30
    num_classes = 6

    engine = build_engine(onnx_path_ppe, engine_path, half_precision=False)
    input_shape = (1,3,640, 640)
    output_shape = (1, 8400, 11)
    context = engine.create_execution_context()

    inp_torch  = torch.empty(size=input_shape, dtype=torch.float32, device="cuda")
    outputs  = torch.empty(size=output_shape, dtype=torch.float32, device="cuda")
    d_in = inp_torch.data_ptr()
    d_out = outputs.data_ptr()

    context.set_tensor_address("input", d_in)
    context.set_tensor_address("output", d_out)
    stream = torch.cuda.Stream()

    all_img_ids = torch.empty(0, dtype=torch.int64).to(device)
    # the 5 is 4 for bbox and 1 for class
    all_gts = torch.empty(0, max_gt, 5).to(device)
    # 8400 is the number of anchors
    all_preds = torch.empty(0, 8400, 5 + num_classes).to(device)
    val_dataset = PPE_DATA(mode="val")
    for i in range(len(val_dataset)):
        img_id, img, label = val_dataset[i]
        img_id, label = img_id.unsqueeze(0), label.unsqueeze(0)
        img_id, label = img_id.to(device), label.to(device)
        img = img.to(device).unsqueeze(0)
        inp_torch.copy_(img)
        with torch.cuda.stream(stream):
            context.execute_async_v3(stream_handle = stream.cuda_stream)
        stream.synchronize()
        
        all_img_ids = torch.cat((all_img_ids, img_id), dim=0)
        all_gts = torch.cat((all_gts, label), dim=0) 
        all_preds = torch.cat((all_preds, outputs.clone()), dim=0)

    map = calculate_mAP(
        all_img_ids,
        all_gts,
        all_preds,
        iou_thresh = 0.5,
        num_classes=num_classes,
    )
    print(map)

def evaluate_torch():

    device = "cuda"
    max_gt = 30
    num_classes = 6

    model_path = "model_checkpoints/yolox_m_uaTrue_nc6_ep200_bs8_lr1e-04_wd5e-04_07-15_19_ce100.pth"
    model = create_yolox_m(num_classes)
    model = load_pretrained_weights(model, model_path, num_classes = num_classes, remap=False)
    model = model.to(device).eval()

    all_img_ids = torch.empty(0, dtype=torch.int64).to(device)
    # the 5 is 4 for bbox and 1 for class
    all_gts = torch.empty(0, max_gt, 5).to(device)
    # 8400 is the number of anchors
    all_preds = torch.empty(0, 8400, 5 + num_classes).to(device)
    val_dataset = PPE_DATA(mode="val")
    for i in range(len(val_dataset)):
        img_id, img, label = val_dataset[i]
        img_id, label = img_id.unsqueeze(0), label.unsqueeze(0)
        img_id, label = img_id.to(device), label.to(device)
        img = img.to(device).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img)
        
        all_img_ids = torch.cat((all_img_ids, img_id), dim=0)
        all_gts = torch.cat((all_gts, label), dim=0) 
        all_preds = torch.cat((all_preds, outputs), dim=0)

    map = calculate_mAP(
        all_img_ids,
        all_gts,
        all_preds,
        iou_thresh = 0.5,
        num_classes=num_classes,
    )
    print(map)

evaluate_trt()
evaluate_trt(engine_path = "./engines/dummyfp32.plan")
evaluate_torch()