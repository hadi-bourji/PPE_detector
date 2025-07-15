import torch
from yolox.model import create_yolox_s
from yolox.test_weights import load_pretrained_weights

model = create_yolox_s(num_classes=4)
weights = "model_checkpoints\\yolox_s_nc4_ep300_bs16_lr1e-04_wd5e-04_07-08_00.pth"
model = load_pretrained_weights(model, weights, remap=False)

model.eval()
dummy_input = torch.randn(1, 3, 640, 640)
torch.onnx.export(
    model,                        # the model to export
    dummy_input,                  # one example input (or a tuple)
    "onnx\\yolox_s_nc4_ep300_bs16_lr1e-04_wd5e-04_07-08_00.onnx",                 # where to save the ONNX file
    opset_version=13,             # ONNX opset version to target
    do_constant_folding=True,     # whether to apply optimizations for constants
    input_names=['input'],        # model’s input names
    output_names=['output'],      # model’s output names
)
print("wrote to: onnx\\yolox_s_nc4_ep300_bs16_lr1e-04_wd5e-04_07-08_00.onnx")

