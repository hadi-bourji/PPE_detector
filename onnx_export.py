import torch
from yolox.model import create_yolox_s, create_yolox_m
from yolox.handle_weights import load_pretrained_weights

# # model_int8.eval()
def onnx_export(model, output_path):
    dummy_input = torch.randn(1, 3, 640, 640)
    torch.onnx.export(
        model,                        # the model to export
        dummy_input,                  # one example input (or a tuple)
        f"{output_path}.onnx",                 # where to save the ONNX file
        opset_version=14,             # ONNX opset version to target
        do_constant_folding=False,     # whether to apply optimizations for constants
        input_names=['input'],        # model’s input names
        output_names=['output'],      # model’s output names
    )
    print(f"wrote to: {output_path}")

if __name__ == "__main__":
    
    num_classes = 6
    model = create_yolox_s(num_classes=num_classes)
    weights = "model_checkpoints/yolox_s_uaFalse_transformsTrue_nc6_ep300_bs8_lr1e-04_wd5e-04_07-29_10.pth"

    model = load_pretrained_weights(model, weights, remap=False)
    model.eval()
    onnx_export(model, "yolox_s_best")
