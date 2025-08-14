from data_utils.ppe_dataset import PPE_DATA  # unused now but kept per your stub
from data_utils.metrics import post_process_img
import os, cv2, torch, numpy as np, shutil
from yolox.test_weights import load_pretrained_weights
from yolox.model import create_yolox_m
from tqdm import tqdm

def label_imgs(img_dir, output_dir, model, device='cuda'):
    os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        f.write(
            "names:\n"
            "  0: Lab-Coat\n"
            "  1: No-Coat\n"
            "  2: Eyewear\n"
            "  3: No-Eyewear\n"
            "  4: Glove\n"
            "  5: No-Glove\n"
            "  6: Cell-Phone\n"
            "path: .\n"
        )
    
    

    def letterbox(img, size=640):
        h0, w0 = img.shape[:2]
        r = min(size / h0, size / w0)
        nw, nh = int(round(w0 * r)), int(round(h0 * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((size, size, 3), dtype=img.dtype)
        dw, dh = (size - nw) // 2, (size - nh) // 2
        canvas[dh:dh+nh, dw:dw+nw] = resized
        return canvas, r, dw, dh


    model.to(device).eval()
    exts = (".jpg", ".jpeg", ".png")

    with torch.no_grad():
        for name in tqdm(sorted(os.listdir(img_dir))):
            if not name.lower().endswith(exts):
                continue

            in_path  = os.path.join(img_dir, name)
            stem     = os.path.splitext(name)[0]
            out_img  = os.path.join(output_dir, "images", "train", name)
            out_lbl  = os.path.join(output_dir, "labels", "train", f"{stem}.txt")

            im0 = cv2.imread(in_path)
            if im0 is None:
                raise RuntimeError(f"Image {in_path} could not be read")
            H0, W0 = im0.shape[:2]

            # Save ORIGINAL image
            if os.path.abspath(in_path) != os.path.abspath(out_img):
                shutil.copy2(in_path, out_img)

            # Inference image
            img640, r, dw, dh = letterbox(im0, 640)
            inp = torch.from_numpy(img640).permute(2,0,1).unsqueeze(0).to(device).float()  # BGR, 0..255

            preds = model(inp)
            det = post_process_img(preds, confidence_threshold=0.50, iou_threshold=0.50)  # (N,6): cls,x1,y1,x2,y2,conf

            # Write labels (empty if none)
            lines = []
            if det is not None and len(det) > 0:
                det = det.detach().cpu().numpy()
                cls = det[:, 0].astype(int)
                x1, y1, x2, y2 = det[:, 1], det[:, 2], det[:, 3], det[:, 4]

                # map back from letterbox coords -> original coords
                x1o = (x1 - dw) / r
                y1o = (y1 - dh) / r
                x2o = (x2 - dw) / r
                y2o = (y2 - dh) / r

                # YOLO cx,cy,w,h normalized to ORIGINAL size
                cx = ((x1o + x2o) / 2.0) / W0
                cy = ((y1o + y2o) / 2.0) / H0
                w  = (x2o - x1o) / W0
                h  = (y2o - y1o) / H0

                for c, xc, yc, ww, hh in zip(cls, cx, cy, w, h):
                    lines.append(f"{int(c)} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

            with open(out_lbl, "w") as f:
                if lines:
                    f.write("\n".join(lines))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Automatically label images using a YOLO model.")
    parser.add_argument("img_dir", type=str, help="Directory containing images to label.")
    parser.add_argument("output_dir", type=str, help="Directory to save labeled images and labels.")
    parser.add_argument("--device", type=str, default='cuda', help="Device to run the model on ('cuda' or 'cpu').")
    
    args = parser.parse_args()

    # Load the model
    weight_path = "model_checkpoints/best_ppe.pth"
    model = create_yolox_m(num_classes = 6)
    load_pretrained_weights(model, weight_path, num_classes=6, remap=False)
    model.eval()

    label_imgs(args.img_dir, args.output_dir, model, device=args.device)
