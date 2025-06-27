import torch
from data_utils.ppe_dataset import PPE_DATA
from yolox.test_weights import load_pretrained_weights
from yolox.model import create_yolox_s
from data_utils.metrics import post_process_img

def main(weight_path, device = "cuda"):
    dataset = PPE_DATA(data_path="./data", mode="val")
    id1, img1, label1 = dataset[1]
    id2, img2, label2 = dataset[11]
    id3, img3, label3 = dataset[22]
    id4, img4, label4 = dataset[28]
    id5, img5, label5 = dataset[34]
    ids = [id1, id2, id3, id4, id5]
    images = [img1, img2, img3, img4, img5]
    model = create_yolox_s(num_classes=4)
    model = load_pretrained_weights(model, weight_path, num_classes=4, remap = False)
    model.eval().to(device)
    for id, img in zip(ids, images):
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img)
        processed_preds = post_process_img(outputs[0], confidence_threshold=0.25, iou_threshold=0.5)
        PPE_DATA.show_img(img.squeeze(0), processed_preds, f"img{id}.png", 
                          rect_coords_centered=True,
                          normalized = False,
                          show_conf_score=True)
    print(f"successfully saved {len(ids)} images with predictions.")
if __name__ == "__main__":
    main("model_checkpoints/yolox_s_ep50_bs16_lr1e-03_wd5e-04_06-27_09.pth", device = "cuda")
        