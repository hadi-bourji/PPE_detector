import torch
from data_utils.ppe_dataset import PPE_DATA
from yolox.test_weights import load_pretrained_weights
from yolox.model import create_yolox_s
from data_utils.metrics import post_process_img

def main(weight_path, device = "cuda", validation = True):
    if validation:
        dataset = PPE_DATA(data_path="./data", mode="val")
        id1, img1, label1 = dataset[1]
        id2, img2, label2 = dataset[11]
        id3, img3, label3 = dataset[22]
        id4, img4, label4 = dataset[28]
        id5, img5, label5 = dataset[34]
        ids = [id1, id2, id3, id4, id5]
        labels = [label1, label2, label3, label4, label5]
    else:
        dataset = PPE_DATA(data_path="./data", mode="train")
        img1, label1 = dataset[0]
        img2, label2 = dataset[16]
        img3, label3 = dataset[47]
        img4, label4 = dataset[98]
        img5, label5 = dataset[245]
        ids = [0, 16, 47, 98, 245]
        labels = [label1, label2, label3, label4, label5]

    images = [img1, img2, img3, img4, img5]
    model = create_yolox_s(num_classes=4)
    model = load_pretrained_weights(model, weight_path, num_classes=4, remap = False)
    if validation:
        model.eval().to(device)
        for id, img in zip(ids, images):
            img = img.unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img)
            processed_preds = post_process_img(outputs[0], confidence_threshold=0.25, iou_threshold=0.5)
            PPE_DATA.show_img(img.squeeze(0), processed_preds, f"val_img{id}_50.png", 
                            rect_coords_centered=False,
                            normalized = False,
                            show_conf_score=True)

    else:
        model.train().to(device)
        for id, img, label in zip(ids, images, labels):
            img = img.unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img)
            outputs[...,4:].sigmoid_()
            processed_preds = post_process_img(outputs[0], confidence_threshold=0.25, iou_threshold=0.5)
            PPE_DATA.show_img(img.squeeze(0), processed_preds, f"train_img{id}_200_fixed.png", 
                            rect_coords_centered=False,
                            normalized = True,
                            show_conf_score=False)
    print(f"successfully saved {len(ids)} images with predictions.")

if __name__ == "__main__":
    main("model_checkpoints/yolox_s_ep50bs16_lr1e-03_wd5e-04_06-30_09.pth", device = "cuda",
          validation = True)
        