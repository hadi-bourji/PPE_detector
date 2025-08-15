import torch
from torch.utils.data import DataLoader
from yolox.handle_weights import load_pretrained_weights
from yolox.model import create_yolox_s, create_yolox_l, create_yolox_m
from data_utils.ppe_dataset import PPE_DATA
from yolox.loss import YOLOXLoss
from torch.optim import AdamW
from data_utils.metrics import calculate_mAP
from torch.utils.tensorboard import SummaryWriter
from time import perf_counter
from datetime import datetime
from tqdm import tqdm
import os

def train(num_classes = 4, num_epochs = 50, validate = True, batch_size = 16, max_gt=30, 
          logging = True, device="cuda", lr = 0.001, weight_decay = 0.0005, save_epochs = [50, 100, 200, 250],
          use_amp = True, model_name = "yolox_m", data_name = "", apply_transforms = True):

    today = datetime.today()
    date_str = today.strftime("%m-%d_%H")
    exp_name = f"{model_name}_ua{use_amp}_transforms{apply_transforms}_dn({data_name})_nc{num_classes}_ep{num_epochs}_bs{batch_size}_lr{lr:.0e}_wd{weight_decay:.0e}_{date_str}"
    print(f"Experiment Name: {exp_name}")
    print("using amp: ", use_amp)


    # weight_path = download_weights("yolox", model = "yolox_m")

    if model_name == "yolox_m":
        weight_path = os.path.join("yolox", "yolox_m.pth")
        model = create_yolox_m(num_classes)
    elif model_name == "yolox_s":
        weight_path = os.path.join("yolox", "yolox_s.pth")
        model = create_yolox_s(num_classes)
    else:
        raise Exception("model name must be yolox_m or yolox_s")
    model = load_pretrained_weights(model, weight_path, num_classes= num_classes)

    model.train().to(device)

    for k, v in model.named_parameters():
         if k.startswith("backbone"):
            v.requires_grad = False

    dataset = PPE_DATA(data_path=f"./data/{data_name}", mode="train", max_gt = max_gt, p_mosaic = 1 / batch_size, apply_transforms = apply_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    if validate:
        val_dataset = PPE_DATA(data_path="./data", mode="val")
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    loss_fn = YOLOXLoss(num_classes=num_classes)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(enabled=use_amp)  # for mixed precision training

    if logging:
        writer = SummaryWriter(log_dir=f"./logs/{exp_name}")
        writer.add_text("Hyperparameters", f"num_classes: {num_classes}, num_epochs: {num_epochs}, "
                                            f"batch_size: {batch_size}, max_gt: {max_gt}")

    for epoch in tqdm(range(num_epochs), desc="Training Epochs", unit="epoch"):
        
        # Unfreeze model parameters after 10 epochs
        if epoch == 10:
            for k, v in model.named_parameters():
                if k.startswith("backbone"):
                    v.requires_grad = True

        # on the last 15 epochs just show the regular dataset
        if epoch == num_epochs - 15:
            dataset.transforms = False
        
        running_train_loss = 0.0
        running_train_box_loss = 0.0
        running_train_cls_loss = 0.0
        running_train_obj_loss = 0.0

        for batch in tqdm(dataloader, desc="Training Batches", unit="batch", total = len(dataloader)):
            img, labels = batch
            img = img.to(device)
            labels = labels.to(device)
            # img is shape (batch_size, 3, 640, 640)
            # labels is shape (batch_size, max_gt, 5) where 5 is [c, cx, cy, w, h], all normalized

            # Forward pass
            # output shape: (batch, 8400, 5 + num_classes). Boxes are already decoded to pixel space
            # but left in cx cy format
            with torch.autocast(device_type=device, dtype = torch.float16, enabled=use_amp):
                outputs = model(img)
                loss_dict = loss_fn(outputs, labels)

            total_loss = loss_dict["total_loss"]
            running_train_loss += total_loss.item()
            running_train_box_loss += loss_dict["box_loss"]
            running_train_cls_loss += loss_dict["cls_loss"]
            running_train_obj_loss += loss_dict["obj_loss"]

            # Backward pass and optimization
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if validate:
            with torch.no_grad():
                
                all_img_ids = torch.empty(0, dtype=torch.int64).to(device)
                # the 5 is 4 for bbox and 1 for class
                all_gts = torch.empty(0, max_gt, 5).to(device)
                # 8400 is the number of anchors
                all_preds = torch.empty(0, 8400, 5 + num_classes).to(device)
                running_val_loss = 0.0
                running_val_box_loss = 0.0
                running_val_cls_loss = 0.0
                running_val_obj_loss = 0.0
                for batch in tqdm(val_dataloader, desc="Validation Batches", unit="batch", total=len(val_dataloader)):
                    img_ids, img, labels = batch
                    img_ids, img, labels = img_ids.to(device), img.to(device), labels.to(device)

                    # Forward pass
                    with torch.autocast(device_type=device, dtype = torch.float16, enabled=use_amp):
                        outputs = model(img)
                        val_loss_dict = loss_fn(outputs, labels)
                    val_loss = val_loss_dict["total_loss"]

                    running_val_loss += val_loss
                    running_val_box_loss += val_loss_dict["box_loss"]
                    running_val_cls_loss += val_loss_dict["cls_loss"]
                    running_val_obj_loss += val_loss_dict["obj_loss"]
                    # apply in place sigmoid for mAP calcua
                    outputs[...,4:].sigmoid_()
                    all_img_ids = torch.cat((all_img_ids, img_ids), dim=0)
                    all_gts = torch.cat((all_gts, labels), dim=0)
                    all_preds = torch.cat((all_preds, outputs), dim=0)
            
            mAP = calculate_mAP(
                all_img_ids, 
                all_gts, 
                all_preds, 
                iou_thresh=0.5, 
                num_classes=num_classes,
                writer = writer if logging else None,
                epoch = epoch if logging else 0,
            )
            
        # Compute metrics
        train_loss = running_train_loss / len(dataloader)
        val_loss = running_val_loss / len(val_dataloader)
            
        if logging:

            writer.add_scalar("Total Loss/Train", train_loss, epoch)
            
            writer.add_scalar("Total Loss/Val", val_loss, epoch)
            writer.add_scalar("BCE Loss/Train", running_train_cls_loss / len(dataloader), epoch)
            writer.add_scalar("IoU Loss/Train", running_train_box_loss / len(dataloader), epoch)
            writer.add_scalar("Objectness Loss/Train", running_train_obj_loss / len(dataloader), epoch)
    
            if validate:
                writer.add_scalar("mAP >50", mAP, epoch)
                writer.add_scalar("BCE Loss/Val", running_val_cls_loss / len(val_dataloader), epoch)
                writer.add_scalar("IoU Loss/Val", running_val_box_loss / len(val_dataloader), epoch)
                writer.add_scalar("Objectness Loss/Val", running_val_obj_loss / len(val_dataloader), epoch)
                
        if (epoch + 1) in save_epochs:
            torch.save(model.state_dict(), f"model_checkpoints/{exp_name}_ce{epoch+1}.pth")

    # save written table
    torch.save(model.state_dict(), f"model_checkpoints/{exp_name}.pth")

    if logging:
        writer.close()
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    train(
        num_classes=6,
        num_epochs=200,
        validate=True,
        batch_size=8,
        max_gt=50,
        logging=True,
        device=device,
        lr=0.0001,
        weight_decay=0.0005,
        save_epochs=[50, 100],
        use_amp=True,
        model_name="yolox_m",
        data_name="data_389.txt",
        apply_transforms=False
    )
