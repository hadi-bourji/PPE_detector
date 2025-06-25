import torch
from torchvision.ops import batched_nms
from torch.utils.data import DataLoader
from yolox.test_weights import download_weights, load_pretrained_weights
from yolox.model import create_yolox_s
from data_utils.ppe_dataset import PPE_DATA
from yolox.loss import YOLOXLoss
from torch.optim import AdamW
from data_utils.metrics import calculate_mAP
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def train(num_classes = 4, num_epochs = 50, validate = True, batch_size = 16, max_gt=30, 
          logging = True):
    # dataset and viz functions are only configured for 4 classes anyway
    weight_path = download_weights("yolox/yolox_s.pth")
    model = create_yolox_s(num_classes)
    model = load_pretrained_weights(model, weight_path, num_classes)
    model.train().cuda()

    dataset = PPE_DATA(data_path="./data", mode="train")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    if validate:
        val_dataset = PPE_DATA(data_path="./data", mode="val")
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    loss_fn = YOLOXLoss(num_classes=num_classes)
    # TODO try out weight decay and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)

    if logging:
        writer = SummaryWriter(log_dir="./logs")
        writer.add_text("Hyperparameters", f"num_classes: {num_classes}, num_epochs: {num_epochs}, "
                                            f"batch_size: {batch_size}, max_gt: {max_gt}")
        

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        running_train_loss = 0.0
        model.train()

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            img, labels = batch
            img = img.cuda()
            labels = labels.cuda()

            # Forward pass
            # output shape: (batch, 8400, 9)
            outputs = model(img)
            loss_dict = loss_fn(outputs, labels)
            total_loss = loss_dict["total_loss"]
            running_train_loss += total_loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if validate:
            model.eval()
            with torch.no_grad():
                all_img_ids = torch.empty(0, dtype=torch.int64).cuda()
                all_gts = torch.empty(0, max_gt, 5).cuda()
                all_preds = torch.empty(0, 8400, 6).cuda()
                running_val_loss = 0.0
                for batch in tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
                    img_ids, img, labels = batch
                    img = img.cuda()
                    labels = labels.cuda()

                    # Forward pass
                    outputs = model(img)
                    val_loss = loss_fn(outputs, labels)
                    running_val_loss += val_loss.item()
                    all_img_ids = torch.cat((all_img_ids, img_ids), dim=0)
                    all_gts = torch.cat((all_gts, labels), dim=0)
                    all_preds = torch.cat((all_preds, outputs), dim=0)

            mAP = calculate_mAP(
                all_img_ids, 
                all_gts, 
                all_preds, 
                IoU_thresh=0.5, 
                num_classes=num_classes
            )
            print(f"mAP: {mAP:.4f}")

        # Compute metrics
        train_loss = running_train_loss / len(dataloader)
        val_loss = running_val_loss / len(val_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}")
        if logging:
            
            writer.add_scalar("Total Loss/Train", train_loss, epoch)
            
            writer.add_scalar("Total Loss/Val", val_loss, epoch)
            if validate:
                writer.add_scalar("mAP >50", mAP, epoch)
    writer.close()
if __name__ == "__main__":
    train(num_classes=4, num_epochs=50, validate=True, batch_size=16, max_gt=30)
