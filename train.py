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
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn
from pathlib import Path

def train(num_classes = 4, num_epochs = 50, validate = True, batch_size = 16, max_gt=30, 
          logging = True, device="cuda"):

    console = Console(record=True, force_terminal=True, width=110, height=1000,log_path=False)        # record=True lets us export later
    metrics_history = []                                   # will also dump to JSON/CSV if you like
    
    # Build a table once; we’ll *mutate* it inside Live()
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Epoch", justify="right")
    table.add_column("Train Loss", justify="right")
    table.add_column("Val Loss",   justify="right")
    table.add_column("mAP >50",    justify="right")
    table.add_column("Cls Loss (T)", justify="right")
    table.add_column("IoU Loss (T)", justify="right")
    table.add_column("Obj Loss (T)", justify="right")
    table.add_column("Cls Loss (V)", justify="right")
    table.add_column("IoU Loss (V)", justify="right")
    table.add_column("Obj Loss (V)", justify="right")
    
    progress = Progress(
    SpinnerColumn(),
    "[progress.description]{task.description}",
    BarColumn(bar_width=None),
    TimeElapsedColumn()
    )
    # dataset and viz functions are only configured for 4 classes anyway
    weight_path = download_weights("yolox/yolox_s.pth")
    model = create_yolox_s(num_classes)
    model = load_pretrained_weights(model, weight_path, num_classes)
    model.train().to(device)

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
    epoch_task = progress.add_task("Epochs", num_epochs)
    with Live(table, console=console, refresh_per_second=4):
        for epoch in progress.track(range(num_epochs)):
            
            running_train_loss = 0.0
            running_train_box_loss = 0.0
            running_train_cls_loss = 0.0
            running_train_obj_loss = 0.0
            for batch in dataloader:
                img, labels = batch
                img = img.to(device)
                labels = labels.to(device)
    
                # Forward pass
                # output shape: (batch, 8400, 9)
                outputs = model(img)
                loss_dict = loss_fn(outputs, labels)
                total_loss = loss_dict["total_loss"]
                running_train_loss += total_loss.item()
                running_train_box_loss += loss_dict["box_loss"]
                running_train_cls_loss += loss_dict["cls_loss"]
                running_train_obj_loss += loss_dict["obj_loss"]
    
                # Backward pass and optimization
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
    
            if validate:
                # model.eval()
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
                    for batch in val_dataloader:
                        img_ids, img, labels = batch
                        img_ids, img, labels = img_ids.to(device), img.to(device), labels.to(device)
    
                        # Forward pass
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
                    num_classes=num_classes
                )
                
            progress.advance(epoch_task)
            # Compute metrics
            train_loss = running_train_loss / len(dataloader)
            val_loss = running_val_loss / len(val_dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}")

            table.add_row(
            str(epoch + 1),
            f"{train_loss:.4f}",
            f"{val_loss:.4f}",
            f"{mAP:.4f}",
            f"{running_train_cls_loss / len(dataloader):.4f}",
            f"{running_train_box_loss / len(dataloader):.4f}",
            f"{running_train_obj_loss / len(dataloader):.4f}",
            f"{running_val_cls_loss / len(val_dataloader):.4f}",
            f"{running_val_box_loss / len(val_dataloader):.4f}",
            f"{running_val_obj_loss / len(val_dataloader):.4f}",
            )

            metrics_history.append(
            dict(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                mAP=mAP,
                train_cls=running_train_cls_loss / len(dataloader),
                train_iou=running_train_box_loss / len(dataloader),
                train_obj=running_train_obj_loss / len(dataloader),
                val_cls=running_val_cls_loss / len(val_dataloader),
                val_iou=running_val_box_loss / len(val_dataloader),
                val_obj=running_val_obj_loss / len(val_dataloader)
            )
            )

            
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
                    writer.close()
    export_console = Console(record=True, force_terminal=True, width=110, height=table.row_count+4, log_path=False)
    export_console.print(table)

    Path("logs_rich/final_table.txt").write_text(
        export_console.export_text(clear=False),
        encoding="utf-8"
    )
    log_dir = Path("./logs_rich")
    log_dir.mkdir(exist_ok=True)
    
    console.print(f"[green]\N{check mark} Rich logs saved to {log_dir.absolute()}[/]")
        
    
               
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Using GPU for training")
        device = "cuda"
    else:
        print("Using CPU for training")
        device = "cpu"
    train(num_classes=4, num_epochs=50, validate=True, batch_size=16, max_gt=30, device=device, logging=False)
