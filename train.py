import torch
from torch.utils.data import DataLoader
from yolox.test_weights import download_weights, load_pretrained_weights
from yolox.model import create_yolox_s, create_yolox_l, create_yolox_m
from data_utils.ppe_dataset import PPE_DATA
from yolox.loss import YOLOXLoss
from torch.optim import AdamW
from data_utils.metrics import calculate_mAP
from torch.utils.tensorboard import SummaryWriter
from rich.console import Console, Group
from rich.table import Table
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn, TaskProgressColumn, MofNCompleteColumn, TimeRemainingColumn
from time import perf_counter
from datetime import datetime
import csv

def make_table(metrics_history, num_rows_to_show=25):
    """Create a table from the metrics history"""
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

    for metrics in metrics_history[-num_rows_to_show:]:
        table.add_row(
            str(metrics["epoch"]),
            f"{metrics['train_loss']:.4f}",
            f"{metrics['val_loss']:.4f}",
            f"{metrics['mAP']:.4f}",
            f"{metrics['train_cls_loss']:.4f}",
            f"{metrics['train_box_loss']:.4f}",
            f"{metrics['train_obj_loss']:.4f}",
            f"{metrics['val_cls_loss']:.4f}",
            f"{metrics['val_box_loss']:.4f}",
            f"{metrics['val_obj_loss']:.4f}"
        )
    
    return table



def train(num_classes = 4, num_epochs = 50, validate = True, batch_size = 16, max_gt=30, 
          logging = True, device="cuda", lr = 0.001, weight_decay = 0.0005, save_epochs = [50, 100, 200, 250],
          use_amp = True):

    today = datetime.today()
    date_str = today.strftime("%m-%d_%H")
    exp_name = f"yolox_m_ua{use_amp}_nc{num_classes}_ep{num_epochs}_bs{batch_size}_lr{lr:.0e}_wd{weight_decay:.0e}_{date_str}"
    print(f"Experiment Name: {exp_name}")
    print("using amp: ", use_amp)

    console = Console(record=True, force_terminal=True, width=110, height=1000,log_path=False)        # record=True lets us export later
    metrics_history = []
    num_rows_to_show = 25
    # Build a table once; we’ll *mutate* it inside Live()

    progress = Progress(
    SpinnerColumn(),
    "[progress.description]{task.description}",
    BarColumn(bar_width=None),
    TaskProgressColumn(),                
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    console=console,
    transient=True
    )

    weight_path = download_weights("yolox", model = "yolox_m")
    model = create_yolox_m(num_classes)
    model = load_pretrained_weights(model, weight_path, num_classes)
    model.train().to(device)
    for k, v in model.named_parameters():
       if k.startswith("backbone"):
            v.requires_grad = False

    dataset = PPE_DATA(data_path="./data", mode="train",p_mosaic = 1 / batch_size, apply_transforms = True)
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

    # wrap training loop in live environment to have rich printing
    with Live(console=console, refresh_per_second=2) as live:

        epoch_task = progress.add_task("Epochs", total=num_epochs)
        batch_task_id = None
        t0 = perf_counter()
    
        for epoch in range(num_epochs):
            
            # Unfreeze model parameters after 10 epochs, decrease learning rate
            if epoch == 10:
                for k, v in model.named_parameters():
                    if k.startswith("backbone"):
                        v.requires_grad = True

            # on the last 15 epochs just show the regular dataset
            if epoch == num_epochs - 15:
               dataset.transforms = False
            
            if batch_task_id is not None:          # remove previous batch bar
                progress.remove_task(batch_task_id)
            batch_task_id = progress.add_task(
                f"  [green]train batches", total=len(dataloader)
            )

            running_train_loss = 0.0
            running_train_box_loss = 0.0
            running_train_cls_loss = 0.0
            running_train_obj_loss = 0.0

            for batch in dataloader:
                img, labels = batch
                img = img.to(device)
                labels = labels.to(device)
                # img is shape (batch_size, 3, 640, 640)
                # labels is shape (batch_size, max_gt, 5) where 5 is [c, cx, cy, w, h], all normalized

                # Forward pass
                # output shape: (batch, 8400, 9). Boxes are already decoded to pixel space
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
                progress.advance(batch_task_id)
                progress.advance(epoch_task, advance = 1 / (len(dataloader) + len(val_dataloader)))

            progress.remove_task(batch_task_id)
            if validate:
                batch_task_id = progress.add_task(
                    "[magenta]val batches",
                    completed=0,
                    total=len(val_dataloader),
                )
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
                        progress.advance(batch_task_id)
                        progress.advance(epoch_task, advance = 1 / (len(dataloader) + len(val_dataloader)))
                    progress.remove_task(batch_task_id)
                
                mAP = calculate_mAP(
                    all_img_ids, 
                    all_gts, 
                    all_preds, 
                    iou_thresh=0.5, 
                    num_classes=num_classes
                )
            batch_task_id = None
                
            # Compute metrics
            train_loss = running_train_loss / len(dataloader)
            val_loss = running_val_loss / len(val_dataloader)
            metrics_history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                                    "mAP": mAP,
                                    "train_cls_loss": (running_train_cls_loss / len(dataloader)).item(),
                                    "train_box_loss": (running_train_box_loss / len(dataloader)).item(),
                                    "train_obj_loss": (running_train_obj_loss / len(dataloader)).item(),
                                    "val_cls_loss": (running_val_cls_loss / len(val_dataloader)).item(),
                                    "val_box_loss": (running_val_box_loss / len(val_dataloader)).item(),
                                    "val_obj_loss": (running_val_obj_loss / len(val_dataloader)).item()}
                                    )
            live.update(Group(progress, make_table(metrics_history, num_rows_to_show)), refresh=True)
                
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

    console.print("[bold green] Training Complete")
    console.print(f"Total training time: {perf_counter() - t0:.2f} seconds")
    # save written table
    with open(f"logs_rich/{exp_name}_metrics.csv", "w", newline="") as f:
        richwriter = csv.DictWriter(f, fieldnames=metrics_history[0].keys())
        richwriter.writeheader()
        richwriter.writerows(metrics_history)
    torch.save(model.state_dict(), f"model_checkpoints/{exp_name}.pth")

    if logging:
        writer.close()

def map_test():
    model = create_yolox_s(num_classes=4)
    weight_path = "model_checkpoints/yolox_s_bs16_lr1e-03_wd5e-04_06-27_14_ce200.pth"
    model = load_pretrained_weights(model, weight_path, num_classes=4, remap=False)
    dataset = PPE_DATA(data_path="./data", mode="val")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    all_img_ids = torch.empty(0, dtype=torch.int64).to(device)
    # the 5 is 4 for bbox and 1 for class
    max_gt = 30
    all_gts = torch.empty(0, max_gt, 5).to(device)
    # 8400 is the number of anchors
    num_classes = 4
    all_preds = torch.empty(0, 8400, 5 + num_classes).to(device)
    for ids, imgs, targets in dataloader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        ids = ids.to(device)
        with torch.no_grad():
            outputs = model(imgs)
            outputs[...,4:].sigmoid_()
            all_img_ids = torch.cat((all_img_ids, ids), dim=0)
            all_gts = torch.cat((all_gts, targets), dim=0)
            all_preds = torch.cat((all_preds, outputs), dim=0)
    mAP = calculate_mAP(
        all_img_ids,
        all_gts,
        all_preds,
        iou_thresh=0.5,
        num_classes=num_classes,
        device=device,
        plot_pr = False
    )
    print(f"mAP: {mAP:.4f}")

if __name__ == "__main__":
    # unit_test()
    if torch.cuda.is_available():
        print("Using GPU for training")
        device = "cuda"
    else:
        print("Using CPU for training")
        device = "cpu"
    train(num_classes=4, num_epochs=300, validate=True, batch_size=8, 
          max_gt=30, device=device, logging=False, lr = 0.0001, use_amp = True)
