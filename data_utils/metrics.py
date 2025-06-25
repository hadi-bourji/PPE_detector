import torch
import einops
from torchvision.ops import batched_nms

# pred: (6,) gt: (num_gt, 5). For both, 1:5 are box coordinates
def pairwise_iou(pred, gt):

    pcx, pcy, pw, ph = pred[1:]
    px1, py1 = pcx - pw / 2, pcy - ph / 2
    px2, py2 = pcx + pw / 2, pcy + ph / 2

    gcx = gt[:, 1]
    gcy = gt[:, 2]
    gw = gt[:, 3]
    gh = gt[:, 4]
    gx1, gy1 = gcx - gw / 2, gcy - gh / 2
    gx2, gy2 = gcx + gw / 2, gcy + gh / 2
    # calculate intersection
    inter_x1 = torch.max(px1, gx1)
    inter_y1 = torch.max(py1, gy1)
    inter_x2 = torch.min(px2, gx2)
    inter_y2 = torch.min(py2, gy2)
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h
    # calculate union
    pred_area = pw * ph
    gt_area = gw * gh
    union_area = pred_area + gt_area - inter_area
    iou = inter_area / (union_area + 1e-7)
    return iou

def calculate_AP_per_class(gt, preds, gt_to_img, preds_to_img, iou_thresh):

    tp = torch.zeros(preds.shape[0], dtype=torch.float32)
    fp = torch.zeros(preds.shape[0], dtype=torch.float32)
    precision = torch.zeros(preds.shape[0], dtype=torch.float32)
    recall = torch.zeros(preds.shape[0], dtype=torch.float32)
    num_gt = gt.shape[0]
    gt_matched = torch.zeros(gt.shape[0], dtype = torch.bool)
    for i, pred in enumerate(preds):
        # index into preds_to img to find which image this prediction is a part of,
        # then extract all ground truth boxes from this image to compute IoU
        pred_img = preds_to_img[i]
        img_mask = (gt_to_img == pred_img) & (~gt_matched)
        same_img_gt = gt[img_mask]
        if same_img_gt.shape[0] == 0:
            fp[i] = 1
            continue

        pairwise_ious = pairwise_iou(pred, same_img_gt)
        # find the best IoU match
        best_iou, best_gt_idx = torch.max(pairwise_ious, dim=0)
        if best_iou >= iou_thresh:
            # mark this ground truth box as used
            gt_matched_idx = torch.nonzero(img_mask)[best_gt_idx] 
            gt_matched[gt_matched_idx] = True
            tp[i] = 1
        else:
            fp[i] = 1
    # calculate precision and recall
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-9)
    recall = tp_cumsum / (float(num_gt) + 1e-9)

    # how precision recall curve should be calculated, according to chat
    mrec = torch.cat([torch.tensor([0.], device=recall.device),
                  recall,
                  torch.tensor([1.], device=recall.device)])

    mpre = torch.cat([torch.tensor([0.], device=precision.device),
                  precision,
                  torch.tensor([0.], device=precision.device)])
    # reverse, cumulative max, then flip back – vectorised, no Python loop
    mpre = torch.flip(torch.cummax(torch.flip(mpre, dims=[0]), dim=0).values,
                  dims=[0])
    chg = (mrec[1:] != mrec[:-1]).nonzero(as_tuple=False).squeeze(1)
    ap  = torch.sum((mrec[chg + 1] - mrec[chg]) * mpre[chg + 1])
    return ap

# gt is a list of the ground truth boxes, preds is a list of predicted boxes+confidence
def calculate_mAP(img_ids: torch.Tensor, gts: torch.Tensor, preds: torch.Tensor, num_classes = 4, iou_thresh = 0.5):

    # a mapping from ground truth boxes to their img id
    # used to ensure predictions are being compared only to the same image

    # shape (batch, max_gt)
    gt_mask = gts[:, :, 0] >= 0
    gt_to_img = torch.repeat_interleave(img_ids, gt_mask.sum(1), dim=0)
    true_gt = gts[gt_mask]
    # ngt = num ground truth, npk = num predictions kept

    # refactor this to work with the way batch_nms returns indices 
    batch_len = img_ids.shape[0]
    preds_to_img = torch.empty(0)
    final_preds = torch.empty(0, 6)
    for batch_idx in range(batch_len):
        # add img id to the beginning of each ground truth box
        img_id = img_ids[batch_idx]
        pred = preds[batch_idx]
        processed_preds, keep = post_process_img(pred, confidence_threshold=0.25, iou_threshold=iou_thresh)
        preds_to_img = torch.cat((preds_to_img, img_id.repeat(processed_preds.shape[0], 1)), dim=0)
        final_preds = torch.cat((final_preds, processed_preds), dim=0)

    scores= final_preds[:, -1]
    # sort by score, descending
    _, order = scores.sort(descending=True)
    final_preds = final_preds[order]
    preds_to_img = preds_to_img[order]

    total_ap = 0
    for i in range(num_classes):
        gt_class_mask = true_gt[:, 0] == i
        pred_class_mask = final_preds[:, 0] == i
        if not gt_class_mask.any() and not pred_class_mask.any():
            continue
        AP = calculate_AP_per_class(true_gt[gt_class_mask], final_preds[pred_class_mask], 
                               gt_to_img[gt_class_mask], preds_to_img[pred_class_mask],
                               iou_thresh)
        total_ap += AP

    return total_ap / num_classes

def post_process_img(output, confidence_threshold = 0.25, iou_threshold = 0.5) -> torch.Tensor:

    x1 = output[..., 0:1] - output[..., 2:3] / 2
    y1 = output[..., 1:2] - output[..., 3:4] / 2
    x2 = output[..., 0:1] + output[..., 2:3] / 2
    y2 = output[..., 1:2] + output[..., 3:4] / 2

    # boxes: (batch, num_anchors, 4)
    boxes = torch.cat([x1, y1, x2, y2], dim=-1)

    # (batch, num_anchors, 1)
    obj = output[..., 4:5]
    class_probs = output[..., 5:]

    scores = obj * class_probs
    best_scores, best_class = scores.max(dim=-1)

    mask = best_scores > confidence_threshold
    best_scores = best_scores[mask] 
    best_class = best_class[mask] 
    boxes = boxes[mask]
    keep = batched_nms(boxes, best_scores, best_class, iou_threshold = iou_threshold)
    final_boxes = boxes[keep]
    final_classes = best_class[keep]
    final_scores = best_scores[keep]
    # final classes and final scores have shape (num_kept,), so unsqueeze to add the dim 1 again
    predictions = torch.cat((final_classes.unsqueeze(1), 
                             final_boxes, 
                             final_scores.unsqueeze(1)), dim=1)
    return predictions, keep

if __name__ == "__main__":
    mAP = calculate_mAP(torch.randn((16)), 
        torch.randn(16, 30, 5), 
        torch.randn(16, 8400, 6))
    print(f"mAP: {mAP:.4f}")