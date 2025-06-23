#!/usr/bin/env python
"""
Better YOLOX loss with proper target assignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOXLoss(nn.Module):
    """YOLOX Loss with SimOTA assignment (simplified)"""
    
    def __init__(self, num_classes, strides=[8, 16, 32]):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='none')
        
    def forward(self, predictions, targets, input_size=640):
        """
        Args:
            predictions: [batch, total_anchors, 5+num_classes]
            targets: [batch, max_objects, 5] - (class, x, y, w, h) normalized
            input_size: Input image size
        """
        device = predictions.device
        batch_size = predictions.shape[0]
        
        # Generate anchors/grids
        grids, strides = self.generate_grids(input_size, device)
        
        total_loss = 0
        total_obj_loss = 0
        total_cls_loss = 0
        total_box_loss = 0
        num_fg = 0
        
        for batch_idx in range(batch_size):
            # Get valid targets
            valid_mask = targets[batch_idx, :, 0] >= 0
            if not valid_mask.any():
                # No objects - only background loss
                obj_targets = torch.zeros(predictions.shape[1], device=device)
                obj_loss = self.bce_loss(predictions[batch_idx, :, 4], obj_targets).mean()
                total_obj_loss += obj_loss
                continue
                
            gt_targets = targets[batch_idx][valid_mask]  # [num_objects, 5]
            gt_classes = gt_targets[:, 0].long()
            gt_boxes = gt_targets[:, 1:5] * input_size  # Convert to pixels
            
            # Simple assignment (replace with SimOTA for better results)
            pos_mask, matched_gt_inds = self.assign_targets(
                predictions[batch_idx], gt_boxes, gt_classes, grids, strides
            )
            
            num_pos = pos_mask.sum()
            num_fg += num_pos
            
            # Objectness loss
            obj_targets = torch.zeros(predictions.shape[1], device=device)
            obj_targets[pos_mask] = 1.0
            obj_loss = self.bce_loss(predictions[batch_idx, :, 4], obj_targets).mean()
            total_obj_loss += obj_loss
            
            if num_pos > 0:
                # Classification loss
                cls_targets = torch.zeros(num_pos, self.num_classes, device=device)
                cls_targets[range(num_pos), gt_classes[matched_gt_inds]] = 1.0
                cls_loss = self.bce_loss(
                    predictions[batch_idx, pos_mask, 5:], cls_targets
                ).mean()
                total_cls_loss += cls_loss
                
                # Box loss
                pred_boxes = predictions[batch_idx, pos_mask, :4]
                target_boxes = gt_boxes[matched_gt_inds]
                box_loss = self.iou_loss(pred_boxes, target_boxes).mean()
                total_box_loss += box_loss
        
        # Combine losses
        total_loss = 5.0 * total_box_loss + total_obj_loss + total_cls_loss
        
        return {
            'total_loss': total_loss,
            'box_loss': total_box_loss,
            'obj_loss': total_obj_loss,
            'cls_loss': total_cls_loss,
            'num_fg': num_fg
        }
    
    def generate_grids(self, input_size, device):
        """Generate anchor grids for all scales"""
        grids = []
        strides = []
        
        for stride in self.strides:
            size = input_size // stride
            yv, xv = torch.meshgrid([torch.arange(size), torch.arange(size)], indexing='ij')
            grid = torch.stack((xv, yv), 2).view(-1, 2).float()
            grid = grid * stride + stride // 2  # Center of cells
            grids.append(grid)
            strides.extend([stride] * len(grid))
        
        all_grids = torch.cat(grids, dim=0).to(device)
        all_strides = torch.tensor(strides, device=device)
        
        return all_grids, all_strides
    
    def assign_targets(self, predictions, gt_boxes, gt_classes, grids, strides):
        """Simple target assignment - replace with SimOTA"""
        num_gt = len(gt_boxes)
        num_anchors = len(predictions)
        
        if num_gt == 0:
            return torch.zeros(num_anchors, dtype=torch.bool), torch.tensor([])
        
        # Simple distance-based assignment
        gt_centers = gt_boxes[:, :2]  # [num_gt, 2]
        anchor_centers = grids  # [num_anchors, 2]
        
        # Compute distances
        distances = torch.cdist(gt_centers, anchor_centers)  # [num_gt, num_anchors]
        
        # Assign each anchor to closest GT
        closest_gt = distances.argmin(dim=0)  # [num_anchors]
        min_distances = distances.min(dim=0)[0]  # [num_anchors]
        
        # Keep only close anchors
        pos_mask = min_distances < 64  # Threshold in pixels
        matched_gt_inds = closest_gt[pos_mask]
        
        return pos_mask, matched_gt_inds
    
    def iou_loss(self, pred_boxes, target_boxes):
        """IoU loss"""
        # Convert predictions to actual coordinates
        pred_centers = pred_boxes[:, :2] 
        pred_sizes = torch.exp(pred_boxes[:, 2:4])
        
        pred_x1 = pred_centers[:, 0] - pred_sizes[:, 0] / 2
        pred_y1 = pred_centers[:, 1] - pred_sizes[:, 1] / 2
        pred_x2 = pred_centers[:, 0] + pred_sizes[:, 0] / 2
        pred_y2 = pred_centers[:, 1] + pred_sizes[:, 1] / 2
        
        target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2
        
        # Intersection
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Union
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area
        
        iou = inter_area / (union_area + 1e-6)
        return 1 - iou


# Usage example
def create_loss_fn(num_classes):
    return YOLOXLoss(num_classes=num_classes)
