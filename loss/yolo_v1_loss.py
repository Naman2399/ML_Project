import torch
import torch.nn as nn
from utils.yolo_utils import intersection_over_union

class YoloLoss(nn.Module) :
    def __init__(self, split_size = 7, num_boxes = 2, classes = 20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.classes = classes
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions =  predictions.reshape(-1, self.split_size, self.split_size, self.classes +  self.num_boxes * 5)

        iou_bl = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_bl.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, best_box = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3) # Identity func will be 1 if obj exists

        # ------------ #
        # For box coordinates
        # ------------ #

        '''
                 ... ----> it represent that we have shape of (S, S, C, B)
                 it represents that we are doing for all the shape of (S, S, C) ---> ...
                 and we are working on B part 
                 Shape of B : (x, y, w, h) and probability score for class
        '''


        box_predicitons = exists_box * (
            best_box * predictions[..., 26 : 30]  # index denotes shape of (x, y, w, h)
            + (1 - best_box) * predictions[..., 21 : 25] # index denotes shape of (x, y, w, h)
        )

        box_targets = exists_box * target[..., 21:25] # index denotes shape of (x, y, w, h)

        box_predicitons[..., 2:4] = torch.sign(box_predicitons[..., 2:4]) * torch.sqrt(
            torch.abs(box_predicitons[..., 2 : 4] + 1e-6)
        )  # index [ ..., 2 : 4] denotes w, h for the box

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])  # index [ ..., 2 : 4] denotes w, h for the box

        # (N, S, S, 4) ----> (N * S * S, 4)
        box_loss = self.mse(
            torch.flatten(box_predicitons, end_dim= -2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ------------ #
        # For object loss
        # ------------ #

        pred_box = (
            best_box * predictions[..., 25:26] + (1 - best_box) * predictions[..., 20:21]
        )

        # (N * S * S)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21])
        )

        # ------------ #
        # For No object loss
        # ------------ #

        # (N, S, S, 1) ---> (N, S*S)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # ------------ #
        # For Class loss
        # ------------ #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim= -2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2)
        )

        loss = (
            self.lambda_coord * box_loss +
            object_loss +
            self.lambda_noobj * no_object_loss +
            class_loss
        )

        return loss






