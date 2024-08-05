"""
Main file for training Yolo model on Pascal VOC dataset
"""
import os

import torch.optim as optim

import model_utils.yolo_v1.train as training
from loss.yolo_v1_loss import YoloLoss
from utils.checkpoints import create_checkpoint_filename, save_checkpoint
from utils.yolo_utils import (
    non_max_suppression,
    mean_average_precision,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
)


def run(model, train_loader, test_loader, device, writer, args ) :

    model = model.to(device)

    # Define loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = YoloLoss()

    # Update ckpts file name
    checkpoint_filename = create_checkpoint_filename(args)
    checkpoint_path = os.path.join(args.ckpt_path, checkpoint_filename)

    # Output file path
    output_path = "/data/home/karmpatel/karm_8T/naman/demo/pascal_voc/naman_output_samples"

    for epoch in range(args.epochs) :

        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4, device=device
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")

        train_loss = training.train_epoch(train_loader, model, optimizer, loss_fn, writer, checkpoint_path, device, args, epoch)

        writer.add_scalar('train_loss', train_loss, epoch + 1)

        if epoch % 50 == 0 :
            save_checkpoint(args, model, optimizer, epoch, checkpoint_path, train_loss)


    # Plotting bounding boxes
    for x, y in train_loader:
       x = x.to(device)
       for idx in range(8):
           bboxes = cellboxes_to_boxes(model(x))
           bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
           plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes, output_path, f"sample_{idx+1}")

       break
