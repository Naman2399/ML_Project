"""
Main file for training Yolo model on Pascal VOC dataset
"""

from tqdm import tqdm


def train_epoch(train_loader, model, optimizer, loss_fn, writer, checkpoint_path, device, args):

    # Create pbar
    pbar = tqdm(train_loader, leave=True)

    mean_loss = []

    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        pbar.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")
    mean_loss.append(sum(mean_loss)/len(mean_loss))


    return mean_loss, model


