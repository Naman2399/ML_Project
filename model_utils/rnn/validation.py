import torch
import torch.nn as nn

def validation(model, dataset, criterion, device, args) :
    # Training
    model.eval()

    with torch.no_grad() :
        running_loss = 0.0

        # Forward Propagation
        input, target = dataset.random_training_batch()
        target_one_hot = nn.functional.one_hot(target, num_classes=dataset.n_characters).to(torch.float).to(device)
        hidden = model.init_hidden(batch_size=args.batch)
        input = input.to(device)
        target = target.to(device)
        hidden = hidden.to(device)
        output_hat, hidden = model(input, hidden)

        # Loss
        loss = criterion(output_hat, target_one_hot)
        running_loss = loss.item() / args.batch

        return running_loss