import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from dataset.text_corpus_v2 import DatasetShakespeare
from utils.checkpoints import create_checkpoint_filename, load_checkpoint
from utils.data_utils import plot_losses
from model_utils.rnn.train import train
from model_utils.rnn.validation import validation


def run(dataset : DatasetShakespeare, model, args, device, writer):

    input_tensor_example, target_tensor_example = dataset.random_training_batch()

    input_tensor_example = input_tensor_example.to(device)
    hidden_tensor_example = model.init_hidden(batch_size= args.batch).to(device)

    print("Input Tensor Example : ", input_tensor_example.shape)
    print("Model Hidden Shape : ", hidden_tensor_example.shape)

    model.to(device)
    # Model Summary
    # summary(model, [(input_tensor_example.shape[0], input_tensor_example.shape[1]), (hidden_tensor_example.shape[0], hidden_tensor_example.shape[1], hidden_tensor_example.shape[2])])

    # Tensorboard - adding model
    writer.add_graph(model, [input_tensor_example, hidden_tensor_example])

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    current_epoch = 0
    least_val_loss = float('inf')

    # Checkpoint filename
    if args.ckpt_filename is None:
        # Create new ckpts file
        checkpoint_filename = create_checkpoint_filename(args)
        checkpoint_path = os.path.join(args.ckpt_path, checkpoint_filename)

    else:
        print("Loading contents from checkpoints")
        # Load model, optimizer, epoch from checkpoints
        checkpoint_path = os.path.join(args.ckpt_path, args.ckpt_filename)
        model, optimizer, current_epoch, val_loss = load_checkpoint(model, optimizer, checkpoint_path=checkpoint_path)
        current_epoch += 1
        least_val_loss = val_loss

    # Model Training and Validation
    train_losses, val_losses = train(model= model, dataset= dataset,
                    criterion= criterion, optimizer= optimizer, epochs= args.epochs, writer= writer,
                    checkpoint_path= checkpoint_path, current_epoch= current_epoch,
                    least_val_loss= least_val_loss, args= args, device= device)

    # Plot losses and accuracies
    plot_losses(train_losses=train_losses, val_losses=val_losses, file_name=f"{args.model.lower()}-loss.png")

    # Test and evaluation and confusion matrix
    from  model_utils.rnn.evluation import evaluate
    start_words = 'A king'
    predicted_len = 100
    predicted_sentence = evaluate(model= model, dataset= dataset, start_words= start_words, predict_len = predicted_len, temperature = 0.8, device= device)
    print("Sentence : ", start_words)
    print(f"After prediction for predicted length : {predicted_len} ")
    print("Predicted Sentence : ", predicted_sentence)
