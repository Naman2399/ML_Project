import argparse
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from model_utils.checkpoints import create_checkpoint_filename, load_checkpoint
from model_utils.device import get_available_device
from utils import split_dataset, plot_feature_vs_target, create_dataloader, plot_losses, create_dataloaders, plot_accuracies
import dataset.housing_dataset as housing
import dataset.breast_cancer_dataset as breast_cancer
import dataset.digits_dataset as digits
import dataset.cifar10 as cifar10
import logisitc_regression.multiclass_classification as multi_class
import models.lenet_5 as lenet_5
import model_utils.train as training
import model_utils.evaluation as evaluation

import os
from torch.utils.tensorboard import SummaryWriter

def load_dataset(name)  :
    datasets = {
        'housing': housing.load_dataset,
        'breast_cancer': breast_cancer.load_dataset,
        'digits': digits.load_dataset,
        'cifar10': cifar10.load_dataset
    }
    return datasets[name]()

def main():
    parser = argparse.ArgumentParser(description="Describe dataset details")
    parser.add_argument("--dataset", type=str, help="Name of the dataset (e.g., 'housing', 'breast_cancer', 'digits', 'cifar10')")
    parser.add_argument("--model", type=str, help="Name of the model to use (e.g., 'linear_reg', 'binary_class', 'multi_class', 'lenet', 'lenetv2')")
    parser.add_argument("--batch", type=int, default=256, help="Enter batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Enter number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate")
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment Name")
    parser.add_argument("--ckpt_path", type=str, default="ckpts", help="Load ckpt file for model")
    parser.add_argument("--ckpt_filename", type=str, default=None, help="Load ckpt file for model")

    args = parser.parse_args()
    writer = SummaryWriter(f"runs/{create_checkpoint_filename(args)}")


    '''
    Device details
    '''
    # Example usage
    device = get_available_device()
    print(f"Using device: {device}")

    '''
    Load Dataset 
    '''
    if args.dataset.lower() in ['housing', 'breast_cancer', 'digits', 'cifar10']:
        X, y = load_dataset(args.dataset.lower())
    else:
        print("Dataset doesn't exist")
        print("Please provide a dataset name using the --dataset argument.")
        return

    X = X.to(device)
    y = y.to(device)
    train_loader, test_loader, val_loader = create_dataloaders(X, y, batch_size=args.batch, test_frac=0.1, val_frac=0.1, device = device)

    '''
    Dataset ----> Mapping to Models 
    '''

    '''
    Continuous Dataset
    
    Housing Dataset 
    Input Dimensions : (N * number of features)
    Labels Dimension : Continuous values 
    Features : ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
    '''
    if args.dataset.lower() == 'housing':
        plot_feature_vs_target(X, y, output_dir="plots", file_prefix="features_vs_target")

        # Linear Regression Model
        import main_modules.main_linear_regression as main_modules
        main_modules.run(X, args, device, test_loader, train_loader, val_loader)

    '''
    Binary Dataset 
    
    Breast Cancer Dataset
    Input Dimension - ( N * 30)
    Labels - N # Here the labels are 0 or 1, 1 ----> positive for cancer
    '''

    if args.dataset.lower() == 'breast_cancer' :

        # Binary Classification Model
        import main_modules.main_binary_classification as main_modules
        model = main_modules.run(X, args, device, test_loader, train_loader, val_loader)

    '''
    Multiclass classification 
    
    Digits - 
    Input Dim - (N * 64)
    Labels Dim - (N * 10) denotes one hot vector for 10 labels
    '''
    if args.dataset.lower() == 'digits' :

        # Multi-class Classification Model
        import main_modules.main_multi_class_classification as main_modules
        model = main_modules.run(X, args, device, test_loader, train_loader, val_loader, y)

    '''
    Multi-class classification 
    Input dim - (N, 3, 32, 32)
    Labels dim - (N, 10)
    '''
    if args.dataset.lower() == 'cifar10' :

        if args.model.lower() == 'lenet' :
            model = lenet_5.LeNet5()
        elif args.model.lower() == 'lenetv2' :
            model = lenet_5.LeNet5_v2()

        import main_modules.main_mnist_image_classification as main_modules
        main_modules.run(X, args, device, model, test_loader, train_loader, val_loader, writer)

    writer.close()
    sys.exit()





if __name__ == "__main__":
    main()