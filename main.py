import argparse
import sys

import torch
from torch.utils.tensorboard import SummaryWriter

import dataset.breast_cancer_dataset as breast_cancer
import dataset.cifar10 as cifar10
import dataset.digits_dataset as digits
import dataset.housing_dataset as housing
import dataset.images_wild_cats as wild_cats
import dataset.text_corpus_v2 as text_corpus_v2
import models.alexnet as alexnet
import models.lenet_5 as lenet_5
import models.rnn as rnn
import models.vgg16 as vgg16
import models.vgg19 as vgg19
from models import binary_classification as binary_classification
from models import encoder_decoder as encoder_decoder
from models import linear_regression as linear_regression
from models import multiclass_classification as multiclass_classification
from utils.checkpoints import create_checkpoint_filename
from utils.data_utils import plot_feature_vs_target, create_dataloaders
from utils.device import get_available_device
from utils.func_utils import remove_folder_content


def load_dataset(name, args)  :
    datasets = {
        'housing': housing.load_dataset,
        'breast_cancer': breast_cancer.load_dataset,
        'digits': digits.load_dataset,
        'cifar10': cifar10.load_dataset,
        'text_corpus_v2' : text_corpus_v2.load_dataset,
        'wild_cats' : wild_cats.load_dataset
    }
    return datasets[name](args)

def main():

    # Clearing cuda cache
    torch.cuda.empty_cache()

    # Parser arguments
    parser = argparse.ArgumentParser(description="Describe dataset details")
    parser.add_argument("--dataset", type=str,
                        help="Name of the dataset (e.g., 'housing', 'breast_cancer', 'digits', 'cifar10', "
                             "'text_corpus_v1', 'text_corpus_v2' ,'wild_cats')")
    parser.add_argument("--model", type=str,
                        help="Name of the model to use (e.g., 'linear_reg', 'binary_class', "
                             "'multi_class', 'lenet', 'lenetv2', 'encoder_decoder', 'rnn', 'lstm', "
                             "'alexnet', 'vgg16', 'vgg19')")
    parser.add_argument("--batch", type=int, default=256, help="Enter batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Enter number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate")
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment Name")
    parser.add_argument("--ckpt_path", type=str, default="ckpts", help="Load ckpt file for model")
    parser.add_argument("--ckpt_filename", type=str, default=None, help="Load ckpt file for model")

    # Adding parameters for RNNs, LSTMs, GRUs
    parser.add_argument("--chunk", type=int, default=100, help="Chunk size for model")
    parser.add_argument("--embed_size", type=int, default= 300, help = "Embedding size for model")
    parser.add_argument("--hidden_size", type=int, default= 500, help= "Hidden size for model")
    parser.add_argument("--num_layers", type= int, default= 2, help= "No. of layers to stack for RNNs, LSTMs, GRUs")

    args = parser.parse_args()
    writer = SummaryWriter(f"runs/{create_checkpoint_filename(args)}")

    # Remove folder contents
    remove_folder_content(f"runs/{create_checkpoint_filename(args)}")

    '''
    Device details
    '''
    # Example usage
    device = get_available_device()
    print(f"Using device: {device}")
    args.device = device

    '''
    Load Dataset 
    '''
    print(args.dataset)
    if args.dataset.lower() in ['housing', 'breast_cancer', 'digits', 'cifar10']:
        X, y = load_dataset(args.dataset.lower(), args)
        X = X.to(device)
        y = y.to(device)
        train_loader, test_loader, val_loader = create_dataloaders(X, y, batch_size=args.batch, test_frac=0.1, val_frac=0.1, device=device)

    elif args.dataset.lower() in ['text_corpus_v1', 'text_corpus_v2' ] :
        dataset_obj = load_dataset(args.dataset.lower(), args)

    elif args.dataset.lower() in ['wild_cats'] :
        dataloader_train, dataloader_valid, dataloader_test = load_dataset(args.dataset.lower(), args)

    else:
        print("Dataset doesn't exist")
        print("Please provide a dataset name using the --dataset argument.")
        return


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
        import model_run.linear_regression as main_modules
        model = linear_regression.LinearRegression(input_size=X.shape[1])
        main_modules.run(model, X, args, device, test_loader, train_loader, val_loader)

    '''
    Binary Dataset 
    
    Breast Cancer Dataset
    Input Dimension - ( N * 30)
    Labels - N # Here the labels are 0 or 1, 1 ----> positive for cancer
    '''

    if args.dataset.lower() == 'breast_cancer' :

        # Binary Classification Model
        model = binary_classification.BinaryClassifier(input_size=X.shape[1])
        import model_run.binary_classification as main_modules
        model = main_modules.run(X, args, device, model, test_loader, train_loader, val_loader, writer)

    '''
    Multiclass classification 
    
    Digits - 
    Input Dim - (N * 64)
    Labels Dim - (N * 10) denotes one hot vector for 10 labels
    '''
    if args.dataset.lower() == 'digits' :

        # Multi-class Classification Model
        if args.model.lower() == 'multi_class' :
            model = multiclass_classification.SimpleNN(input_size=X.shape[1], num_classes=y.shape[1])
            import model_run.multi_class_classification as main_modules
            main_modules.run(X, args, device, model, test_loader, train_loader, val_loader, writer)

        # Encoder decoder Model
        if args.model.lower() == 'encoder_decoder' :
            model = encoder_decoder.EncoderDecoder(input_size=X.shape[1], hidden_size= 10)
            import model_run.encoder_decoder as main_modules
            main_modules.run(X, args, device, model, test_loader, train_loader, val_loader, writer)


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

        import model_run.multi_class_image_classification as main_modules
        main_modules.run(X, args, device, model, test_loader, train_loader, val_loader, writer)


    '''
    Multi-class classification 
    Input dim - (N, 3, 224, 224)
    Labels dim - (N, 10)
    '''
    if args.dataset.lower() == 'wild_cats' :
        print("OK")

        if args.model.lower() == 'alexnet' :
            model = alexnet.AlexNet()

        if args.model.lower() == 'vgg16' :
            model = vgg16.VGG16()

        if args.model.lower() == 'vgg19' :
            model = vgg19.VGG19()

        for images, labels in dataloader_train :
            X = images
            break

        import model_run.multi_class_image_classification as main_modules
        main_modules.run(X, args, device, model, dataloader_test, dataloader_train, dataloader_valid, writer)


    '''
    Text Corpus Data
        Input Dim : (batch_size, num_of_batches) where each entry is a token representation for that word 
    '''
    if args.dataset.lower() == 'text_corpus_v2' :
        print("OK")

        if args.model.lower() == 'rnn' :
            model = rnn.RNN(input_size= dataset_obj.n_characters, embedding_size= args.embed_size,
                            hidden_size= args.hidden_size , output_size= dataset_obj.n_characters,
                            n_layers= args.num_layers, vocab_size= dataset_obj.n_characters)

            import model_run.rnn as main_modules
            main_modules.run(dataset= dataset_obj, model= model, args= args, device= device, writer= writer)

        return

    writer.close()
    sys.exit()





if __name__ == "__main__":
    main()