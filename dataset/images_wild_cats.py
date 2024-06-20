import os
import random

import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path_label, num_classes, transform=None):
        self.path_label = path_label
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.path_label)

    def __getitem__(self, idx):
        path, label = self.path_label[idx]
        img = Image.open(path).convert('RGB')
        label = torch.nn.functional.one_hot(torch.tensor(label), num_classes= self.num_classes)
        label = label.float()

        if self.transform is not None:
            img = self.transform(img)

        return img, label


class ImageDataset(pl.LightningDataModule):
    def __init__(self, path_label, num_classes, batch_size=32):
        super().__init__()
        self.path_label = path_label
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize(224),  # resize shortest side to 224 pixels
            transforms.CenterCrop(224),  # crop longest side to 224 pixels at center
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.num_classes = num_classes

    def setup(self, stage=None):
        if stage == 'Test':
            test_dataset = CustomDataset(self.path_label, self.num_classes, self.transform)
            self.test_dataset = torch.utils.data.Subset(test_dataset, range(len(test_dataset)))
            self.test_dataset = DataLoader(self.test_dataset, batch_size=self.batch_size)

        elif stage == 'Train':
            train_dataset = CustomDataset(self.path_label, self.num_classes, self.transform)
            self.train_dataset = torch.utils.data.Subset(train_dataset, range(len(train_dataset)))
            self.train_dataset = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        else :
            val_dataset = CustomDataset(self.path_label, self.num_classes,  self.transform)
            self.val_dataset = torch.utils.data.Subset(val_dataset, range(len(val_dataset)))
            self.val_dataset = DataLoader(self.val_dataset, batch_size=self.batch_size)


    def __len__(self):
        if self.train_dataset is not None:
            return len(self.train_dataset)
        elif self.val_dataset is not None:
            return len(self.val_dataset)
        else:
            return 0

    def __getitem__(self, index):
        if self.train_dataset is not None:
            return self.train_dataset[index]
        elif self.test_dataset is not None:
            return self.test_dataset[index]
        else:
            raise IndexError("Index out of range. The dataset is empty.")

    def train_dataset(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataset(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataset(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


def create_path_label_list(df):
    path_label_list = []
    for _, row in df.iterrows():
        path = row['path']
        label = row['label']
        path_label_list.append((path, label))
    return path_label_list


def describe_dataset(args) :

    # Adding directory paths

    path = args.dataset_path
    train_dir = os.path.join(args.dataset_path, 'train' )
    valid_dir = os.path.join(args.dataset_path, 'valid' )
    test_dir = os.path.join(args.dataset_path, 'test' )

    # Adding classes and paths for train, valid, test

    classes_train = []
    paths_train = []
    for dirname, _, filenames in os.walk(train_dir):
        for filename in filenames:
            classes_train += [dirname.split('/')[-1]]
            paths_train += [(os.path.join(dirname, filename))]

    classes_valid = []
    paths_valid = []
    for dirname, _, filenames in os.walk(valid_dir):
        for filename in filenames:
            classes_valid += [dirname.split('/')[-1]]
            paths_valid += [(os.path.join(dirname, filename))]

    classes_test = []
    paths_test = []
    for dirname, _, filenames in os.walk(test_dir):
        for filename in filenames:
            classes_test += [dirname.split('/')[-1]]
            paths_test += [(os.path.join(dirname, filename))]

    # Printing different categories for Wild cats
    N = list(range(len(classes_train)))
    class_names = sorted(set(classes_train))
    print("Classes for wild cats : ", class_names)

    normal_mapping = dict(zip(class_names, N))
    reverse_mapping = dict(zip(N, class_names))

    # Train
    data_train = pd.DataFrame(columns=['path', 'class', 'label'])
    data_train['path'] = paths_train
    data_train['class'] = classes_train
    data_train['label'] = data_train['class'].map(normal_mapping)
    # Valid
    data_valid = pd.DataFrame(columns=['path', 'class', 'label'])
    data_valid['path'] = paths_valid
    data_valid['class'] = classes_valid
    data_valid['label'] = data_valid['class'].map(normal_mapping)
    # Test
    data_test = pd.DataFrame(columns=['path', 'class', 'label'])
    data_test['path'] = paths_test
    data_test['class'] = classes_test
    data_test['label'] = data_test['class'].map(normal_mapping)

    # Printing label for some dataset

    # Train
    path_label_train = create_path_label_list(data_train)
    path_label_train = random.sample(path_label_train, len(path_label_train))
    print("Train labels : ", len(path_label_train))
    print("Train labels samples : ",path_label_train[0:3])

    path_label_valid = create_path_label_list(data_valid)
    path_label_valid = random.sample(path_label_valid, len(path_label_valid))
    print("Valid labels : ", len(path_label_valid))
    print("Valid labels samples :",path_label_valid[0:3])

    path_label_test = create_path_label_list(data_test)
    path_label_test = random.sample(path_label_test, len(path_label_test))
    print("Test labels : ",len(path_label_test))
    print("Test labels samples : ",path_label_test[0:3])

    train_dataset = ImageDataset(path_label_train, batch_size= args.batch, num_classes = len(class_names))
    val_dataset = ImageDataset(path_label_valid, batch_size= args.batch, num_classes = len(class_names))
    test_dataset = ImageDataset(path_label_test, batch_size= args.batch, num_classes = len(class_names))
    train_dataset.setup(stage = "Train")
    val_dataset.setup(stage= "Valid")
    test_dataset.setup(stage="Test")

    dataloader_train = train_dataset.train_dataset
    dataloader_valid = val_dataset.val_dataset
    dataloader_test = test_dataset.test_dataset

    for images, labels in dataloader_train :
        print(f"Train Shape of one image: {images.shape}, \t Type : {type(images)}")
        print(f"Train Shape of one label: {labels.shape}, \t Type : {type(labels)}")
        break

    for images, labels in dataloader_valid :
        print(f"Valid Shape of one image: {images.shape}, \t Type : {type(images)}")
        print(f"Valid Shape of one label: {labels.shape}, \t Type : {type(labels)}")
        break

    for images, labels in dataloader_test :
        print(f"Test Shape of one image: {images.shape}, \t Type : {type(images)}")
        print(f"Test Shape of one label: {labels.shape}, \t Type : {type(labels)}")
        break

    return dataloader_train, dataloader_valid, dataloader_test


def load_dataset(args) :
    return describe_dataset(args)

