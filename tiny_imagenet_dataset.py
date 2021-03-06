import os
import torch.utils.data
from torchvision.datasets.utils import verify_str_arg
from PIL import Image
import random

CLASSES_FILE = "wnids.txt"
IMAGES_DIR = "images"
IMAGE_FILE_EXT = ".JPEG"
TRAIN_DIR = "train"
VALIDATION_DIR = "val"

class TinyImageNet(torch.utils.data.Dataset):
    def __init__(self, root_dir="./tiny-imagenet-200", split='train', transform=None, images_per_class_train=20, num_val_images=1000):
        verify_str_arg(split, "split", ("train", "val"))
        self.transform = transform
        self.train_dir = os.path.join(root_dir, TRAIN_DIR)
        self.val_dir = os.path.join(root_dir, VALIDATION_DIR)
        self.images_per_class = images_per_class_train
        self.num_val_images = num_val_images

        if split == 'train':
            self.dataset = self.create_training_set()

        if split == 'val':
            self.dataset = self.create_validation_set()

    def create_training_set(self):
        training_set = []
        for c in os.listdir(self.train_dir):
            c_dir = os.path.join(self.train_dir, c, IMAGES_DIR)
            try:
                c_images = os.listdir(c_dir)
            except NotADirectoryError:
                print(c_dir, " not a directory")
                continue

            random.shuffle(c_images)
            for img_name_i in c_images[0:self.images_per_class]:
                img_path = os.path.join(c_dir, img_name_i)

                # Ensure that the image is RGB
                image = Image.open(img_path)
                if len(image.getbands()) != 3:
                    continue

                training_set.append(img_path)

        random.shuffle(training_set)
        return training_set

    def create_validation_set(self):
        validation_set = []
        val_dir = os.path.join(self.val_dir, IMAGES_DIR)
        val_images = os.listdir(val_dir)

        for img_name_i in val_images[0:self.num_val_images]:
            img_path = os.path.join(val_dir, img_name_i)

            # Ensure that the image is RGB
            image = Image.open(img_path)
            if len(image.getbands()) != 3:
                continue

            validation_set.append(img_path)

        return validation_set

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path = self.dataset[index]
        image = Image.open(img_path)

        if self.transform is not None:
            try:
                image = self.transform(image)
            except:
                print("Caught non-RGB image in dataset")
                print(img_path)

        return image
