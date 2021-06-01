import os
import torch.utils.data
from torchvision.datasets.folder import default_loader
from torchvision.io import read_image
from torchvision.datasets.utils import verify_str_arg

class TinyImageNet(torch.utils.data.Dataset):
    base_folder = 'tiny-imagenet-200/'

    def __init__(self, root="./", split='train', transform=None):
        self.root = root
        self.dataset_path = os.path.join(root, self.base_folder)
        self.loader = default_loader
        self.split = verify_str_arg(split, "split", ("train", "val",))
        self.transform = transform
        self.class_to_idx = self.find_classes()
        self.data = self.make_dataset()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]
        image = read_image(img_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def find_classes(self):
        class_file = os.path.join(self.dataset_path, 'wnids.txt')
        with open(class_file) as r:
            classes = list(map(lambda s: s.strip(), r.readlines()))

        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return class_to_idx

    def make_dataset(self):
        images = []
        dir_path = os.path.join(self.root, self.base_folder, self.split)

        if self.split == 'train':
            for fname in sorted(os.listdir(dir_path)):
                cls_fpath = os.path.join(dir_path, fname)
                if os.path.isdir(cls_fpath):
                    cls_imgs_path = os.path.join(cls_fpath, 'images')
                    for imgname in sorted(os.listdir(cls_imgs_path)):
                        path = os.path.join(cls_imgs_path, imgname)
                        item = (path, self.class_to_idx[fname])
                        images.append(item)
        else:
            imgs_path = os.path.join(dir_path, 'images')
            imgs_annotations = os.path.join(dir_path, 'val_annotations.txt')

            with open(imgs_annotations) as r:
                data_info = map(lambda s: s.split('\t'), r.readlines())

            cls_map = {line_data[0]: line_data[1] for line_data in data_info}

            for imgname in sorted(os.listdir(imgs_path)):
                path = os.path.join(imgs_path, imgname)
                item = (path, self.class_to_idx[cls_map[imgname]])
                images.append(item)

        return images
