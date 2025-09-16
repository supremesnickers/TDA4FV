import torch
from torch.utils.data import Dataset
import pickle

import mxnet as mx
from mxnet import ndarray as nd

class FacePairDataset(Dataset):
    """Dataset for face pairs with genuine/impostor labels"""

    def __init__(self, pairs, labels, transform=None):
        self.pairs = pairs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1, img2 = self.pairs[idx]
        label = self.labels[idx]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label


class BinDataset(Dataset):
    def __init__(self, path, image_size=(112, 112)):
        try:
            with open(path, "rb") as f:
                bins, issame_list = pickle.load(f)  # py2
        except UnicodeDecodeError:
            with open(path, "rb") as f:
                bins, issame_list = pickle.load(f, encoding="bytes")  # py3
        data_list = []
        for flip in [0, 1]:
            data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
            data_list.append(data)
        for idx in range(len(issame_list) * 2):
            _bin = bins[idx]
            img = mx.image.imdecode(_bin)
            if img.shape[1] != image_size[0]:
                img = mx.image.resize_short(img, image_size[0])
            img = nd.transpose(img, axes=(2, 0, 1))
            for flip in [0, 1]:
                if flip == 1:
                    img = mx.ndarray.flip(data=img, axis=2)
                data_list[flip][idx][:] = torch.from_numpy(img.asnumpy())
            if idx % 1000 == 0:
                print("loading bin", idx)
        print(data_list[0].shape)
        print("num of pairs: ", len(issame_list))
        self.data_list = data_list  # Store original and flipped data
        self.issame_list = issame_list

    def __len__(self):
        return len(self.issame_list)

    def __getitem__(self, index):
        if index >= len(self.issame_list):
            raise IndexError("Index out of range for dataset")

        # Each pair consists of two consecutive images: index*2 and index*2+1
        img1_idx = index * 2
        img2_idx = index * 2 + 1

        # Get original pair (flip=0)
        img1_orig = self.data_list[0][img1_idx]
        img2_orig = self.data_list[0][img2_idx]
        original_pair = (img1_orig, img2_orig)

        # Get flipped pair (flip=1)
        img1_flipped = self.data_list[1][img1_idx]
        img2_flipped = self.data_list[1][img2_idx]
        flipped_pair = (img1_flipped, img2_flipped)

        # Get label (True if same identity, False if different)
        label = self.issame_list[index]

        return original_pair, flipped_pair, label
