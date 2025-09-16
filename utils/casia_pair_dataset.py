from torch.utils.data import Dataset
from casia import CasiaWebFace
import random


class CasiaPairDataset(Dataset):
    """
    Wrapper around CasiaWebFace that yields pairs of images and a label:
    - Returns (img1, img2, label) where:
      - img1, img2 are preprocessed image tensors
      - label = 1 if same identity, 0 if different identity
    """
    
    def __init__(self, casia_dataset: CasiaWebFace, same_prob=0.5):
        """
        Args:
            casia_dataset: Instance of CasiaWebFace dataset
            same_prob: Probability of generating same-identity pairs (default: 0.5)
        """
        self.casia_dataset = casia_dataset
        self.same_prob = same_prob
        
        # Build index mapping: label -> list of sample indices
        self.label_to_indices = {}
        for idx, label in enumerate(self.casia_dataset.labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
        
        self.labels_list = list(self.label_to_indices.keys())

    def __len__(self):
        return len(self.casia_dataset)

    def __getitem__(self, idx):
        # Get first image and its label
        img1, label1, _ = self.casia_dataset[idx]
        
        # Decide whether to create same-identity or different-identity pair
        if random.random() < self.same_prob:
            # Same identity pair (label = 1)
            pair_label = 1
            # Choose a different sample from the same identity
            same_identity_indices = self.label_to_indices[label1]
            if len(same_identity_indices) > 1:
                # Pick a different sample from same identity
                idx2 = idx
                while idx2 == idx:
                    idx2 = random.choice(same_identity_indices)
            else:
                # If only one sample for this identity, use the same sample
                idx2 = idx
        else:
            # Different identity pair (label = 0)
            pair_label = 0
            # Choose a random different identity
            different_label = label1
            while different_label == label1:
                different_label = random.choice(self.labels_list)
            # Pick random sample from different identity
            idx2 = random.choice(self.label_to_indices[different_label])
        
        # Get second image
        img2, _, _ = self.casia_dataset[idx2]
        
        return img1, img2, pair_label
