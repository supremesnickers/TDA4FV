# %%
import os

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, RandomSampler
from torchvision import transforms

import cv2

from utils.losses import ArcFace
from backbones.iresnet import iresnet50

from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
CHECKPOINTS_PATH_FORMAT = '../checkpoints/{}backbone.pth' # 479 steps per epoch
CASIA_NUM_CLASSES = 10572
NUM_CLASSES = 100
NUM_SAMPLES_PER_CLASS = 100

# %%
def get_file_count(directory):
    file_count = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        file_count += len(filenames)
    return file_count

def sort_directories_by_file_count(base_path):
    directories = [
        d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))
    ]
    directories_file_counts = [
        (d, get_file_count(os.path.join(base_path, d))) for d in directories
    ]
    directories_file_counts.sort(key=lambda x: x[1], reverse=True)
    return directories_file_counts

class CasiaWebFace(Dataset):
    def __init__(self, root_dir, local_rank, num_classes=10572, num_samples=None, selective=False):
        super(CasiaWebFace, self).__init__()
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.root_dir = root_dir
        self.local_rank = local_rank
        self.imgidx, self.labels = self.scan(root_dir, num_classes, num_samples, selective)
        self.imageindex = np.array(range(len(self.imgidx)))

    def scan(self, root, num_classes, num_samples, selective):
        imgidex = []
        labels = []
        lb = -1
        list_dir = os.listdir(root)
        list_dir.sort()

        current_num_classes = 0

        if selective:
            directories = sort_directories_by_file_count(root)
        else:
            directories = [(l, len(os.listdir(os.path.join(root, l)))) for l in list_dir]

        for l, file_count in directories:
            if num_classes is not None and current_num_classes >= num_classes:
                break
            
            images = os.listdir(os.path.join(root, l))
            if len(images) < num_samples:
                # Skip classes with fewer than 500 images
                continue

            lb += 1
            for idx, img in enumerate(images):
                if idx >= num_samples:
                    break
                imgidex.append(os.path.join(l, img))
                labels.append(lb)

            current_num_classes += 1

        return imgidex, labels
    
    
    def read_image(self, path):
        return cv2.imread(os.path.join(self.root_dir, path))

    def __getitem__(self, index):
        path = self.imgidx[index]
        imageindex = self.imageindex[index]
        img = self.read_image(path)
        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.long)
        sample = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, imageindex

    def __len__(self):
        return len(self.imgidx)

# %%
full_dataset = CasiaWebFace("/data/ozgur/faces_webface_112x112/casia_training", 0, NUM_CLASSES, NUM_SAMPLES_PER_CLASS, True)
assert len(full_dataset) == NUM_CLASSES * NUM_SAMPLES_PER_CLASS, "Dataset size does not match expected size. Expected: {}, Found: {}".format(
    NUM_CLASSES * NUM_SAMPLES_PER_CLASS, len(full_dataset)
)

# Create a seeded generator for reproducible sampling
generator = torch.Generator()
generator.manual_seed(42)

random_sampler = RandomSampler(full_dataset, generator=generator)

dl = DataLoader(
    full_dataset,
    sampler=random_sampler,
    batch_size=20,
    num_workers=0,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
)

backbone = iresnet50().to(device)

fc_layer = list(backbone.children())[-2]

header = ArcFace(512, CASIA_NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss(reduction="none").to(device)

# %%
inverse_normalize = transforms.Compose([
    transforms.Normalize(mean = [0., 0., 0.], std = [1/0.5, 1/0.5, 1/0.5]),
    transforms.Normalize(mean = [-0.5, -0.5, -0.5], std = [1., 1., 1.]), 
])

# %%

def load_checkpoint(backbone: nn.Module, header, checkpoint_path: str):
    """
    Load a checkpoint from the specified path and return the learning rate.
    Args:
        model (nn.Module): The model to load the checkpoint into.
        checkpoint_path (str): The path to the checkpoint file.
    Returns:
        learning_rate (float): The learning rate from the checkpoint.
    """

    # Load the checkpoint
    backbone_checkpoint = torch.load(checkpoint_path)
    backbone.load_state_dict(backbone_checkpoint)

    header_checkpoint = torch.load(checkpoint_path.replace('backbone', 'header'))
    header.load_state_dict(header_checkpoint)

    backbone.eval()
    header.eval()
    learning_rate = backbone_checkpoint.get('learning_rate', 1.0)

    # Return the loaded checkpoint
    return learning_rate


def calc_gradients(inputs: torch.Tensor, labels: torch.Tensor):
    with torch.autograd.set_grad_enabled(True):
        # Forward pass
        features = backbone(inputs)
        thetas = header(features, labels)
        loss = criterion(thetas, labels)

        assert loss.shape[0] == thetas.shape[0], "Loss and output batch sizes do not match."

        # print("Loss shape:", loss.shape)
        grads_list = [torch.autograd.grad(outputs=loss[i], inputs=thetas, grad_outputs=torch.ones_like(
            loss[i]), retain_graph=True) for i in range(loss.shape[0])]
        # print("gradients list length:", grads_list[0])
        grads = [torch.stack(x) for x in zip(*grads_list)]
        
        return grads


# %%
def main():
    checkpoint_paths = [CHECKPOINTS_PATH_FORMAT.format(i * 4790) for i in range(1, 5)]  # use every tenth epoch as a checkpoint to evaluate
    
    # Process each checkpoint
    for checkpoint_idx, checkpoint_path in enumerate(checkpoint_paths):
        print(f"Processing checkpoint {checkpoint_idx + 1}/{len(checkpoint_paths)}: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint {checkpoint_path} does not exist, skipping.")
            continue
        
        # Load the checkpoint
        learning_rate = load_checkpoint(backbone, header, checkpoint_path)
        
        # Create checkpoint-specific output directory
        checkpoint_output_dir = f"output/gradients/checkpoint_{checkpoint_idx + 1}"
        
        # Create a new dataloader with the same seed for each checkpoint to ensure reproducibility
        checkpoint_generator = torch.Generator()
        checkpoint_generator.manual_seed(42)
        
        checkpoint_sampler = RandomSampler(full_dataset, generator=checkpoint_generator)
        checkpoint_dl = DataLoader(
            full_dataset,
            sampler=checkpoint_sampler,
            batch_size=20,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        
        # Process all batches for this checkpoint
        for batch_idx, (samples, labels, imageindex) in enumerate(tqdm(checkpoint_dl, desc=f"Checkpoint {checkpoint_idx + 1}")):
            samples = samples.to(device)
            labels = labels.to(device)

            # Calculate gradients
            gradients = calc_gradients(samples, labels)
            
            # Dump gradients in same folder structure as images
            dump_gradients_by_structure(gradients, labels, imageindex, full_dataset, checkpoint_output_dir)
        
        print(f"Completed checkpoint {checkpoint_idx + 1}")
        
        # Clear GPU memory after each checkpoint
        torch.cuda.empty_cache()

        
# %%
def dump_gradients_by_structure(gradients, labels, imageindices, dataset, output_base_dir="output/gradients"):
    """
    Dump gradient tensors in the same folder structure as the original images.
    
    Args:
        gradients: List of gradient tensors for each sample in the batch
        labels: Tensor of labels for each sample in the batch
        imageindices: Tensor of image indices for each sample in the batch  
        dataset: The CasiaWebFace dataset instance to get image paths
        output_base_dir: Base directory where gradients will be saved
    """
    import json
    
    # Create base output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Create or update metadata file for this checkpoint
    metadata_path = os.path.join(output_base_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {
            "checkpoint_info": os.path.basename(output_base_dir),
            "gradient_shapes": [g.shape for g in gradients] if gradients else [],
            "num_samples_processed": 0,
            "description": "Gradients saved in same structure as original images"
        }
    
    batch_size = labels.shape[0]
    
    for i in range(batch_size):
        # Get the original image path from dataset
        image_idx = imageindices[i].item()
        original_path = dataset.imgidx[image_idx]  # This gives us "class_folder/image.jpg"        # Extract class folder and image filename
        class_folder = os.path.dirname(original_path)
        image_filename = os.path.basename(original_path)
        
        # Remove image extension and add .npy for gradient tensor
        gradient_filename_base = os.path.splitext(image_filename)[0] + "_grad"
        
        # Create class directory structure
        class_dir = os.path.join(output_base_dir, class_folder)
        os.makedirs(class_dir, exist_ok=True)
        
        # Extract gradients for this sample (index i from each gradient tensor)
        sample_gradients = [grad[i] for grad in gradients]
        
        # Save each gradient parameter as a separate .npy file
        for j, grad in enumerate(sample_gradients):
            param_filename = f"{gradient_filename_base}_param_{j}.npy"
            param_path = os.path.join(class_dir, param_filename)
            np.save(param_path, grad.cpu().numpy())
        
        # Save metadata as a separate .npy file
        metadata_filename = f"{gradient_filename_base}_metadata.npy"
        metadata_path = os.path.join(class_dir, metadata_filename)
        metadata_array = np.array([{
            "original_path": original_path,
            "label": labels[i].item(),
            "image_index": image_idx
        }], dtype=object)
        np.save(metadata_path, metadata_array)
    
    # Update metadata
    metadata["num_samples_processed"] += batch_size
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved gradients for batch of {batch_size} samples to {output_base_dir}")

def load_gradients_from_structure(gradients_dir="output/gradients"):
    """
    Load gradient tensors from the folder structure.
    
    Args:
        gradients_dir: Directory containing the gradient tensors
        
    Returns:
        Dictionary mapping class_folder/filename to gradient arrays and metadata
    """
    gradients_dict = {}
    
    if not os.path.exists(gradients_dir):
        print(f"Gradients directory {gradients_dir} does not exist")
        return gradients_dict
    
    for class_folder in os.listdir(gradients_dir):
        class_path = os.path.join(gradients_dir, class_folder)
        if os.path.isdir(class_path):
            # Group files by their base name (without param suffix)
            file_groups = {}
            for grad_file in os.listdir(class_path):
                if grad_file.endswith("_grad_param_0.npy"):
                    # Extract base name
                    base_name = grad_file.replace("_param_0.npy", "")
                    file_groups[base_name] = []
            
            # For each base name, collect all parameter files
            for base_name in file_groups:
                try:
                    gradient_data = {}
                    param_idx = 0
                    
                    # Load all parameter files
                    while True:
                        param_file = f"{base_name}_param_{param_idx}.npy"
                        param_path = os.path.join(class_path, param_file)
                        if os.path.exists(param_path):
                            gradient_data[f"param_{param_idx}"] = np.load(param_path)
                            param_idx += 1
                        else:
                            break
                    
                    # Load metadata file
                    metadata_file = f"{base_name}_metadata.npy"
                    metadata_path = os.path.join(class_path, metadata_file)
                    if os.path.exists(metadata_path):
                        gradient_data["metadata"] = np.load(metadata_path, allow_pickle=True)[0]
                    
                    relative_path = os.path.join(class_folder, base_name)
                    gradients_dict[relative_path] = gradient_data
                    
                except Exception as e:
                    print(f"Error loading gradient files for {base_name}: {e}")
    
    print(f"Loaded {len(gradients_dict)} gradient sets from {gradients_dir}")
    return gradients_dict

# %%
if __name__ == "__main__":
    main()

