import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path

class FMRI_TorchDataset(Dataset):
    """
    A PyTorch Dataset for combined fMRI and MRI data along with risk labels 
    from pre-saved .pt files.

    Each sample is stored in a .pt file (saved using torch.save) containing:
      - 'fmri': a torch.Tensor of shape [2, 560, voxels] (fMRI data)
      - 't1': a torch.Tensor of the anatomical T1 image.
      - 'label': a torch.Tensor or int with the risk label.

    Optionally, binary_classes can be specified to only load subjects with the 
    desired labels and remap them to 0 and 1.
    """
    def __init__(self, root_dir, transform=None, binary_classes=None):
        """
        Args:
            root_dir (str): Path to the directory containing the .pt files 
                            (e.g., torch_derivatives).
            transform (callable, optional): A function/transform to apply to each sample.
            binary_classes (list or tuple, optional): Two class labels to include for binary classification.
                If provided, only subjects with labels in this list are included and their labels are remapped.
                For example, binary_classes=[0, 1] will include subjects with labels 0 and 1.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.binary_classes = binary_classes

        # Validate binary_classes if provided
        if binary_classes is not None:
            if not (isinstance(binary_classes, (list, tuple)) and len(binary_classes) == 2):
                raise ValueError("binary_classes must be a list or tuple of two unique class labels.")
            # Map the first element to 0 and the second element to 1.
            self.binary_mapping = {binary_classes[0]: 0, binary_classes[1]: 1}
        
        # List all .pt files that start with "sub-"
        all_files = sorted(self.root_dir.glob("sub-*.pt"))
        if binary_classes is not None:
            self.files = []
            # Load each file to check the label before including it in the dataset.
            for file in all_files:
                try:
                    data = torch.load(file, weights_only=False)
                    # Ensure the label is an integer value.
                    label = int(data['label'].item()) if isinstance(data['label'], torch.Tensor) else int(data['label'])
                except Exception as e:
                    print(f"Skipping {file.name} due to error: {e}")
                    continue
                if label in binary_classes:
                    self.files.append(file)
            if len(self.files) == 0:
                raise RuntimeError("No subject files found with the specified binary_classes.")
        else:
            self.files = all_files
        
        if len(self.files) == 0:
            raise RuntimeError("No .pt files found in the provided root directory.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pt_file = self.files[idx]
        data = torch.load(pt_file,weights_only=False) 
        fmri_tensor = data['fmri']  # Already a torch.Tensor in float16
        t1_tensor = data['t1']      # Already a torch.Tensor in float16
        label = data['label']       # Could be a torch.Tensor or int
        
        # If binary_classes is specified, remap the label.
        if self.binary_classes is not None:
            label_int = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
            if label_int not in self.binary_mapping:
                raise ValueError(f"Label {label_int} in file {pt_file.name} is not in the specified binary_classes {self.binary_classes}")
            label = torch.tensor(self.binary_mapping[label_int], dtype=torch.int64)
        else:
            # Ensure label is an integer tensor.
            label = torch.tensor(int(label), dtype=torch.int64)
        
        
        return fmri_tensor, t1_tensor, label


def get_leave_one_out_loaders(dataset, leave_out_idx, batch_size=1):

    all_indices = list(range(len(dataset)))
    
    # Use the subject at leave_out_idx as the test set.
    test_indices = [leave_out_idx]
    train_indices = all_indices.copy()
    train_indices.pop(leave_out_idx)
    
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,     num_workers=64,              # Set based on your CPU cores
                                pin_memory=True,            # Helpful if training on GPU
                                persistent_workers=True)          # Number of batches preloaded by each worker (if applicable))
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=64,pin_memory=True,            # Helpful if training on GPU
                                persistent_workers=True)
    
    return train_loader, test_loader


# -----------------------------------------------
# Example usage:
# -----------------------------------------------
# Set the path to your torch derivatives folder (where the .pt files are saved)
# data_dir = '/data/s.dharia-ra/PEARL/PEARL/torch_derivatives'

# # Initialize the dataset. You can specify binary_classes if desired.
# dataset = FMRI_TorchDataset(root_dir=data_dir, binary_classes=[0, 1])

# num_subjects = len(dataset)
# print(f"Total number of subjects: {num_subjects}")

# # Create leave-one-out DataLoaders and iterate over them.
# for leave_out_idx in range(num_subjects):
#     train_loader, test_loader = get_leave_one_out_loaders(dataset, leave_out_idx, batch_size=5)
#     print(f"\nLOOCV iteration with subject {leave_out_idx} as test set:")
#     print(f"Train set size: {len(train_loader.dataset)}; Test set size: {len(test_loader.dataset)}")
    
#     print("Starting Epochs...")
#     # Here you would typically run your training loop.
#     # For demonstration, we'll just print the labels from the test loader.
#     for epoch in range(10):
#         print(f"Epoch {epoch + 1}")
#         for fmri, t1, label in train_loader:
#             print(f"Train batch label: {fmri.shape}")
    




# # Set the path to your selected derivatives folder
# data_dir = '/data/s.dharia-ra/PEARL/PEARL/derivatives_selected'

# dataset = FMRI_MRIDataset(root_dir=data_dir, binary_classes=[0, 1])

# num_subjects = len(dataset)
# print(f"Total number of subjects: {num_subjects}")

# for leave_out_idx in range(num_subjects):
#     train_loader, test_loader = get_leave_one_out_loaders(dataset, leave_out_idx, batch_size=1)
#     print(f"\nLOOCV iteration with subject {leave_out_idx} as test set:")
#     print(f"Train set size: {len(train_loader.dataset)}; Test set size: {len(test_loader.dataset)}")
    
#     # Example: iterate through the test loader and print the label
#     for fmri, t1, label in test_loader:
#         print("Test subject label:", label.item())


def train_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    for batch in train_loader:
        # print("Batch data:")
        # Unpack batch data: fMRI data, MRI data, labels
        fmri, _, labels = batch
        #change to float16
        fmri = fmri.float()
        fmri = fmri.to(device)
        labels = labels.long()
        labels = labels.to(device)
        optimizer.zero_grad()
        # if torch.isnan(fmri).any() or torch.isinf(fmri).any():
        #     print("NaNs or Infs in input!")

        outputs = model(fmri)  # Model expects two inputs
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Multiply by batch size
        running_loss += loss.item() * fmri.size(0)
        
        # print(f"Batch loss: {loss.item()}")
        # Get predictions (assumes outputs are logits over classes)
        preds = torch.argmax(outputs, dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return epoch_loss, epoch_acc, epoch_f1

def evaluate(model, criterion, test_loader, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in test_loader:
            fmri, _, labels = batch
            fmri = fmri.float()
            fmri = fmri.to(device)
            labels = labels.long()
            labels = labels.to(device)
            
            outputs = model(fmri)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * fmri.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    avg_loss = running_loss / len(test_loader.dataset)
    test_acc = accuracy_score(all_labels, all_preds)
    conf_mat = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    
    return avg_loss, test_acc, conf_mat
