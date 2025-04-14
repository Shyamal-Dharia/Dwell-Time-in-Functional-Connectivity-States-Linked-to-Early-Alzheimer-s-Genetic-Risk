import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix




class FMRI_MRIDataset(Dataset):

    def __init__(self, root_dir, transform=None, binary_classes=None):
        self.root_dir = root_dir
        self.transform = transform
        self.binary_classes = binary_classes

        # Validate binary_classes if provided
        if binary_classes is not None:
            if not (isinstance(binary_classes, (list, tuple)) and len(binary_classes) == 2):
                raise ValueError("binary_classes must be a list or tuple of two unique class labels.")
            # Create mapping: map the first element to 0 and second element to 1.
            self.binary_mapping = {binary_classes[0]: 0, binary_classes[1]: 1}
        
        # List all subject directories that start with "sub-"
        all_subject_dirs = sorted([
            d for d in os.listdir(root_dir)
            if d.startswith('sub-') and os.path.isdir(os.path.join(root_dir, d))
        ])
        
        # If binary_classes is specified, filter the subjects based on label.txt file
        if binary_classes is not None:
            self.subject_dirs = []
            for subject in all_subject_dirs:
                label_file = os.path.join(root_dir, subject, 'label.txt')
                if os.path.exists(label_file):
                    try:
                        with open(label_file, 'r') as f:
                            label = int(f.read().strip())
                    except Exception as e:
                        print(f"Could not read label for {subject}: {e}")
                        continue
                    if label in binary_classes:
                        self.subject_dirs.append(subject)
            if len(self.subject_dirs) == 0:
                raise RuntimeError("No subject directories found with the specified binary_classes.")
        else:
            self.subject_dirs = all_subject_dirs

        if len(self.subject_dirs) == 0:
            raise RuntimeError("No subject directories found in the provided root directory.")

    def __len__(self):
        return len(self.subject_dirs)

    def __getitem__(self, idx):
        subject_id = self.subject_dirs[idx]
        subject_path = os.path.join(self.root_dir, subject_id)
        
        # -------------------
        # Load fMRI data from func folder
        # -------------------
        func_path = os.path.join(subject_path, 'func')
        fmri_PA_file = os.path.join(
            func_path,
            f'{subject_id}_task-rest_dir-PA_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'
        )
        fmri_AP_file = os.path.join(
            func_path,
            f'{subject_id}_task-rest_dir-AP_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'
        )
        
        # Load PA file
        if not os.path.exists(fmri_PA_file):
            raise FileNotFoundError(f"fMRI PA file not found for {subject_id}: {fmri_PA_file}")
        fmri_PA_img = nib.load(fmri_PA_file)
        fmri_PA_data = fmri_PA_img.get_fdata()

        # Load AP file
        if not os.path.exists(fmri_AP_file):
            raise FileNotFoundError(f"fMRI AP file not found for {subject_id}: {fmri_AP_file}")
        fmri_AP_img = nib.load(fmri_AP_file)
        fmri_AP_data = fmri_AP_img.get_fdata()

        # Stack the two fMRI images along a new axis (channel dimension)
        fmri_data = np.stack((fmri_PA_data, fmri_AP_data), axis=0)

        # -------------------
        # Load MRI (anatomical) data from anat folder
        # -------------------
        anat_path = os.path.join(subject_path, 'anat')
        t1_file = os.path.join(
            anat_path,
            f'{subject_id}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz'
        )
        if not os.path.exists(t1_file):
            raise FileNotFoundError(f"Anatomical T1 file not found for {subject_id}: {t1_file}")
        t1_img = nib.load(t1_file)
        t1_data = t1_img.get_fdata()

        # Apply any provided transforms (should handle numpy arrays)
        if self.transform:
            fmri_data = self.transform(fmri_data)
            t1_data = self.transform(t1_data)

        # Convert numpy arrays to PyTorch tensors
        fmri_tensor = torch.tensor(fmri_data, dtype=torch.float32)
        t1_tensor = torch.tensor(t1_data, dtype=torch.float32)

        # -------------------
        # Load label from label.txt file in the subject folder
        # -------------------
        label_file = os.path.join(subject_path, 'label.txt')
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file not found for {subject_id}: {label_file}")
        with open(label_file, 'r') as f:
            try:
                label = int(f.read().strip())
            except ValueError:
                raise ValueError(f"Label file for {subject_id} does not contain a valid integer.")
        
        # If binary_classes is specified, remap the label to binary (0 or 1)
        if self.binary_classes is not None:
            if label not in self.binary_mapping:
                raise ValueError(f"Label {label} for {subject_id} is not in the specified binary_classes {self.binary_classes}")
            label = self.binary_mapping[label]
        
        # Return a tuple: (fMRI tensor, MRI tensor, label)
        return fmri_tensor, t1_tensor, label



def get_leave_one_out_loaders(dataset, leave_out_idx, batch_size=1, num_workers=0, shuffle_train=True):

    all_indices = list(range(len(dataset)))
    
    # Test indices: only the subject at leave_out_idx
    test_indices = [leave_out_idx]
    
    # Train indices: all other subject indices
    train_indices = all_indices.copy()
    train_indices.pop(leave_out_idx)
    
    # Create subsets
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    
    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader

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
        # Unpack batch data: fMRI data, MRI data, labels
        fmri, t1, labels = batch
        fmri = fmri.to(device)
        t1 = t1.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(fmri, t1)  # Model expects two inputs
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Multiply by batch size
        running_loss += loss.item() * fmri.size(0)
        
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
            fmri, t1, labels = batch
            fmri = fmri.to(device)
            t1 = t1.to(device)
            labels = labels.to(device)
            
            outputs = model(fmri, t1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * fmri.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    avg_loss = running_loss / len(test_loader.dataset)
    test_acc = accuracy_score(all_labels, all_preds)
    conf_mat = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, test_acc, conf_mat
