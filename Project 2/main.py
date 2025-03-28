# Import necessary libraries
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import cv2
import joblib
import warnings
from collections import Counter
import gc

warnings.filterwarnings('ignore')

# PyTorch related libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Set file paths
BASE_DIR = r"C:\Users\wan9y\OneDrive\Desktop"
DATA_DIR = r"C:\Users\wan9y\OneDrive\Desktop\downsampled_data-20250327T053121Z-001\downsampled_data"
METADATA_PATH = os.path.join(BASE_DIR, "metadata_BR00116991.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# Get available GPU device
def get_device():
    """Get available GPU device or CPU"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Detected {gpu_count} GPU devices")

        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_capability = torch.cuda.get_device_capability(i)
            print(f"GPU {i}: {gpu_name}, CUDA capability: {gpu_capability}")

        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1024 ** 3:.2f} GB")
        print(f"Current allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB")

        return device
    else:
        print("No GPU detected, using CPU")
        return torch.device("cpu")


# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Clear GPU cache
def clear_gpu_cache():
    """Clear GPU cache to free memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"GPU cache cleared, current allocated memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")


# 1. Load metadata and find image files
def load_data(metadata_path, data_dir):
    """Load metadata and match image paths using regex"""
    print(f"Loading metadata: {metadata_path}")
    metadata = pd.read_csv(metadata_path)

    # Check if data directory exists
    if not os.path.isdir(data_dir):
        print(f"Warning: Image directory not found: {data_dir}")
        parent_dir = os.path.dirname(data_dir)
        if os.path.isdir(parent_dir):
            print(f"Checking parent directory: {parent_dir}")

            # Find possible subdirectories
            for item in os.listdir(parent_dir):
                item_path = os.path.join(parent_dir, item)
                if os.path.isdir(item_path):
                    print(f"Found subdirectory: {item_path}")
                    if "downsampled_data" in item:
                        data_dir = item_path
                        print(f"Using image directory: {data_dir}")
                        break

    # Check directory again
    if not os.path.isdir(data_dir):
        print(f"Error: Data directory doesn't exist: {data_dir}")
        return None

    # List files in directory
    files = os.listdir(data_dir)
    print(f"{len(files)} files found in data directory")

    # Parse filenames and create mapping
    file_map = {}
    pattern = r'(r\d+c\d+f\d+)'  # Match rXXcXXfXX pattern

    for filename in files:
        match = re.search(pattern, filename)
        if match:
            key = match.group(1)
            file_map[key] = filename

    print(f"Created {len(file_map)} filename mappings")

    # Find corresponding image files for each metadata entry
    matched_files = []
    matched_count = 0

    for idx, row in metadata.iterrows():
        orig_filename = row['FileName_OrigRNA']
        match = re.search(pattern, orig_filename)

        if match:
            key = match.group(1)
            if key in file_map:
                matched_files.append(os.path.join(data_dir, file_map[key]))
                matched_count += 1
            else:
                matched_files.append(None)
        else:
            matched_files.append(None)

    print(f"Successfully matched {matched_count}/{len(metadata)} files")

    if matched_count == 0:
        print("Error: No files matched. Check file naming format.")
        return None

    # Update metadata
    metadata['image_path'] = matched_files
    # Filter out rows without matched files
    metadata = metadata[metadata['image_path'].notna()]

    print(f"Final valid samples: {len(metadata)}")
    print(f"Unique drug count: {metadata['Metadata_pert_iname'].nunique()}")

    # View drug distribution
    try:
        plt.figure(figsize=(15, 6))
        plt.title('Drug Treatment Distribution (Top 20)')
        metadata['Metadata_pert_iname'].value_counts()[:20].plot(kind='bar')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, 'drug_distribution.png'))
        plt.close()
    except Exception as e:
        print(f"Error plotting drug distribution: {e}")

    return metadata


# 2. Image processing - enhanced image reading and processing
def read_image(image_path, size=(224, 224)):
    """Read image and preprocess it for PyTorch"""
    try:
        # Try multiple methods to read image
        img = None
        error_messages = []

        # Method 1: tifffile (preferred for TIFF)
        try:
            import tifffile
            img = tifffile.imread(image_path)
        except ImportError:
            error_messages.append("tifffile library not installed")
        except Exception as e:
            error_messages.append(f"tifffile error: {str(e)}")

        # Method 2: PIL/Pillow
        if img is None:
            try:
                from PIL import Image
                pil_img = Image.open(image_path)
                img = np.array(pil_img)
            except Exception as e:
                error_messages.append(f"PIL error: {str(e)}")

        # Method 3: OpenCV
        if img is None:
            try:
                img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    error_messages.append("OpenCV couldn't read image")
            except Exception as e:
                error_messages.append(f"OpenCV error: {str(e)}")

        # Method 4: matplotlib
        if img is None:
            try:
                import matplotlib.pyplot as plt
                img = plt.imread(image_path)
            except Exception as e:
                error_messages.append(f"matplotlib error: {str(e)}")

        if img is None:
            print(f"Failed to read image {image_path}. Errors: {error_messages}")
            return None

        # Enhanced image preprocessing

        # 1. Apply CLAHE to improve contrast
        if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
            # Single channel image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            if len(img.shape) == 2:
                img = clahe.apply(img.astype(np.uint8))
            else:
                img = clahe.apply(img[:, :, 0].astype(np.uint8))
                img = img[:, :, np.newaxis]

        # 2. Process image channels
        if len(img.shape) == 2:  # Single channel image
            img = np.stack((img,) * 3, axis=-1)
        elif len(img.shape) == 3:
            if img.shape[2] == 1:  # Single channel 3D image
                img = np.concatenate([img, img, img], axis=2)
            elif img.shape[2] > 3:  # Multi-channel image, keep first 3
                img = img[:, :, :3]
            elif img.shape[2] == 3 and 'cv2' in str(type(img)):  # BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. Gaussian blur to reduce noise
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # 4. Resize
        img = cv2.resize(img, size)

        # 5. Normalize to [0,1]
        img = img.astype(np.float32) / 255.0

        return img

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


# 3. Prepare dataset - balanced class strategy
def prepare_dataset(metadata, min_samples_per_class=10, target_samples_per_class=20):
    """Prepare training and test datasets with balanced classes"""
    print("Preparing dataset with optimized class balancing...")

    # Filter out classes with few samples
    class_counts = metadata['Metadata_pert_iname'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples_per_class].index
    print(f"Classes before filtering: {len(class_counts)}")
    print(f"Classes after filtering: {len(valid_classes)}")
    print(f"Removed {len(class_counts) - len(valid_classes)} classes with insufficient samples")

    # Keep only classes with enough samples
    filtered_metadata = metadata[metadata['Metadata_pert_iname'].isin(valid_classes)].copy()
    print(f"Samples after filtering: {len(filtered_metadata)}")

    # Enhanced class balancing - equal samples for all classes
    balanced_metadata = []

    for drug_type in valid_classes:
        drug_samples = filtered_metadata[filtered_metadata['Metadata_pert_iname'] == drug_type]
        sample_count = len(drug_samples)

        # For large classes, downsample to target
        if sample_count > target_samples_per_class:
            drug_samples = drug_samples.sample(target_samples_per_class, random_state=42)
        # For small classes, oversample to target
        elif sample_count < target_samples_per_class:
            # Calculate repeat times needed
            repeat_times = int(np.ceil(target_samples_per_class / sample_count))
            # Repeat the dataframe
            repeated_samples = pd.concat([drug_samples] * repeat_times)
            # Sample to target count
            drug_samples = repeated_samples.sample(target_samples_per_class, random_state=42)

        balanced_metadata.append(drug_samples)

    # Merge all balanced samples
    balanced_metadata = pd.concat(balanced_metadata, ignore_index=True)
    print(f"Samples after balancing: {len(balanced_metadata)}")

    # View balanced class distribution
    balanced_class_counts = balanced_metadata['Metadata_pert_iname'].value_counts()
    plt.figure(figsize=(15, 6))
    plt.title('Balanced Drug Treatment Distribution')
    balanced_class_counts.plot(kind='bar')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'balanced_drug_distribution.png'))
    plt.close()

    # Encode labels
    le = LabelEncoder()
    balanced_metadata['encoded_label'] = le.fit_transform(balanced_metadata['Metadata_pert_iname'])

    # Split into train and test using stratified K-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(skf.split(balanced_metadata, balanced_metadata['encoded_label']))

    train_metadata = balanced_metadata.iloc[train_idx].reset_index(drop=True)
    test_metadata = balanced_metadata.iloc[test_idx].reset_index(drop=True)

    print(f"Training samples: {len(train_metadata)}")
    print(f"Testing samples: {len(test_metadata)}")

    # Check train and test class distributions
    print("Training class distribution:\n", train_metadata['Metadata_pert_iname'].value_counts())
    print("Testing class distribution:\n", test_metadata['Metadata_pert_iname'].value_counts())

    # Save label encoder
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))

    return train_metadata, test_metadata, le


# 4. Elastic transform for cell image augmentation
class ElasticTransform:
    """Apply elastic transform, suitable for cell image augmentation"""

    def __init__(self, alpha=50, sigma=5):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            # If input is PyTorch tensor, convert to NumPy array
            is_tensor = True
            img_np = img.numpy().transpose(1, 2, 0)
        else:
            is_tensor = False
            img_np = img

        shape = img_np.shape[:2]

        dx = np.random.rand(*shape) * 2 - 1
        dy = np.random.rand(*shape) * 2 - 1

        dx = cv2.GaussianBlur(dx, (0, 0), self.sigma) * self.alpha
        dy = cv2.GaussianBlur(dy, (0, 0), self.sigma) * self.alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        mapx = np.float32(x + dx)
        mapy = np.float32(y + dy)

        distorted = cv2.remap(img_np, mapx, mapy, cv2.INTER_LINEAR)

        if is_tensor:
            return torch.from_numpy(distorted.transpose(2, 0, 1))
        return distorted


# Random Erasing augmentation
class RandomErasing:
    """Randomly erase parts of the image, simulating missing cells or artifacts"""

    def __init__(self, p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, img):
        if np.random.random() > self.p:
            return img

        if isinstance(img, torch.Tensor):
            is_tensor = True
            img_np = img.numpy().transpose(1, 2, 0)
            h, w = img_np.shape[:2]
        else:
            is_tensor = False
            img_np = img.copy()
            h, w = img_np.shape[:2]

        # Select erasing region size
        area = h * w
        target_area = np.random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = np.random.uniform(self.ratio[0], self.ratio[1])

        # Calculate erasing region height and width
        ew = int(np.sqrt(target_area * aspect_ratio))
        eh = int(np.sqrt(target_area / aspect_ratio))

        if ew < w and eh < h:
            # Randomly select starting point
            x = np.random.randint(0, w - ew)
            y = np.random.randint(0, h - eh)

            # Erase region
            img_np[y:y + eh, x:x + ew, :] = self.value

        if is_tensor:
            return torch.from_numpy(img_np.transpose(2, 0, 1))
        return img_np


# 5. Cell-specific data transformations
def get_cell_transforms():
    """Create data augmentations specific to cell images"""

    # Training transforms - stronger augmentations
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),  # Cells can be oriented any way
        transforms.RandomAffine(
            degrees=0,
            translate=(0.15, 0.15),
            scale=(0.75, 1.25),
            shear=15
        ),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05
        ),
        # Gaussian blur to simulate different focus levels
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        # Custom augmentations
        ElasticTransform(alpha=60, sigma=6),
        RandomErasing(p=0.3, scale=(0.02, 0.15)),
        # Normalization
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Test transforms - only essential processing
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, test_transform


# 6. Enhanced dataset class
class EnhancedCellDataset(Dataset):
    """Enhanced cell image dataset with caching"""

    def __init__(self, metadata, transform=None, phase='train'):
        self.metadata = metadata
        self.transform = transform
        self.phase = phase
        self.image_cache = {}  # Cache loaded images
        self.cache_hits = 0
        self.cache_misses = 0

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = row['image_path']
        label = row['encoded_label']

        # Check if image is already cached
        if img_path in self.image_cache:
            img = self.image_cache[img_path]
            self.cache_hits += 1
        else:
            # Read image
            img = read_image(img_path)
            self.cache_misses += 1

            if img is None:
                # If image reading failed, return a zero-filled image
                img = np.zeros((224, 224, 3), dtype=np.float32)

            # Add image to cache
            if len(self.image_cache) < 500:  # Limit cache size
                self.image_cache[img_path] = img

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        return img, label

    # Print cache hit rate
    def get_cache_stats(self):
        total = self.cache_hits + self.cache_misses
        if total > 0:
            hit_rate = self.cache_hits / total * 100
            return f"Cache hit rate: {hit_rate:.2f}% ({self.cache_hits}/{total})"
        return "Cache not yet accessed"


# 7. Channel Attention Module
class ChannelAttention(nn.Module):
    """Channel attention module"""

    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return torch.sigmoid(out)


# 8. Spatial Attention Module
class SpatialAttention(nn.Module):
    """Spatial attention module"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        # Compute average and max along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate features
        x = torch.cat([avg_out, max_out], dim=1)

        # Apply convolution and sigmoid
        x = self.conv(x)
        return torch.sigmoid(x)


# 9. CBAM (Convolutional Block Attention Module)
class CBAM(nn.Module):
    """CBAM attention module, combining channel and spatial attention"""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()

        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # Apply channel attention
        x = x * self.channel_attention(x)

        # Apply spatial attention
        x = x * self.spatial_attention(x)

        return x


# 10. Advanced Cell Classification Network
class AdvancedCellNet(nn.Module):
    """Enhanced cell classification network with advanced backbone and attention"""

    def __init__(self, num_classes, backbone='resnet50', pretrained=True, dropout_rate=0.5):
        super(AdvancedCellNet, self).__init__()

        self.backbone_name = backbone

        # Select backbone network
        if backbone == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif backbone == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            self.feature_dim = 512
        elif backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif backbone == 'resnet101':
            base_model = models.resnet101(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            # Default to ResNet50
            print(f"Unknown backbone '{backbone}', using ResNet50")
            base_model = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048

        # Remove final layers
        self.base_model = nn.Sequential(*list(base_model.children())[:-2])

        # Add CBAM attention
        self.attention = CBAM(self.feature_dim, reduction=16)

        # Improved classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate - 0.1),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate - 0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Extract features
        features = self.base_model(x)

        # Apply attention
        features = self.attention(features)

        # Classification
        output = self.classifier(features)

        return output


# 11. Label Smoothing Cross Entropy Loss
class LabelSmoothingLoss(nn.Module):
    """Label smoothing cross entropy loss to prevent overconfidence"""

    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # Create smoothed labels
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        # Calculate loss
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# 12. Advanced training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None,
                num_epochs=50, device='cuda', patience=10, label_smoothing=0.1,
                multi_gpu=False):
    """Training function with advanced techniques"""

    # Set up mixed precision training
    scaler = GradScaler()

    # Use DataParallel for multi-GPU training
    if multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training!")
        model = nn.DataParallel(model)

    # Create label smoothing loss if needed
    if label_smoothing > 0:
        if hasattr(model, 'module'):  # For DataParallel
            num_classes = model.module.classifier[-1].out_features
        else:
            num_classes = model.classifier[-1].out_features

        smoothing_criterion = LabelSmoothingLoss(classes=num_classes,
                                                 smoothing=label_smoothing)

    # Early stopping setup
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience_counter = 0
    best_model_weights = None

    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }

    # Move model to device
    model = model.to(device)

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 30)

        # Training phase
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        # Progress bar
        progress_bar = tqdm(train_loader, desc="Training")

        for inputs, labels in progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Zero gradients
            optimizer.zero_grad()

            # Mixed precision training
            with autocast():
                outputs = model(inputs)

                # Use label smoothing or original loss
                if label_smoothing > 0:
                    loss = smoothing_criterion(outputs, labels)
                else:
                    loss = criterion(outputs, labels)

            # Backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update learning rate
            if scheduler is not None and isinstance(scheduler, OneCycleLR):
                scheduler.step()

            # Statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)

            # Collect predictions and labels for F1 score
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            progress_bar.set_postfix(loss=loss.item())

            # Free memory
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        # Calculate training metrics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}')

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['train_f1'].append(epoch_f1)

        # Validation phase
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Statistics
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)

                # Collect predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Free memory
                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()

        # Calculate validation metrics
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

        print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}')

        history['val_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc)
        history['val_f1'].append(epoch_f1)

        # Update learning rate (if not OneCycleLR)
        if scheduler is not None and not isinstance(scheduler, OneCycleLR):
            if isinstance(scheduler, CosineAnnealingWarmRestarts):
                scheduler.step()
            else:
                scheduler.step(epoch_loss)

        # Early stopping check - using weighted combo of F1 and accuracy
        current_score = 0.7 * epoch_f1 + 0.3 * epoch_acc
        best_score = 0.7 * best_val_f1 + 0.3 * best_val_acc

        if current_score > best_score:
            best_val_acc = epoch_acc
            best_val_f1 = epoch_f1
            best_val_loss = epoch_loss

            # Save best model
            if hasattr(model, 'module'):  # For DataParallel
                best_model_weights = model.module.state_dict().copy()
                torch.save(model.module.state_dict(), os.path.join(MODEL_DIR, 'best_model.pth'))
            else:
                best_model_weights = model.state_dict().copy()
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_model.pth'))

            patience_counter = 0
            print(f"Saved new best model, Acc: {best_val_acc:.4f}, F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Clear GPU memory after each epoch
        clear_gpu_cache()

    print(f"Training completed. Best val accuracy: {best_val_acc:.4f}, Best F1: {best_val_f1:.4f}")

    # Load best model weights
    if hasattr(model, 'module'):  # For DataParallel
        model.module.load_state_dict(best_model_weights)
    else:
        model.load_state_dict(best_model_weights)

    return model, history


# 13. Model evaluation
def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model and return detailed metrics"""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []  # Store probabilities

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

            # Free memory
            del inputs, labels, outputs, probs, preds
            torch.cuda.empty_cache()

    all_probs = np.vstack(all_probs)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")

    return np.array(all_labels), np.array(all_preds), all_probs


# 14. Ensemble training
def train_ensemble(train_metadata, test_metadata, label_encoder, num_models=5,
                   model_configs=None, use_multi_gpu=True, num_workers=4,
                   amp_enabled=True):
    """Train multiple models and ensemble them for better performance"""

    # Default model configurations
    if model_configs is None:
        model_configs = [
            {'backbone': 'resnet50', 'dropout_rate': 0.5},
            {'backbone': 'resnet101', 'dropout_rate': 0.5},
            {'backbone': 'resnet34', 'dropout_rate': 0.5},
            {'backbone': 'resnet50', 'dropout_rate': 0.6},
            {'backbone': 'resnet101', 'dropout_rate': 0.6}
        ]

    # Fill in missing configs
    while len(model_configs) < num_models:
        model_configs.append(model_configs[-1])

    # Set device
    device = get_device()

    # Check if multi-GPU is available
    multi_gpu = use_multi_gpu and torch.cuda.device_count() > 1
    if multi_gpu:
        print(f"Enabling multi-GPU training with {torch.cuda.device_count()} GPUs")

    # Get data transforms
    train_transform, test_transform = get_cell_transforms()

    # Create test dataset
    test_dataset = EnhancedCellDataset(
        test_metadata,
        transform=test_transform,
        phase='test'
    )

    # Create test dataloader
    batch_size = 32 if torch.cuda.is_available() else 16  # Larger batch for GPU
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True  # Speed up GPU data transfer
    )

    # Print GPU memory status
    if torch.cuda.is_available():
        print(f"GPU memory after dataloader creation: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    # Train multiple models
    models = []
    all_predictions = []
    all_probabilities = []

    for i in range(num_models):
        print(f"\nTraining model {i + 1}/{num_models} - {model_configs[i]['backbone']}")

        # Set different random seed
        set_seed(42 + i)

        # Create training dataset
        train_dataset = EnhancedCellDataset(
            train_metadata,
            transform=train_transform,
            phase='train'
        )

        # Calculate sampling weights for class balance
        label_counts = Counter(train_metadata['encoded_label'])
        weights = [1.0 / label_counts[label] for label in train_metadata['encoded_label']]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

        # Create training dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,  # Use weighted sampling
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

        # Create validation dataloader
        val_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        # Clear memory
        clear_gpu_cache()

        # Create model
        model = AdvancedCellNet(
            num_classes=len(label_encoder.classes_),
            backbone=model_configs[i]['backbone'],
            pretrained=True,
            dropout_rate=model_configs[i]['dropout_rate']
        )

        # Set loss function
        criterion = nn.CrossEntropyLoss()

        # Set optimizer with different learning rates
        param_groups = [
            {'params': model.base_model.parameters(), 'lr': 1e-4},  # Backbone
            {'params': model.attention.parameters(), 'lr': 5e-4},  # Attention
            {'params': model.classifier.parameters(), 'lr': 1e-3}  # Classifier
        ]

        optimizer = optim.AdamW(param_groups, weight_decay=1e-4)

        # Set learning rate scheduler - OneCycleLR
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[1e-4, 5e-4, 1e-3],  # Match param_groups
            steps_per_epoch=len(train_loader),
            epochs=30,
            pct_start=0.3,  # Warmup period
            div_factor=10.0,  # Initial LR = max_lr / div_factor
            final_div_factor=100.0  # Final LR = max_lr / (div_factor * final_div_factor)
        )

        # Train the model
        model, history = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler=scheduler,
            num_epochs=30,
            device=device,
            patience=10,
            label_smoothing=0.1,
            multi_gpu=multi_gpu
        )

        # Save model
        if hasattr(model, 'module'):  # For DataParallel
            torch.save(model.module.state_dict(), os.path.join(MODEL_DIR, f'model_{i + 1}.pth'))
        else:
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'model_{i + 1}.pth'))

        # Evaluate model
        y_test, y_pred, probs = evaluate_model(model, test_loader, device)
        all_predictions.append(y_pred)
        all_probabilities.append(probs)

        # Print performance
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"Model {i + 1} - {model_configs[i]['backbone']} Test Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        # Save model
        models.append(model)

        # Check GPU memory
        if torch.cuda.is_available():
            print(f"GPU memory after model {i + 1}: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

        # Clear memory
        clear_gpu_cache()

    # Soft voting ensemble
    stacked_probs = np.mean(all_probabilities, axis=0)
    ensemble_preds = np.argmax(stacked_probs, axis=1)

    # Get true labels
    y_test = []
    for _, labels in test_loader:
        y_test.extend(labels.numpy())

    # Calculate ensemble performance
    ensemble_accuracy = accuracy_score(y_test, ensemble_preds)
    ensemble_f1 = f1_score(y_test, ensemble_preds, average='weighted')

    print(f"\nEnsemble model test accuracy: {ensemble_accuracy:.4f}, F1: {ensemble_f1:.4f}")

    return models, y_test, ensemble_preds, stacked_probs


# 15. Visualization
def visualize_results(y_test, y_pred, label_encoder, probs=None, history=None):
    """Visualize model results with enhanced analysis"""

    # Ensure y_test and y_pred are numpy arrays
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    # 1. If training history exists, plot training curves
    if history:
        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        plt.plot(history['train_acc'])
        plt.plot(history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.subplot(1, 3, 2)
        plt.plot(history['train_loss'])
        plt.plot(history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        # Add F1 curve
        plt.subplot(1, 3, 3)
        plt.plot(history['train_f1'])
        plt.plot(history['val_f1'])
        plt.title('Model F1 Score')
        plt.ylabel('F1 Score')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, 'training_history.png'))
        plt.close()

    # 2. Calculate per-class performance - fixed one-vs-rest F1
    class_performance = {}
    for i in np.unique(y_test):
        class_mask = (y_test == i)
        if np.sum(class_mask) > 0:
            class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])

            # Create binary labels (current class = positive, others = negative)
            y_true_binary = (y_test == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)

            # Use weighted average for this binary setting
            try:
                class_f1 = f1_score(y_true_binary, y_pred_binary, average='weighted')
            except:
                # If any issues, use accuracy as fallback
                class_f1 = class_acc

            try:
                class_name = label_encoder.inverse_transform([i])[0]
            except:
                class_name = f"Class_{i}"

            class_performance[class_name] = {
                'accuracy': class_acc,
                'f1': class_f1,
                'count': np.sum(class_mask)
            }

    # Plot class performance by F1
    plt.figure(figsize=(15, 10))
    top_classes_f1 = {k: v['f1'] for k, v in
                      sorted(class_performance.items(), key=lambda item: item[1]['f1'], reverse=True)[:20]}
    sns.barplot(x=list(top_classes_f1.values()), y=list(top_classes_f1.keys()))
    plt.title('Class F1 Scores (Top 20)')
    plt.xlabel('F1 Score')
    plt.ylabel('Drug Type')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'class_f1.png'))
    plt.close()

    # Plot class accuracy
    plt.figure(figsize=(15, 10))
    top_classes_acc = {k: v['accuracy'] for k, v in
                       sorted(class_performance.items(), key=lambda item: item[1]['accuracy'], reverse=True)[:20]}
    sns.barplot(x=list(top_classes_acc.values()), y=list(top_classes_acc.keys()))
    plt.title('Class Accuracy (Top 20)')
    plt.xlabel('Accuracy')
    plt.ylabel('Drug Type')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'class_accuracy.png'))
    plt.close()

    # 3. Generate confusion matrix
    try:
        # Get unique classes in test set
        unique_classes = np.unique(y_test)

        # Limit displayed classes
        top_classes = min(20, len(unique_classes))

        # Count samples per class
        class_counts = {}
        for cls in unique_classes:
            class_counts[int(cls)] = np.sum(y_test == cls)

        # Sort by sample count to get top classes
        top_indices = []
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        for cls, _ in sorted_classes[:top_classes]:
            top_indices.append(int(cls))

        # Generate full confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Create filtered matrix for top classes
        cm_filtered = np.zeros((len(top_indices), len(top_indices)), dtype=int)

        # Fill filtered matrix
        for i, cls_i in enumerate(top_indices):
            for j, cls_j in enumerate(top_indices):
                # Find position in original matrix
                idx_i = np.where(unique_classes == cls_i)[0][0]
                idx_j = np.where(unique_classes == cls_j)[0][0]
                cm_filtered[i, j] = cm[idx_i, idx_j]

        # Get class names
        class_names = []
        for i in top_indices:
            try:
                class_names.append(label_encoder.inverse_transform([int(i)])[0])
            except:
                class_names.append(f"Class_{i}")

        # Create normalized confusion matrix
        cm_norm = cm_filtered.astype('float') / cm_filtered.sum(axis=1)[:, np.newaxis]

        # Plot raw confusion matrix
        plt.figure(figsize=(18, 15))
        sns.heatmap(cm_filtered, annot=True, fmt='d',
                    xticklabels=class_names, yticklabels=class_names, cmap='Blues')
        plt.title('Confusion Matrix (Top 20 Classes)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix.png'))
        plt.close()

        # Plot normalized confusion matrix
        plt.figure(figsize=(18, 15))
        sns.heatmap(cm_norm, annot=True, fmt='.2f',
                    xticklabels=class_names, yticklabels=class_names, cmap='Blues')
        plt.title('Normalized Confusion Matrix (Top 20 Classes)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix_normalized.png'))
        plt.close()

    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")
        import traceback
        traceback.print_exc()

    # 4. Plot ROC and PR curves if probabilities provided
    if probs is not None:
        try:
            from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
            from sklearn.preprocessing import label_binarize

            # Binarize labels
            classes = np.unique(y_test)
            n_classes = len(classes)
            y_test_bin = label_binarize(y_test, classes=classes)

            # Compute ROC curve and ROC area
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], probs[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Plot top 10 classes ROC curves
            plt.figure(figsize=(15, 10))

            # Get class names
            class_names = []
            for i in range(min(10, n_classes)):
                cls_idx = int(classes[i])
                try:
                    class_names.append(label_encoder.inverse_transform([cls_idx])[0])
                except:
                    class_names.append(f"Class_{cls_idx}")

            for i, name in enumerate(class_names):
                plt.plot(fpr[i], tpr[i], lw=2,
                         label=f'{name} (AUC = {roc_auc[i]:.2f})')

            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves for Major Drug Classes')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(BASE_DIR, 'roc_curves.png'))
            plt.close()

            # Compute PR curve
            precision = dict()
            recall = dict()
            avg_precision = dict()

            for i in range(n_classes):
                precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], probs[:, i])
                avg_precision[i] = average_precision_score(y_test_bin[:, i], probs[:, i])

            # Plot top 10 classes PR curves
            plt.figure(figsize=(15, 10))

            for i, name in enumerate(class_names):
                plt.plot(recall[i], precision[i], lw=2,
                         label=f'{name} (AP = {avg_precision[i]:.2f})')

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('PR Curves for Major Drug Classes')
            plt.legend(loc="lower left")
            plt.tight_layout()
            plt.savefig(os.path.join(BASE_DIR, 'pr_curves.png'))
            plt.close()

        except Exception as e:
            print(f"Error plotting ROC curves: {e}")
            import traceback
            traceback.print_exc()

    # 5. Classification report
    try:
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)

        # Convert to DataFrame
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(BASE_DIR, 'classification_report.csv'))
        print("Classification report successfully generated")

        # Plot classification summary
        plt.figure(figsize=(15, 8))
        metrics = ['precision', 'recall', 'f1-score']
        summary = report_df.loc[['macro avg', 'weighted avg'], metrics]
        sns.heatmap(summary, annot=True, cmap='Blues', fmt='.3f', cbar=False)
        plt.title('Classification Performance Summary')
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, 'classification_summary.png'))
        plt.close()

    except Exception as e:
        print(f"Error generating classification report: {e}")
        import traceback
        traceback.print_exc()


# Main function
def main():
    print("===== Cell Drug Response Classification - PyTorch =====")

    # Get device
    device = get_device()

    # Set random seed
    set_seed(42)

    # 1. Load data
    print("\n[Step 1] Loading metadata and finding image files")
    metadata = load_data(METADATA_PATH, DATA_DIR)
    if metadata is None:
        print("Error: Metadata loading failed, check file paths")
        return

    # 2. Prepare dataset
    print("\n[Step 2] Preparing dataset with class balancing")
    train_metadata, test_metadata, label_encoder = prepare_dataset(
        metadata,
        min_samples_per_class=5,
        target_samples_per_class=20
    )

    # 3. Train ensemble model
    print("\n[Step 3] Training ensemble model with multiple architectures")
    model_configs = [
        {'backbone': 'resnet50', 'dropout_rate': 0.5},
        {'backbone': 'resnet101', 'dropout_rate': 0.5},
        {'backbone': 'resnet34', 'dropout_rate': 0.5},
        {'backbone': 'resnet50', 'dropout_rate': 0.6},
        {'backbone': 'resnet101', 'dropout_rate': 0.6}
    ]

    # Determine CPU cores for data loading
    num_workers = min(os.cpu_count(), 4)
    print(f"Using {num_workers} worker threads for data loading")

    # Check for multi-GPU
    use_multi_gpu = torch.cuda.device_count() > 1

    # Train multiple models and ensemble
    models, y_test, ensemble_preds, ensemble_probs = train_ensemble(
        train_metadata,
        test_metadata,
        label_encoder,
        num_models=5,
        model_configs=model_configs,
        use_multi_gpu=use_multi_gpu,
        num_workers=num_workers
    )

    # 4. Visualize results
    print("\n[Step 4] Visualizing results and analysis")
    visualize_results(y_test, ensemble_preds, label_encoder, probs=ensemble_probs)

    print("\n===== Project completed! =====")
    print(f"All results saved to: {BASE_DIR}")
    print(f"Models saved to: {MODEL_DIR}")

    # Final performance
    accuracy = accuracy_score(y_test, ensemble_preds)
    f1 = f1_score(y_test, ensemble_preds, average='weighted')
    print(f"Final ensemble model accuracy: {accuracy:.4f}, F1 score: {f1:.4f}")


if __name__ == "__main__":
    main()