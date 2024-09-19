import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from swin_transformer_model import SwinTransformer
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

# Define the transformations for the training and testing datasets
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 for Swin Transformer
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomCrop(224, padding=4),  # Random crop with padding
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR-10 dataset from local files
train_dataset = datasets.CIFAR10(
    root='data',
    train=True,
    transform=transform_train,
    download=False  # Assume data is already downloaded
)

test_dataset = datasets.CIFAR10(
    root='data',
    train=False,
    transform=transform_test,
    download=False  # Assume data is already downloaded
)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=128,  # Increase batch size
    shuffle=True,
    num_workers=16, 
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=16, 
    pin_memory=True
)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = SwinTransformer(
    hidden_dim=96,
    layers=(2, 2, 18, 2), #SWIN-T Layers = (2, 2, 6, 2), SWIN-B Layers = (2, 2, 18, 2)
    heads=(4, 8, 16, 32), #SWIN-T Heads = (3, 6, 12, 24), SWIN-B Heads = (4, 8, 16, 32)
    channels=3,
    num_classes=10,
    head_dim=32,
    window_size=7,
    downscaling_factors=(4, 2, 2, 2),
    relative_pos_embedding=True
)

# Use DataParallel
model = nn.DataParallel(model)

# Move model to the device
model = model.to(device)

# Enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Initialize GradScaler for mixed precision
scaler = GradScaler()

# Early stopping parameters
patience = 5
tolerance = 1e-4  # Define a small tolerance value
best_val_loss = float('inf')
epochs_no_improve = 0
early_stop = False

# Training loop
num_epochs = 50  # Increase number of epochs
for epoch in tqdm(range(num_epochs)):
    model.train()
    running_loss = 0.0
    
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    scheduler.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

    val_loss /= len(test_loader)
    print(f'Validation Loss: {val_loss}')

    # Check for early stopping with tolerance
    if best_val_loss - val_loss > tolerance:
        best_val_loss = val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    # if epochs_no_improve >= patience:
    #     print('Early stopping!')
    #     early_stop = False
    #     break

    # if early_stop:
    #     break

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        with autocast():
            outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test images: {100 * correct / total} %')
