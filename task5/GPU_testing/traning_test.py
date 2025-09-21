import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

print("=== Testing Model Training on GPU ===")

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training device: {device}")

# Simple neural network for 4GB VRAM
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Create dummy dataset
print("Creating dataset...")
X = torch.randn(5000, 100)  # 5000 samples, 100 features
y = torch.randint(0, 10, (5000,))  # 10 classes
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize model
model = SimpleNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training loop
print("\nStarting training...")
start_time = time.time()

for epoch in range(10):
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Clear GPU cache every 10 batches (good for 4GB VRAM)
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    
    print(f"Epoch {epoch+1:2d}/10 | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
    
    # Memory monitoring
    if epoch % 3 == 0:
        print(f"  GPU Memory - Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f}MB, "
              f"Cached: {torch.cuda.memory_reserved(0) / 1024**2:.1f}MB")

training_time = time.time() - start_time
print(f"\n✅ Training completed in {training_time:.2f} seconds!")
print("🚀 Your RTX 2050 is ready for deep learning projects!")