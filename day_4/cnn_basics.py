# -------------------------------------------------------------
# Day 4 - PyTorch Neural Networks (Starter Template)
# Goal:
# Understand nn.Module lifecycle, build a CNN using nn.Conv2d
# and nn.MaxPool2d, and train it on a dummy dataset.
# -------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================
# Part A — nn.Module Lifecycle
# =============================================================

print("\n=== Part A: nn.Module Lifecycle ===")

# ---------------------------
# 1. Define a Simple Model
# ---------------------------

# TODO: Define a class that inherits from nn.Module

# ---------------------------
# 2. Instantiate and Inspect the Model
# ---------------------------

# TODO: Instantiate the model and print architecture

class Model(nn.Module): # Define class 
    def __init__(self, num_features):
      super().__init__()
      #Define a model 
      self.linear1 = nn.Linear(num_features, 5)
      self.relu = nn.ReLU()
      self.linear2 = nn.Linear(5, 2)
      self.relu = nn.ReLU()
      self.linear3 = nn.Linear(2, 1)
      self.sigmoid = nn.Sigmoid()


    def forward(self, x):
      out = self.linear(x)
      out = self.relu(out)
      out = self.linear2(out)
      out = self.relu(out)
      out = self.linear3(out)
      out = self.sigmoid(out)
      return out

model = Model(10) #10 input features
print(model) #Model Architecture

# ---------------------------
# 3. Train and Evaluate Lifecycle
# ---------------------------

# TODO: Train the model on a dummy dataset

# Creating a dummy dataset 


X_train = torch.randn(100,10)
y_train = torch.randint(0,2,(100,1)).float()  # Training data


X_val = torch.randn(20,10)
y_val = torch.randint(0,2,(20,1)).float()  # Validation data

# define model
model = Model(10)

# define loss function and optimizer
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Define important parameters learning rate and epochs
epochs = 25
lr = 0.1

# Define the loop
for epoch in range(epochs):
  # Training loop
  model.train()
  optimizer.zero_grad()

  y_pred = model(X_train)
  loss = loss_fn(y_pred, y_train)
  loss.backward()
  optimizer.step()

  # Validation loop
  model.eval()
  with torch.no_grad():
    y_pred = model(X_val)
    val_loss = loss_fn(y_pred, y_val)

  print(f'Epoch {epoch+1}/{epochs}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

# Accuracy 
y_pred = model(X_val)
y_pred_class = (y_pred > 0.5).float()
accuracy = (y_pred_class == y_val).float().mean()
print(f'Accuracy: {accuracy.item():.4f}')

# =============================================================
# Part B — Build a Simple CNN
# =============================================================

print("\n=== Part B: Build a Simple CNN ===")

# TODO: Define a CNN class with nn.Conv2d and nn.MaxPool2d

# TODO: Train the CNN on a dummy dataset


import torch
import torch.nn as nn
import torch.optim as optim

# Device
device = torch.device("cpu")
print("Using device:", device)

# -------------------------
# Dummy image dataset (grayscale 28x28 like MNIST)
# -------------------------
X_train = torch.randn(100, 1, 28, 28)   # shape: (N, C, H, W) 
y_train = torch.randint(0, 2, (100, 1)).float()  # shape: (N,1) float for BCEWithLogitsLoss

X_val = torch.randn(20, 1, 28, 28)
y_val = torch.randint(0, 2, (20, 1)).float()


# -------------------------
# Define a simple CNN
# -------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 1)   
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN()  
print(model)

# -------------------------
# Loss + Optimizer
# -------------------------
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# -------------------------
# Training and Evaluation
# -------------------------
epochs = 12

for epoch in range(1, epochs + 1):
    # ---- train ----
    model.train()
    optimizer.zero_grad()

    logits = model(X_train)                      # shape (N,1) logits 
    loss = loss_fn(logits, y_train)
    loss.backward()
    optimizer.step()

    # compute training accuracy
    with torch.no_grad():
        probs_train = torch.sigmoid(logits)
        preds_train = (probs_train > 0.5).float()
        train_acc = (preds_train == y_train).float().mean()

    # ---- validate ----
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
        val_loss = loss_fn(val_logits, y_val)

        probs_val = torch.sigmoid(val_logits)
        preds_val = (probs_val > 0.5).float()
        val_acc = (preds_val == y_val).float().mean()

    print(
        f"Epoch {epoch:02d}/{epochs} | "
        f"Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | "
        f"Train Acc: {train_acc.item():.4f} | Val Acc: {val_acc.item():.4f}"
    )

# =============================================================
# Expected Output Summary
# =============================================================

print("\n=== Script Must Print ===")
print("1. Model architecture")
print("2. Number of parameters")
print("3. Training and evaluation results")