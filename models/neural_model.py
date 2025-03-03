import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
import mlflow

# Separate features (X) and target (y) for train and test
X_train_np = X_train.values  # Convert DataFrame to numpy array
y_train_np = y_train.values
X_test_np = X_test.values
y_test_np = y_test.values

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32).view(-1, 1)



# Define model
class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)  # Normalización Batch
        self.dropout1 = nn.Dropout(0.05)  # Dropout de 20%

        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)  # Normalización Batch
        self.dropout2 = nn.Dropout(0.1)  # Dropout de 20%

        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = torch.relu(  self.fc1(x)  )
        # x = self.dropout1(x)
        x = torch.relu( self.fc2(x)   ) 
        # x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Initialize model
model = SimpleModel(X_train.shape[1])


class RMSELoss(nn.Module):
    def forward(self, y_pred, y_true):
        return torch.sqrt(nn.MSELoss()(y_pred, y_true) + 1e-8)



# with torch.no_grad():
#     y_pred = model(X_test_tensor).numpy()

# y_test_np = y_test_tensor.numpy()
# print('RMSE:', mean_squared_error(y_test_np, y_pred, squared=False))
# print('R2 Score:', r2_score(y_test_np, y_pred))

def train_neural_network(self):
    # Initialize model
    model = SimpleModel(self.X_train.shape[1])  # Ensure SimpleModel is imported
    criterion = RMSELoss()  # Ensure RMSELoss is imported
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Define optimizer

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred = model(self.X_train)  # Forward pass
        loss = criterion(y_pred, self.y_train)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Log model with MLflow
    mlflow.pytorch.log_model(model, "neural_network_model")
