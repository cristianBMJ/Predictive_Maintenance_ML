#!/usr/bin/env python
# coding: utf-8

# # **Model Training: Gas Turbina**

# Since the EDA showed some non-linearity in Turbine Energy Yield (TEY) with respect to features, we will use models that capture these non-linearities.
# 
# Models:
# 
# - Randon Forests
# - XGboost
# - Neural Networks
# - Ensemble Methods

# In[4]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


from xgboost import XGBRegressor
import torch


import pandas as pd 


# data's 2011
gt_2011 = pd.read_csv('../data/gas_turbine_emision/gt_2011.csv' )
gt_2011['Year'] = 2011

# data's 2012
gt_2012 = pd.read_csv('../data/gas_turbine_emision/gt_2012.csv' )
gt_2012['Year'] = 2014

# data's 2013
gt_2013 = pd.read_csv('../data/gas_turbine_emision/gt_2013.csv' )
gt_2013['Year'] = 2013

# data's 2014
gt_2014 = pd.read_csv('../data/gas_turbine_emision/gt_2014.csv' )
gt_2014['Year'] = 2014

# data's 2015
gt_2015 = pd.read_csv('../data/gas_turbine_emision/gt_2015.csv' )
gt_2015['Year'] = 2015


gt = pd.concat([gt_2011, gt_2012, gt_2013, gt_2014, gt_2015], ignore_index=True)



# In[5]:


# simple feature engineer

gt['TAT_TIT_Ratio'] = gt['TAT'] / gt['TIT']


# ### **Random Forests**

# In[6]:


# Split data
X = gt.drop(columns=['TEY'])
y = gt['TEY']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print('RMSE:', mean_squared_error(y_test, y_pred, squared=False))
print('R2 Score:', r2_score(y_test, y_pred))


# ### **XGboost**

# In[7]:


from xgboost import XGBRegressor

# Train model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print('RMSE:', mean_squared_error(y_test, y_pred, squared=False))
print('R2 Score:', r2_score(y_test, y_pred))


# ### **SVR**

# In[8]:


from sklearn.svm import SVR

# Train model
model = SVR(kernel='rbf')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print('RMSE:', mean_squared_error(y_test, y_pred, squared=False))
print('R2 Score:', r2_score(y_test, y_pred))


# ### **Neural Network**
# 





import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score

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

criterion = RMSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)

# DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Train model
num_epochs = 50
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        y_pred = model(batch_X)
        loss = criterion(y_pred, batch_y)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    
# Evaluate
with torch.no_grad():
    y_pred = model(X_test_tensor).numpy()

y_test_np = y_test_tensor.numpy()
print('RMSE:', mean_squared_error(y_test_np, y_pred, squared=False))
print('R2 Score:', r2_score(y_test_np, y_pred))


# ### **Linear Regression**

# In[11]:


from sklearn.linear_model import LinearRegression

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print('RMSE:', mean_squared_error(y_test, y_pred, squared=False))
print('R2 Score:', r2_score(y_test, y_pred))



# ### **Ensemble Method**

# In[13]:


from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Define models
model1 = LinearRegression()
model2 = RandomForestRegressor(n_estimators=100, random_state=42)
model3 = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Create ensemble
ensemble = VotingRegressor(estimators=[('lr', model1), ('rf', model2), ('xgb', model3)])
ensemble.fit(X_train, y_train)

# Evaluate
y_pred = ensemble.predict(X_test)
print('RMSE:', mean_squared_error(y_test, y_pred, squared=False))
print('R2 Score:', r2_score(y_test, y_pred))


# The best implementation are Random Forest and XGboost
