import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
Data=pd.read_csv('dataset_train.csv')
smiles=Data.iloc[ :  , 0]
mols = [Chem.MolFromSmiles(smile) for smile in smiles]
Descrip=[Descriptors.CalcMolDescriptors(mol) for mol in mols]
df_data=pd.DataFrame(Descrip)
#print(df_data.head())

x=df_data.iloc[:,:]
y = Data.iloc[:,4]
x=x.values
y=y.values
#print(x)
#print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=40)
#print(x_train)
#print("*******")
#print(y_train)
import torch
import torch.nn as nn
import torch.nn.functional as F

#Model describe
class Model(nn.Module):
    def __init__(self,in_feature=210,H1=210,out_feature=1):
        super().__init__() # Initiation on nn model 
        self.fc1=nn.Linear(in_feature,H1)
       # self.fc2=nn.Linear(H1,H2)
        self.out=nn.Linear(H1,out_feature)
    def forward(self,x):
       x=F.relu(self.fc1(x))
     #  x=F.relu(self.fc2(x))
       x=self.out(x)
       return x

#Manual seed for randmisiation
torch.manual_seed(40)
model = Model()
#load daata 
import matplotlib.pyplot as plt
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.FloatTensor(y_train).view(-1, 1)
y_test = torch.FloatTensor(y_test).view(-1, 1)

criterion = nn.MSELoss()

import matplotlib.pyplot as plt
from torch.optim import Adam
#learning_rate=1e-2
learning_rate=0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#trainThe dataset
epochs = 3100
losses = []
losses1=[]
for i in range (epochs):
    y_pred = model.forward(x_train)
    loss = criterion(y_pred, y_train)                 # predicted values vs y_train
    loss1 = criterion(model.forward(x_test), y_test)  # predicted values vs y_test
    losses.append(loss.item())
    losses1.append(loss1.item())

    if i%100==0:
        print(f'Epoch {i}, Loss_train{loss.item()} , Loss_test{loss1.item()}')

    #Back Propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
plt.plot(range(epochs) , losses, label='y_train')
plt.plot(range(epochs) , losses1 , label='y_test')
plt.ylabel("Error")
plt.xlabel("Epochs")
plt.legend()
plt.show()
#predictions using model 
with torch.no_grad():
    y_pred_train = model(x_train)
    y_pred_test = model(x_test)

#tensors to numpy arrays
y_train_np = y_train.numpy().flatten()
y_pred_train_np = y_pred_train.numpy().flatten()

y_test_np = y_test.numpy().flatten()
y_pred_test_np = y_pred_test.numpy().flatten()

# actual vs predicted values for training set
plt.subplot(121)
plt.scatter(y_train_np, y_pred_train_np)

plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='-')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (Training)')

# Plot actual vs predicted values for test set
plt.subplot(122)
plt.scatter(y_test_np, y_pred_test_np)

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='-')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (Test)')

plt.tight_layout()
plt.show()

