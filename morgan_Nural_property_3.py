import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Descriptors
Data=pd.read_csv('dataset_train.csv')
smiles=Data.iloc[ :  , 0]
mols = [Chem.MolFromSmiles(smile) for smile in smiles]
#Descrip=[Descriptors.CalcMolDescriptors(mol) for mol in mols]
#df_data=pd.DataFrame(Descrip)
#print(df_data.head())

#MORGAN FINGER PRINT

from rdkit.Chem import AllChem

# Generate Morgan fingerprints
Descrip = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mols]

df_data = []
for i  in Descrip:
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(i, arr)
    df_data.append(arr)
df_data = np.asarray(df_data)
#df_data=pd.DataFrame(Descrip)
#print(df_data)



x=df_data
y = Data.iloc[:,3]

#print(x)
#print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=50)
#print(x_train)
#print("*******")
#print(y_train)
import torch
import torch.nn as nn
import torch.nn.functional as F

#Model describe
class Model(nn.Module):
    def __init__(self,in_feature=1024,H1=100,out_feature=1):
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
#reshape input feature 
x_train = torch.FloatTensor(x_train).reshape(-1,1024)
x_test = torch.FloatTensor(x_test).reshape(-1,1024)
#convey y_train and y_test to numpy array
y_train=np.array(y_train).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)
#convert numpy aerray to torch
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)

criterion = nn.MSELoss()

import matplotlib.pyplot as plt
from torch.optim import Adam
#learning_rate=1e-2
learning_rate=0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#trainThe dataset
epochs = 1000
losses = []
losses1=[]
for i in range (epochs):
    y_pred = model.forward(x_train)
    loss = criterion(y_pred, y_train)                 # predicted values vs y_train
    loss1 = criterion(model.forward(x_test), y_test)  # predicted values vs y_test
    losses.append(loss.item())
    losses1.append(loss1.item())

    if i%100==0:
       print(f'Epoch {i}, Loss_train {loss.item()} , Loss_test {loss1.item()}')

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

#corr=np.corrcoe(y_train_np , y_pred_train_np)
#print(f'{corr} is the correlation in  actual vs predicted(training) value ')

# actual vs predicted values for training set
plt.subplot(121)
plt.scatter(y_train_np, y_pred_train_np)

plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='-')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (Training)')
#plt.show()
# Plot actual vs predicted values for test set
plt.subplot(122)
plt.scatter(y_test_np, y_pred_test_np)

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='-')
plt.xlabel('Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (Test)')

plt.tight_layout()
plt.show()


