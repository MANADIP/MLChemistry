import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
import matplotlib.pyplot as plt

Data = pd.read_csv('dataset_train.csv')
smiles = Data.iloc[:, 0]
mols = [Chem.MolFromSmiles(smile) for smile in smiles]
Descrip = [Descriptors.CalcMolDescriptors(mol) for mol in mols]
df_data = pd.DataFrame(Descrip)
x = df_data.values
y = Data.iloc[:, 2]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=40)
# Train the model
model = HistGradientBoostingRegressor(random_state=0)
model.fit(x_train, y_train)
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

plt.subplot(121)
plt.scatter(y_train, y_train_pred)
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='-')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (Training Set)')

# Plot actual vs predicted values for test set
plt.subplot(122)
plt.scatter(y_test, y_test_pred)
plt.plot([min(y_test) , max(y_test)] , [min(y_test) , max(y_test)] , color = 'red', linestyle='-')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (Test Set)')

plt.tight_layout()
plt.show()
