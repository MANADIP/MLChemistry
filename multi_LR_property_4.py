import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
Data = pd.read_csv('dataset_train.csv')
smiles = Data.iloc[:, 0]
mols = [Chem.MolFromSmiles(smile) for smile in smiles]
Descrip = [Descriptors.CalcMolDescriptors(mol) for mol in mols]
df_data = pd.DataFrame(Descrip)
x = df_data.values
y = Data.iloc[:, 4]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40)

class MyRegression:
    def __init__(self, alpha=1.0):
        self.coef_ = None
        self.intercept_ = None
        self.alpha = alpha

    def fit(self, x_train, y_train):
        ridge = Ridge(alpha=self.alpha)
        ridge.fit(x_train, y_train)
        self.coef_ = ridge.coef_
        self.intercept_ = ridge.intercept_

    def predict(self, x_test):
        y_pred = np.dot(x_test, self.coef_) + self.intercept_
        return y_pred

lr = MyRegression(alpha=0.1)
lr.fit(x_train, y_train)
y_train_pred = lr.predict(x_train)
y_test_pred = lr.predict(x_test)

# actual vs predicted for training set
plt.subplot(121)
plt.scatter(y_train, y_train_pred)
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='-')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (Training Set)')

#actual vs predicted for test set
plt.subplot(122)
plt.scatter(y_test, y_test_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='-')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (Test Set)')
plt.tight_layout()
plt.show()

