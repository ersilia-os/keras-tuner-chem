import pandas as pd
from rdkit import Chem

from src.trainer import train_model

df = pd.read_csv("data/train.csv")

smiles_ = list(df["Drug"])
y_ = list(df["Y"])

smiles = []
y = []
for i, smi in enumerate(smiles_):
	mol = Chem.MolFromSmiles(smi)
	if mol is None:
		continue
	smiles += [smi]
	y += [y_[i]]

mdl = train_model(smiles, y)
print(mdl)