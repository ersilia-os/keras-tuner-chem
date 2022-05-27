# python -m pip install PyTDC

import sys
import pandas as pd

from tdc.single_pred import Tox
data = Tox(name = 'LD50_Zhu')
split = data.get_split()

for k,v in split.items():
	v.to_csv(k+".csv", index=False)
