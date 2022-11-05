from Pfeature.pfeature import *
import numpy as np
import pandas as pd
aac_wp('train.csv', 'a.csv')
df = pd.read_csv('a.csv')
print(df)
