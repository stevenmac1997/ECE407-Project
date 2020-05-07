import numpy as np
import pandas as pd

data = pd.read_excel('data2.xlsx',header=0)

x=data['X']
y=data['Y']

covDat = np.array([x,y])

covMatrix = np.cov(covDat,bias=False)
print (covMatrix)