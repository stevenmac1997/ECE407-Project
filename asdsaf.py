import pandas as pd

lst=[[1,2,3,4,5],[10,20,30,40,50],[100,200,300,400,500]]
pan=pd.DataFrame(lst)
print (pan)
print(pan[3][1])