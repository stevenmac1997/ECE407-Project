# Program to extract number of 
# columns in Python 
import xlrd
import numpy as np
from operator import itemgetter
from math import sqrt

def distance(x1,x2,y1,y2):
    dist=sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))
    dist=float(dist)
    return dist

def KNN(loc,K,tar):
    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0) 
    sheet.cell_value(0, 0) 
    xpred=float(tar[0])
    ypred=float(tar[1])
    Arr=[]

    for i in range(sheet.nrows):
        n=0
        if i==0:
            print("Headers")
        else:
            while n<=3:
                n=n+1
                xdata=float(sheet.cell_value(i,1))
                ydata=float(sheet.cell_value(i,2))
                L=float(sheet.cell_value(i,3))
                result=[distance(xpred,xdata,ypred,ydata)]
                result.append(L)
                Arr.append(result)
    Arr=Arr.sort(key=itemgetter(0),reverse=False)
    #Arr=Arr[np.argsort(Arr[:,0])]
    print("---------------------------------------------------------------\n \n \n")
    print (Arr)
    return

def main():
    target=[6.2,4.9] #vector transposed
    k=5
    location ="q1.xlsx"
    KNN(location,k,target)


if __name__=="__main__":
    main()