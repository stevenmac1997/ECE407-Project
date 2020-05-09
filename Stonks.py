#Steven Mac
#ECE 407
#Seizure Detection 
#Methods: KNN
import numpy as np
import pandas as pd

def KNN (N,data_train,data_test,loc1,loc2,train_tags,test_tags):
    #Confusiong Matrix = 1,1   1,0   1,-1
    #                    0,1   0,0   0,-1
    #                   -1,1  -1,0  -1,-1
    confusion_matrix=[(0,0,0),(0,0,0),(0,0,0)]
    target=0 #Comparing this data cell from test set to training set
    x=0
    row_index=0
    while x <= N:
        
        while row_index < 60:
            # We find the euclidean distance
            # Compares row by row N columns of cells against training data
            # i.e cell [0][0] from test data is compared to the entire column
            #   of the training data
            target = data_test[row_index][x]
            temp_df = data_train[x]-target
            temp_df = temp_df.abs()
            
            shortest_dist=temp_df.idxmin()
            #print (shortest_dist)
            cm_test=test_tags[row_index]
            row_index+=1

            #print(temp_df)
        
        x+=1

    return

def main():

#--------------Reading in Training Data---------------

    df = pd.read_csv('Train.csv', header=None)
    df = df.replace('Epilepsy',1) # "Epilepsys" will be denoted by 1
    df = df.replace('Normal',0)   # "Normal" will be denoted by 0
    df = df.replace('Nothing',-1) # "Nothing" will be denoted by -1
    df_assignments = df.pop(704)
    assignment_locations=[]
    temp_epi=0
    temp_norm=0
    temp_no=0
    #print(df_assignments)
    #print(df)
    for x in df_assignments:
        if x == 1:
            temp_epi+=1
            temp_norm+=1
            temp_no+=1
        elif x == 0:
            temp_norm+=1
            temp_no+=1
        elif x == -1:
            temp_no+=1
    assignment_locations.append(temp_epi-1)
    assignment_locations.append(temp_norm-1)
    assignment_locations.append(temp_no-1)
    #print(assignment_locations)


#-------------Reading in Test Data------------------

    df2 = pd.read_csv('test.csv', header=None)
    df2 = df2.replace('Epilepsy',1) # "Epilepsys" will be denoted by 1
    df2 = df2.replace('Normal',0)   # "Normal" will be denoted by 0
    df2 = df2.replace('Nothing',-1) # "Nothing" will be denoted by -1
    df2_assignments = df2.pop(704)
    assignment_locations2=[]
    temp_epi=0
    temp_norm=0
    temp_no=0
    #print(df2_assignments)
    #print(df)
    for x in df2_assignments:
        if x == 1:
            temp_epi+=1
            temp_norm+=1
            temp_no+=1
        elif x == 0:
            temp_norm+=1
            temp_no+=1
        elif x == -1:
            temp_no+=1
    assignment_locations2.append(temp_epi-1)
    assignment_locations2.append(temp_norm-1)
    assignment_locations2.append(temp_no-1)
    #print(assignment_locations2)
 
    n=5 #KNN number
    KNN(n,df,df2,assignment_locations,assignment_locations2,df_assignments,df2_assignments)
        
    return

if __name__=="__main__":
    main()