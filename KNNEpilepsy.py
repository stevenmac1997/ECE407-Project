#Steven Mac
#ECE 407
#Seizure Detection 
#Methods: KNN
import pandas as pd


def KNN (N,data_train,data_test,loc1,loc2,train_tags,test_tags):
    #Confusiong Matrix = 1,1   1,0   1,-1
    #                    0,1   0,0   0,-1
    #                   -1,1  -1,0  -1,-1
    confusion_matrix=[[0,0,0],[0,0,0],[0,0,0]]
    confusion_matrix=pd.DataFrame(confusion_matrix)
    cm_test=test_tags # holds test_tags for confusion matrix calculations
    iterations = len(cm_test)
    target=0 #Comparing this data cell from test set to training set
    x=0
    i=0
    while x <= N:
        #Parses through columns
        row_index=0
        predicted_results=[] # Holds all the predicted results by KNN
        print("\n----------------------------------------------------Processing KNN #",x,"----------------------------------------------------\n")
        while row_index < iterations:
            # We find the euclidean distance
            # Compares row by row N columns of cells against training data
            # i.e cell [0][0] from test data is compared to the entire column
            #   of the training data
            target = data_test[x][row_index]
            temp_df = data_train[x]-target
            temp_df = temp_df.abs()
            
            cm_train=[] #Reset training   
            i=0         #Reset Iteration
            while i < N:
                #Finding the KNNs
                #1) Find 1st location of shortest dist
                #2) Replace with 100 so we don't find again
                #3) Repeat N times
                shortest_dist_loc=temp_df.idxmin()
                cm_train.append(train_tags[shortest_dist_loc]) 
                temp_df[shortest_dist_loc]=100
                i+=1
            
            #Select the most occuring value out of the 5 in cm_train
            #because that will be the nearest neighbor
            j=0
            count=[0,0,0] #Count = [# of epilepsy, # of normals, # of nothings]
            while j < N:
                if cm_train[j] == 1:
                    count[0]+=1
                elif cm_train[j] == 0:
                    count[1]+=1
                else:
                    count[2]+=1
                j+=1
            
            find_max = count.index(max(count)) #
            if find_max == 0:
                predicted_results.append(1)
            elif find_max==1:
                predicted_results.append(0)
            elif find_max==2:
                predicted_results.append(-1)
            else:
                print("Error")
        
            row_index+=1
        x+=1

    #Time to compare predicted with actual to make confusion table
    k=0
    while k < iterations:
        if predicted_results[k] == 1 and cm_test[k] == 1:
            confusion_matrix[0][0]+=1
        elif predicted_results[k] == 1 and cm_test[k] == 0:
            confusion_matrix[1][0]+=1
        elif predicted_results[k] == 1 and cm_test[k] == -1:
            confusion_matrix[2][0]+=1

        elif predicted_results[k] == 0 and cm_test[k] == 1:
            confusion_matrix[0][1]+=1
        elif predicted_results[k] == 0 and cm_test[k] == 0:
            confusion_matrix[1][1]+=1
        elif predicted_results[k] == 0 and cm_test[k] == -1:
            confusion_matrix[2][1]+=1

        elif predicted_results[k] == -1 and cm_test[k] == 1:
            confusion_matrix[0][2]+=1
        elif predicted_results[k] == -1 and cm_test[k] == 0:
            confusion_matrix[1][2]+=1
        elif predicted_results[k] == -1 and cm_test[k] == -1:
            confusion_matrix[2][2]+=1
        k+=1
        
    accuracy = (confusion_matrix[0][0]+confusion_matrix[1][1]+confusion_matrix[2][2])/iterations
    print("Accuracy of {}-Nearest Neighbors: {}".format(N,accuracy))
    print("Confusion Matrix\n",confusion_matrix)

    return

def main():

    #CHANGE K TO AFFECT # OF NEIGHBORS TO LOOK AT FOR KNN
    K=100 #!!!!!IMPORTANT!!!!! DETERMINES K FOR KNN !!!!!!!!




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
 
    KNN(K,df,df2,assignment_locations,assignment_locations2,df_assignments,df2_assignments)
        
    return

if __name__=="__main__":
    main()
