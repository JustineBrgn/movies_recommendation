import math
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from collaborative_based import *



#transform the table into a database
table_df = pd.DataFrame(data=table)

#we take 20% of the data from table for testing
data_train = table_df.sample(frac=0.8, random_state=2)
data_test=table_df.drop(data_train.index)
data_test.shape




#First approach (evaluate on 20% of users of the dataset)



#y_test to put the ratings of the movies rated and recommended
y_test=[]
#y_recom to put the predicted rate for movie rated
y_recom=[]


for w in range(10):
    
    #print(w,' / 10')

    #we take 20% of the data from table for testing
    data_train = table_df.sample(frac=0.8, random_state=w)
    data_test=table_df.drop(data_train.index)

    INDEX=data_test.index

    for i in INDEX:#i: id of the user
        #we recommand movies to the user i
        recom=collaborative_recomm(i,10,table)

        for j in range(9066): #j:nb of the movie 
            # we go through the data_test

            if data_test[j][i]!=0: 
                #the movie was rated

                for k in range(len(recom)):
                    #we go through all the movies that were recommanded 

                    if recom[k][0]==i:
                        #the movie in the recommanded list was rated by the user i

                        #we add the rate the user i gave to the movie to y_test (the "real" result)
                        y_test.append(data_test[j][i])
                        #we add the recommanded value to y_recom (the "predicted" result)
                        y_recom.append(recom[k][1])
    
    w+=1
                    
                    
print(np.sqrt(mean_squared_error(y_test, y_recom))) 


y_test = pd.Series(y_test) 
y_recom = pd.Series(y_recom)
sns.distplot(y_test-y_recom)



#Second approach (evaluate on 20% of users of the dataset + remove 10% of the ratings)

from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

#y_test to put the ratings of the movies rated and recommended
y_test2=[]
#y_recom to put the predicted rate for movie rated
y_recom2=[]

for w in range(10): 
    
    print(w,' / 10')

    #we take 20% of the data from table for testing
    data_train = table_df.sample(frac=0.8, random_state=w)
    data_test=table_df.drop(data_train.index)

    INDEX=data_test.index
    
    #array avec 90% des films rated
    data_test90=data_test.copy()
    #array avec 10% des films rated (evaluation)
    data_test10=data_test.copy()

    for i in range(len(data_test)):
        index_movies=[]
        for k in INDEX:
            if data_test[i][k]!=0:
                index_movies.append(k)
        np.random.shuffle(index_movies)

        A=int(0.9*len(index_movies))
        index_movies90=index_movies[:A]
        index_movies10=index_movies[A:]

        for w in index_movies90 :
            data_test10[i][w]=0
        for j in index_movies10:
            data_test90[i][w]=0
            
    #Merging the data_train (not modified) and data_test90
    DATA=[data_train, data_test90]
    UseThisOneForTesting = pd.concat(DATA)
    UseThisOneForTesting.sort_index()        
    

    
    for i in INDEX:#i: id of the user
        #we recommand movies to the user i
        recom=collaborative_recomm(i,10,UseThisOneForTesting)

        for j in range(9066): #j:nb of the movie 
            # we go through the UseThisOneForTesting

            if data_test10[j][i]!=0 and math.isnan(data_test10[j][i])==False: 
                #the movie was rated and part of the 10% we removed

                for k in range(len(recom)):
                    #we go through all the movies that were recommanded 

                    if recom[k][0]==i:
                        #the movie in the recommanded list was part of the 10% removed

                        #we add the rate the user i gave to the movie to y_test (the "real" result)
                        y_test2.append(data_test10[j][i])
                        #we add the recommanded value to y_recom (the "predicted" result)
                        y_recom2.append(recom[k][1])
    
    w+=1
                    
                    
print(np.sqrt(mean_squared_error(y_test2, y_recom2))) 


y_test2 = pd.Series(y_test2) 
y_recom2 = pd.Series(y_recom2)
sns.distplot(y_test2-y_recom2)