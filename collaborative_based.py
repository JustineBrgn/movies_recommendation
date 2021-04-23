
import math
import pandas as pd
import numpy as np

# Load Ratings
metadata = pd.read_csv("movies_metadata.csv", low_memory=False)
ratings = pd.read_csv("ratings.csv", low_memory=False)
ratings_small = pd.read_csv("ratings_small.csv", low_memory=False)

#Create de matrix(nb of user, nb of movies) with ratings
#nb of users
nb_user=ratings_small['userId'].nunique()
#list of all the movie Id
movie=ratings_small['movieId'].unique()
#create the matrix with zeros
table=np.zeros((nb_user+1,len(movie)))

#go through the indexes of the dataframe rating_small and fill in the table with the rating of each movie
for i in ratings_small.index:
    II=np.where(movie == ratings_small['movieId'][i])
    table[ratings_small['userId'][i]][II]=ratings_small['rating'][i]
    
    

def pearson_similarity2(Id_person1,Id_person2, table):
    """returns the pearson similarity between person1 and person2
    Id_person1 : int
    Id_person2 : int
    table: array(nb of user,nb of movies)"""
    
    rating_common_1=[] #list to put the 1st person ratings for all the movies in common
    rating_common_2=[] 
    
    #going through all the movies; if person1 and person2 have rated the movie, we add the rating to the corresponding list
    for k in range(len(table[0])):
        if table[Id_person1][k]!=0 and table[Id_person2][k]!=0:
            rating_common_1.append(table[Id_person1][k])
            rating_common_2.append(table[Id_person2][k])
    
    #calculating the pearson similarity:
    
    rating_common_1=np.array(rating_common_1)
    rating_common_2=np.array(rating_common_2)
    
    n=len(rating_common_1)
    
    s1=sum(rating_common_1)
    s2=sum(rating_common_2)
    
    ss1=sum(np.power(rating_common_1,2))
    ss2=sum(np.power(rating_common_2,2))
    ps=sum(rating_common_1*rating_common_2)
    
    num=n*ps-(s1*s2)
    
    den=math.sqrt((n*ss1-pow(s1,2))*(n*ss2-pow(s2,2)))
    
    return (num/den) if den!=0 else 0
    
    
def collaborative_recomm(person_Id,bound, table=table):
    """returns the list of the movies recommanded for person_Id
    person_Id:int
    bound: int
    table: array(nb of users,nb of movies)"""
    
    #the line of table concerning person_Id
    data_person=table[person_Id]
    
    #list of the pearson_similarity with all the other ppl of the table
    scores=[]
    for i in range(nb_user):
        scores.append((pearson_similarity2(person_Id,i, table),i))
    scores.sort()
    scores.reverse()
    #the first one is itself
    scores=scores[1:bound+1]
    
    recomms={}
    
    for (sim,other) in scores:
        #the line of the table concerning the other Id_person 
        ranked=table[other]
        for itm in range(len(ranked)):
            #if other person has ranked the movie and if the person_Id (the one we're interested in) has NOT
            if ranked[itm]!=0 and data_person[itm]==0:
                weight=sim*ranked[itm]
                #if the movie was already in recomms (so it was already ranked by other ppl)
                if itm in recomms:
                    s, weights=recomms[itm]
                    recomms[itm]=(s+sim,weights+[weight])
                #otherwise we add it
                else:
                    recomms[itm]=(sim,[weight])
    
    #calculates the mean if there are several values for 1 movie
    for r in recomms:
        sim,item =recomms[r]
        recomms[r]=sum(item)/sim
    
    #sorts recomms from the most recommanded movies to the least
    sorted_recomms = sorted(recomms.items(), key=lambda x: x[1], reverse=True) 
    sorted_recomms=sorted_recomms[0:bound]

    #list of final results
    movies_indices=[]
    for (movie,sim) in sorted_recomms:
        movies_indices.append(movie)


    return metadata['title'].iloc[movies_indices]
    
#print(recommand(4,10))
    
