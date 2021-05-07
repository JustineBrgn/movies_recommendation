
import numpy as np
import pandas as pd
# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel
#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
#TfidfVectorizer bc if the movie has a lot of votes it's better than not a lot

from sklearn.metrics.pairwise import cosine_similarity

# Load Movies Metadata
metadata = pd.read_csv("movies_database\movies_metadata.csv", low_memory=False)
# otherwise it's too heavy
meta = metadata[:10000].copy()

# Load keywords and credits
credits = pd.read_csv("movies_database\credits.csv")
keywords = pd.read_csv("movies_database\keywords.csv")


###CONTENT-BASED RECOMMENDERS


def content_recomm1(title):
    #Replace NaN with an empty string
    meta['overview'] = meta['overview'].fillna('')

    #Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')
    #Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(meta['overview'])

    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    #Construct a reverse map of indices and movie titles
    indices = pd.Series(meta.index, index=meta['title']).drop_duplicates()


    # Function that takes in movie title as input and outputs most similar movies
    def get_recommendations(title, cosine_sim=cosine_sim):
        # Get the index of the movie that matches the title
        idx = indices[title]
        
        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[1:11]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Return the top 10 most similar movies
        return meta['title'].iloc[movie_indices]
        
    
    return(get_recommendations(title))
        



### Credits, Genres, and Keywords Based Recommender

def content_recomm2(title):
    meta = metadata[:10000].copy()


    # Convert IDs to int. Required for merging
    keywords['id'] = keywords['id'].astype('int')
    credits['id'] = credits['id'].astype('int')
    meta['id'] = meta['id'].astype('int')

    # Merge keywords and credits into your main metadata dataframe
    meta = meta.merge(credits, on='id')
    meta = meta.merge(keywords, on='id')

    # Parse the stringified features into their corresponding python objects
    from ast import literal_eval

    features = ['cast', 'crew', 'keywords', 'genres']
    for feature in features:
        meta[feature] = meta[feature].apply(literal_eval)
        

    #Get the director's name from the crew feature. If the director is not listed, return NaN
    def get_director(x):
        for i in x:
            if i['job'] == 'Director':
                return i['name']
        return np.nan
    

    #return the top 3 elements or the entire list, whichever is more. Here the list refers to the cast, keywords, and genres.
    def get_list(x):
        if isinstance(x, list):
            names = [i['name'] for i in x]
            #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
            if len(names) > 3:
                names = names[:3]
            return names

        #Return empty list in case of missing/malformed data
        return []


    # Define new director, cast, genres and keywords features that are in a suitable form.
    meta['director'] = meta['crew'].apply(get_director)

    features = ['cast', 'keywords', 'genres']
    for feature in features:
        meta[feature] = meta[feature].apply(get_list)
    

    # Function to convert all strings to lower case and strip names of spaces
    def clean_data(x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        else:
            #Check if director exists. If not, return empty string
            if isinstance(x, str):
                return str.lower(x.replace(" ", ""))
            else:
                return ''
                

    # Apply clean_data function to your features.
    features = ['cast', 'keywords', 'director', 'genres']

    for feature in features:
        meta[feature] = meta[feature].apply(clean_data)
        
    def create_soup(x):
        return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
        
    # Create a new soup feature
    meta['soup'] = meta.apply(create_soup, axis=1)


    # Import CountVectorizer and create the count matrix
    from sklearn.feature_extraction.text import CountVectorizer 
    #CountVectorizer bc no need to down-weight the actor/director's presence if he or she has acted or directed in relatively more movies. It doesn't make much intuitive sense to down-weight them in this context.

    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(meta['soup'])
    count_matrix.shape

    # Compute the Cosine Similarity matrix based on the count_matrix # use the cosine_similarity to measure the distance between the embeddings.
    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

    # Reset index of your main DataFrame and construct reverse mapping as before
    meta = meta.reset_index()
    indices = pd.Series(meta.index, index=meta['title'])


    def get_recommendations(title, cosine_sim=cosine_sim2):
        # Get the index of the movie that matches the title
        idx = indices[title]

        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[1:11]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Return the top 10 most similar movies
        
        final_list=metadata['title'].iloc[movie_indices]
        final_list=final_list.tolist()

        return(final_list)

    return(get_recommendations(title))


    