# Import Pandas
import pandas as pd

# Load Movies Metadata
metadata = pd.read_csv("movies_database/movies_metadata.csv", low_memory=False)


###SIMPLE RECOMMENDERS


def simple_recomm():
    # Calculate mean of vote average column
    C = metadata['vote_average'].mean()

    # Calculate the minimum number of votes required to be in the chart
    m = metadata['vote_count'].quantile(0.90)

    # Filter out all qualified movies into a new DataFrame
    q_movies = metadata.copy().loc[metadata['vote_count'] >= m]

    # Function that computes the weighted rating of each movie
    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']
        # Calculation based on the IMDB formula
        return (v/(v+m) * R) + (m/(m+v) * C)
        
    # Define a new feature 'score' and calculate its value with `weighted_rating()`
    q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

    #Sort movies based on score calculated above
    q_movies = q_movies.sort_values('score', ascending=False)
    q_movies=q_movies[['title','overview','imdb_id']].head(10)
    q_movies_title=q_movies['title'].values.tolist()
    q_movies_overview=q_movies['overview'].values.tolist()
    q_movies_imdb=q_movies['imdb_id'].values.tolist()
    final_movies=[]
    final_overview=[]
    final_imdb=[]
    for m in range(len(q_movies_title)):
        final_movies.append(q_movies_title[m])
        final_overview.append(q_movies_overview[m])
        final_imdb.append(q_movies_imdb[m].split('t')[-1])

    #Print the top 15 movies
    return(final_movies, final_overview, final_imdb)