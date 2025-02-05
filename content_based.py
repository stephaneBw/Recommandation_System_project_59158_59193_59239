import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import matplotlib.pyplot as plt

# Load movie names and ratings data
movie_names = pd.read_csv("movies.csv")
ratings_data = pd.read_csv("ratings.csv")

# Display the first few rows of each dataset for confirmation
print(movie_names.head())
print(ratings_data.head())

# Check necessary columns in movie_names
if 'genres' not in movie_names.columns:
    print("The movie data must contain a 'genres' column for recommendations.")

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Fill NaN with empty strings and fit transform the genres
movie_names['genres'] = movie_names['genres'].fillna('')
tfidf_matrix = tfidf.fit_transform(movie_names['genres'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Reset movie names index to enable easier lookups
movie_names = movie_names.reset_index()
indices = pd.Series(movie_names.index, index=movie_names['title'])

# Function to get recommendations based on the content of a movie
# Function to get recommendations based on the content of a movie
def get_recommendations(title, cosine_similarity=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices.get(title)

    if idx is None:
        return f"Movie '{title}' not found in the database."

        # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_similarity[idx]))

    # Sort the movies based on the scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return movie_names['title'].iloc[movie_indices]


# Example usage of the recommendation function
input_movie = "Toy Story (1995)"  # Change this to test with different movies
recommended_movies = get_recommendations(input_movie)
print(f"Recommended movies for '{input_movie}':")
print(recommended_movies)

# Fixing the filtering and visualization of recommendations
if isinstance(recommended_movies, str):
    print(recommended_movies)  # Print error message if movie is not found
else:
    # Get movieIds of recommended movies
    recommended_movies_data = movie_names[movie_names['title'].isin(recommended_movies)]
    recommended_movie_ids = recommended_movies_data['movieId'].tolist()

    # Filter ratings data for recommended movie ids
    recommended_ratings = ratings_data[ratings_data['movieId'].isin(recommended_movie_ids)]

    # Count ratings for recommended movies
    rating_counts = recommended_ratings['movieId'].value_counts()

    # Plot the count of ratings for the recommended movies
    plt.figure(figsize=(10, 4))
    plt.bar(rating_counts.index.astype(str), rating_counts.values, color='lightcoral')
    plt.title('Number of Ratings for Recommended Movies')
    plt.xlabel('Movie ID')
    plt.xticks(rotation=90)
    plt.ylabel('Number of Ratings')
    plt.show()