import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load datasets
movie_names = pd.read_csv("movies.csv")
ratings_data = pd.read_csv("ratings.csv")


# Define the popularity-based recommendation function
def popularity_recommendations():
    # Merge ratings data with movie names
    movie_data = pd.merge(ratings_data, movie_names, on='movieId')

    # Calculate mean ratings and sorting for top-rated movies
    ratings_mean_count = movie_data.groupby('title')['rating'].agg(['mean', 'count']).rename(
        columns={'mean': 'rating', 'count': 'rating_counts'})
    ratings_mean_count['rating'] = ratings_mean_count['rating'].round(1)

    # Filter the DataFrame for movies with a rating greater than 3 and more than 100 counts
    filtered_ratings = ratings_mean_count[
        (ratings_mean_count['rating'] > 3) & (ratings_mean_count['rating_counts'] > 100)]

    # Sort the filtered DataFrame by ratings and get the top 10 movies
    top_movies = filtered_ratings.sort_values(by='rating', ascending=False).head(10)

    print("Top 10 Popular Movies:")
    print(top_movies)

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.barh(top_movies.index, top_movies['rating_counts'], color='lightblue')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Movies')
    plt.title('Top 10 Popular Movies Based on Ratings')
    plt.show()


# Define the content-based recommendation function
def content_based_recommendations():
    # Check necessary columns in movie_names
    if 'genres' not in movie_names.columns:
        print("The movie data must contain a 'genres' column for recommendations.")
        return

        # Initialize TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')

    # Fill NaN with empty strings and fit transform the genres
    movie_names['genres'] = movie_names['genres'].fillna('')
    tfidf_matrix = tfidf.fit_transform(movie_names['genres'])

    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Reset movie names index to enable easier lookups
    movie_names.reset_index(inplace=True)
    indices = pd.Series(movie_names.index, index=movie_names['title'])

    # Function to get recommendations based on the content of a movie
    def get_recommendations(title):
        idx = indices.get(title)

        if idx is None:
            return f"Movie '{title}' not found in the database."

        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]  # Top 10 similar movies
        movie_indices = [i[0] for i in sim_scores]

        return movie_names['title'].iloc[movie_indices]

        # Get user input for movie search

    input_movie = input("Enter a movie title for recommendations: ")
    recommended_movies = get_recommendations(input_movie)

    if isinstance(recommended_movies, str):
        print(recommended_movies)  # Print error message if movie is not found
    else:
        print(f"Recommended movies for '{input_movie}':")
        print(recommended_movies)

        # Get movieIds of recommended movies
        recommended_movies_data = movie_names[movie_names['title'].isin(recommended_movies)]
        recommended_movie_ids = recommended_movies_data['movieId'].tolist()

        # Filter ratings data for recommended movie ids
        recommended_ratings = ratings_data[ratings_data['movieId'].isin(recommended_movie_ids)]

        # Count ratings for recommended movies
        rating_counts = recommended_ratings['movieId'].value_counts()

        # Plotting the count of ratings for the recommended movies
        plt.figure(figsize=(10, 4))
        plt.bar(rating_counts.index.astype(str), rating_counts.values, color='lightcoral')
        plt.title('Number of Ratings for Recommended Movies')
        plt.xlabel('Movie ID')
        plt.xticks(rotation=90)
        plt.ylabel('Number of Ratings')
        plt.show()

    # Main function to run the application


def main():
    print("Welcome to the Movie Recommendation System!")
    print("Please choose a recommendation type:")
    print("1. Popularity-Based Recommendations")
    print("2. Content-Based Recommendations")

    choice = input("Enter 1 or 2: ")

    if choice == '1':
        popularity_recommendations()
    elif choice == '2':
        content_based_recommendations()
    else:
        print("Invalid choice. Please restart the application and choose 1 or 2.")


if __name__ == "__main__":
    main()