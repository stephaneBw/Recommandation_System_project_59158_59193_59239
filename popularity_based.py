import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load movie names and ratings data
movie_names = pd.read_csv("movies.csv")
ratings_data = pd.read_csv("ratings.csv")

# Display the first few rows of each dataset for confirmation
print(movie_names.head())
print(ratings_data.head())

# Merge ratings data with movie names
movie_data = pd.merge(ratings_data, movie_names, on='movieId')
print(movie_data.head())

# Calculate mean ratings and sorting for top-rated movies
top_rated_movies = movie_data.groupby('title')['rating'].mean().sort_values(ascending=False).head(50)
print(top_rated_movies)

# Count ratings for each movie
rating_counts = movie_data.groupby('title')['rating'].count().sort_values(ascending=False)
print(rating_counts.head())

# Creating a DataFrame for mean ratings and counts
ratings_mean_count = movie_data.groupby('title')['rating'].agg(['mean', 'count']).rename(columns={'mean': 'rating', 'count': 'rating_counts'})

# Round the ratings to one decimal place
ratings_mean_count['rating'] = ratings_mean_count['rating'].round(1)

# Display the DataFrame with mean ratings and counts
print(ratings_mean_count.head())

# Plotting the rounded ratings with count of ratings
plt.figure(figsize=(10, 4))
plt.barh(ratings_mean_count['rating'], ratings_mean_count['rating_counts'], color='lightblue')
plt.xlabel('Number of Ratings')
plt.ylabel('Rounded Average Rating')
plt.title('Movie Ratings Overview')
plt.show()

# Filter the DataFrame for movies with a rating greater than 3 and more than 100 counts
filtered_ratings = ratings_mean_count[(ratings_mean_count['rating'] > 3) & (ratings_mean_count['rating_counts'] > 100)]
print(filtered_ratings)

# Sort the filtered DataFrame by ratings and get the top 10 movies
top_movies = filtered_ratings.sort_values(by='rating', ascending=False).head(10)
print(top_movies)