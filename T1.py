import pandas as pd
from flask import Flask, request, render_template
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Load the data
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Preprocess data
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
user_item_matrix.fillna(0, inplace=True)

# Compute cosine similarity between items (movies)
item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# Function to get top-N similar movies based on a given movie name
def get_top_n_similar_movies(movie_name, n=10):
    # Find movies with similar names (case insensitive)
    similar_movies = movies[movies['title'].str.contains(movie_name, case=False)]
    
    # Check if there are any matches
    if similar_movies.empty:
        return None, None, f"Movie '{movie_name}' not found."
    
    # Take the first matched movie for simplicity
    matched_movie = similar_movies.iloc[0]
    movie_id = matched_movie['movieId']
    
    # Get the similarity scores for the given movie
    similarity_scores = item_similarity_df[movie_id]
    
    # Sort the movies based on similarity scores
    similar_movies = similarity_scores.sort_values(ascending=False).index[1:n+1]
    
    # Get the movie details for the top-N similar movies
    top_movies = movies[movies['movieId'].isin(similar_movies)]
    
    return matched_movie, top_movies, None

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for recommendations based on movie name
@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form.get('movie_name')
    matched_movie, top_n, error_message = get_top_n_similar_movies(movie_name=movie_name, n=10)
    
    if error_message:
        return render_template('index.html', error=error_message)
    
    return render_template('index.html', matched_movie=matched_movie.to_dict(), movies=top_n.to_dict(orient='records'))

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
