import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Path location for the input datasets
ratings_data_file = 'Ratings.csv'
books_data_file = 'Books.csv'

# Loading the ratings and books datasets
ratings_data = pd.read_csv(ratings_data_file, delimiter=';')
books_data = pd.read_csv(books_data_file, delimiter=';')

# Filtering valid ratings ratings >= 0
filtered_ratings = ratings_data[ratings_data['Rating'] >= 0]

# Mapping ISBNs to integer indices for creating a sparse matrix
mapped_isbns = filtered_ratings['ISBN'].astype('category').cat.codes
rating_scores = filtered_ratings['Rating']

# Creating a sparse matrix for user-item interactions using original User_ID
user_item_matrix = csr_matrix((rating_scores, (filtered_ratings['User-ID'], mapped_isbns)))

# Creating dictionaries to map back to original ISBNs and book titles
isbn_lookup = {index: isbn for index, isbn in enumerate(filtered_ratings['ISBN'].astype('category').cat.categories)}
isbn_to_title = books_data.set_index('ISBN')['Title'].to_dict()

# Number of similar users to consider for recommendations
num_similar_users = 10

# Initializing a list to store recommendations
recommendation_results = []

# Function to compute cosine similarity for a single user
def compute_user_similarities(user_vector, matrix):
    # Calculate sparse cosine similarities for one user against the matrix
    return cosine_similarity(user_vector, matrix, dense_output=False).toarray()[0]

# Looping through each unique User_ID to generate personalized recommendations for the books
unique_user_ids = filtered_ratings['User-ID'].unique()
for user_index, user_id in enumerate(unique_user_ids):
    
    # Getting the current users rating vector
    current_user_vector = user_item_matrix[user_id]
    
    # Calculating cosine similarities between the current user and all other users
    user_similarities = compute_user_similarities(current_user_vector, user_item_matrix)
    
    # Identifing the indices of the top similar users excluding the current user
    similar_user_indices = np.argsort(user_similarities)[-num_similar_users-1:-1][::-1]
    
    # Aggregate ratings from the top similar users
    top_user_ratings = user_item_matrix[similar_user_indices]
    
    # Find books read by similar users
    books_read_by_similar_users = top_user_ratings.sum(axis=0).A1 > 0
    
    # Find books already read by the current user
    books_read_by_current_user = current_user_vector.toarray().flatten() > 0
    
    # Filtering out books already read by the current user
    candidate_books = np.where(books_read_by_similar_users & ~books_read_by_current_user)[0]
    
    # Calculating recommendation scores for each candidate book
    candidate_scores = {}
    for book_idx in candidate_books:
        # Ratings for the candidate book from similar users
        similar_user_book_ratings = top_user_ratings[:, book_idx].toarray().flatten()
        
        # Weights based on similarity scores
        similarity_weights = user_similarities[similar_user_indices]
        
        # Compute the weighted average, avoiding division by zero
        total_weight = similarity_weights.sum()
        if total_weight > 0:
            weighted_rating = np.dot(similarity_weights, similar_user_book_ratings) / total_weight
            candidate_scores[book_idx] = weighted_rating
    
    # Sorting and selecting the top 5 books for recommendation
    top_recommended_books = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Storing the recommendations for the current user
    for book_idx, score in top_recommended_books:
        original_isbn = isbn_lookup[book_idx]
        recommendation_results.append({
            # Directly using the original User_ID
            'User_ID': user_id,
            'Book_ID': original_isbn,
            'Book_Title': isbn_to_title.get(original_isbn, "Unknown Title"),
            'Recommendation_Score': round(score, 2)
        })

# Saving the recommendations to a CSV file
output_recommendation_file = 'Top_5_Recommendations.csv'
recommendations_df = pd.DataFrame(recommendation_results)
recommendations_df.to_csv(output_recommendation_file, index=False)

print(f"Personalized recommendations saved to: {output_recommendation_file}")