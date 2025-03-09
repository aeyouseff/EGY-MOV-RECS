import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Function to clean and normalize Arabic text
def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove Arabic diacritics (harakat)
    text = re.sub(r'[\u064B-\u065F]', '', text)

    # Standardize similar Arabic letters
    replacements = {
        "أ": "ا", "إ": "ا", "آ": "ا",  # Normalize Alef
        "ى": "ي", "يْ": "ي",           # Normalize Ya
        "ة": "ه",                      # Convert Ta Marbuta to Ha
        "ؤ": "و", "ئ": "ي",            # Normalize Hamza
        "ء": ""                        # Remove standalone Hamza
    }

    # Replace characters
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# Load and preprocess movie dataset
def load_data():
    movies_dataset = pd.read_csv("EgyptionMoviesDataset.csv")

    # Select important features
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    for feature in selected_features:
        movies_dataset[feature] = movies_dataset[feature].fillna('')

    # Clean movie titles
    movies_dataset['clean_title'] = movies_dataset['title'].apply(clean_text)

    # Combine features into a single text representation
    combined_features = movies_dataset[selected_features].agg(' '.join, axis=1)

    # Define Arabic stop words
    arabic_stop_words = [
        "و", "في", "من", "إلى", "على", "عن", "أن", "هذه", "ذلك", "الذي", "التي",
        "ب", "كان", "كون", "تكون", "إلى", "أن", "لكن", "ثم", "أن"
    ]

    # Convert text data into numerical representation using TF-IDF
    vectorizer = TfidfVectorizer(
        min_df=1,
        stop_words=arabic_stop_words,
        lowercase=True,
        ngram_range=(1, 2),
        max_features=1000,
        sublinear_tf=True
    )

    feature_matrix = vectorizer.fit_transform(combined_features)

    # Compute similarity between movies
    similarity_matrix = cosine_similarity(feature_matrix)

    return movies_dataset, similarity_matrix

# Load and preprocess data
movies_dataset, similarity_matrix = load_data()
movie_titles_dict = dict(zip(movies_dataset['clean_title'], movies_dataset['title']))  # Store original titles
movie_titles_list = list(movie_titles_dict.keys())

# Function to get movie recommendations
def recommend_movie(movie_name):
    movie_name = clean_text(movie_name)  # Clean user input

    if movie_name not in movie_titles_list:
        return ["❌ الفيلم ده مش موجود هنا لسه، جرب فيلم تاني."]

    movie_index = movie_titles_list.index(movie_name)
    similarity_scores = similarity_matrix[movie_index]
    similar_movies_indices = np.argsort(similarity_scores)[::-1][1:4]
    recommended_movies = [movie_titles_dict[movie_titles_list[i]] for i in similar_movies_indices]

    return recommended_movies

# Test the recommendation
recommended_movies = recommend_movie('الفيل الأزرق')
print(recommended_movies)
