# EGY-MOV-RECS - Egyptian Movie Recommendation System

## Try the App Online
🔗 **Live Demo:** [EGY-MOV-RECS App](https://egymovierecs.streamlit.app/)

## Project Description

**EGY-MOV-RECS** is a Python-based movie recommendation system focused on Egyptian cinema. The application uses content-based filtering to recommend Egyptian movies based on a given movie title. It leverages natural language processing techniques, specifically **TF-IDF Vectorization** and **Cosine Similarity**, to compare movie features such as genres, keywords, tagline, cast, and director.

This project is designed to help users discover Egyptian movies similar to their favorite ones, offering personalized movie suggestions based on movie attributes.

## Features

- **Personalized Movie Recommendations:** Provides movie recommendations based on a given movie title.
- **Content-Based Filtering:** Uses movie attributes such as genres, keywords, tagline, cast, and director to generate recommendations.
- **Advanced Text Processing:** Implements **TF-IDF Vectorization** with **bi-grams** and custom Arabic stop words for better accuracy.
- **Arabic Text Normalization:** Standardizes Arabic letters (e.g., `أ` → `ا`, `إ` → `ا`, `ى` → `ي`, `ة` → `ه`) for better matching.
- **Efficient Movie Matching:** Uses **Cosine Similarity** to find the most relevant recommendations.
- **Handles Data Issues:** Fills missing values in the dataset to ensure stability.

---

## Requirements

Ensure you have the following Python libraries installed:

- `pandas` (For handling data)
- `scikit-learn` (For TF-IDF and Cosine Similarity)
- `numpy` (For numerical operations)

To install all dependencies at once, run:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains all necessary dependencies for the project.

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/aeyouseff/EGY-MOV-RECS.git
```

2. **Navigate to the project directory:**

```bash
cd EGY-MOV-RECS
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Ensure the dataset is available:**

Make sure `EgyptionMoviesDataset.csv` is in the project directory and contains these columns:
- `title` (Movie title)
- `genres` (Genres of the movie)
- `keywords` (Movie-related keywords)
- `tagline` (Movie tagline)
- `cast` (List of actors)
- `director` (Movie director)

5. **Run the system:**

```bash
python egyptian_movies_recommendation.py
```

---

## Usage

### **Method 1: Import & Call the Function**
1. **Import the recommendation function:**
```python
from egyptian_movies_recommendation import recommend_movie
```

2. **Call the function with a movie title:**
```python
movie_title = "الفيل الأزرق"  # Example
recommended_movies = recommend_movie(movie_title)
print("Recommended Movies:", recommended_movies)
```

3. **Expected Output:**
```bash
🎥 Recommended Movies: ['الفيل الأزرق 2', 'كيرة والجن', 'تراب الماس']
```

---

### **Method 2: Run the Script in Terminal**
You can run the script directly in the terminal:

```bash
python egyptian_movies_recommendation.py
```

The system will prompt you to enter a movie name and display recommendations.

Example Interaction:
```
🎥 نظام توصية الأفلام المصرية
🔹 اكتب اسم فيلم مصري وشوف التوصيات لأفلام مشابهة.


🔍 أدخل اسم الفيلم: الفيل الأزرق

🎥 أفلام مشابهة:
✅ الفيل الأزرق 2
✅ كيرة والجن
✅ تراب الماس

--------------------------------------------------

## **How It Works**

### **1️⃣ Data Preprocessing**
- The dataset is loaded from `EgyptionMoviesDataset.csv`.
- Missing values in columns like `genres`, `keywords`, `tagline`, `cast`, and `director` are replaced with `"Unknown"` or `"No Data"`.
- Movie titles are **cleaned and normalized** using **`clean_text()`** to ensure accurate matching.

### **2️⃣ Feature Combination**
Relevant text features (`genres`, `keywords`, `tagline`, `cast`, and `director`) are combined into a single string for each movie.

```python
combined_features = movies_dataset[selected_features].agg(' '.join, axis=1)
```

### **3️⃣ Text Vectorization (TF-IDF)**
The **TF-IDF Vectorizer** converts text into numerical form with:
- **Bi-grams (ngram_range=(1,2))** to consider word pairs.
- **Custom Arabic stop words** to remove unimportant words.

```python
vectorizer = TfidfVectorizer(
    min_df=1,
    stop_words=arabic_stop_words,
    lowercase=True,
    ngram_range=(1, 2),
    max_features=1000,
    sublinear_tf=True
)
```

### **4️⃣ Cosine Similarity Calculation**
The similarity between movies is calculated based on their TF-IDF vectors.

```python
similarity_matrix = cosine_similarity(feature_matrix)
```

### **5️⃣ Movie Recommendation**
- The system finds the **index of the input movie**.
- It retrieves similarity scores and finds the **top 3 closest matches**.

```python
def recommend_movie(movie_name):
    movie_name = clean_text(movie_name)  # Normalize user input

    if movie_name not in movie_titles_list:
        return ["❌ الفيلم ده مش موجود هنا لسه، جرب فيلم تاني."]

    movie_index = movie_titles_list.index(movie_name)
    similarity_scores = similarity_matrix[movie_index]
    similar_movies_indices = np.argsort(similarity_scores)[::-1][1:4]
    recommended_movies = [movie_titles_dict[movie_titles_list[i]] for i in similar_movies_indices]

    return recommended_movies
```

### **6️⃣ Example Output**
```bash
🎥 Recommended Movies: ['الفيل الأزرق 2', 'كيرة والجن', 'تراب الماس']
```

---

## **Contributing**
Want to contribute? Feel free to fork the project and submit a **Pull Request** with improvements or bug fixes.

