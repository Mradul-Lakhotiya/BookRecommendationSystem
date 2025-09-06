# BookRecommendationSystem

A basic book recommendation system that provides book suggestions using content-based and popularity-based algorithms. Built for learning and future expansion.

---

## 🧑‍💻 Features

- **Get Popular Books:**  
  Returns a list of popular books based on data analysis.  
  Endpoint: `/popular`

- **Get Book Recommendations:**  
  Suggests books similar to a given book using precomputed similarity scores.  
  Endpoint: `/recommend` (POST with book name or ISBN)

- **Content-Based Recommendation:**  
  Uses book attributes (title, author, description, genres) and TF-IDF vectorization to find similar books.

- **Popularity-Based Recommendation:**  
  Ranks books based on popularity metrics and can filter by genres.

- **API Documentation:**  
  The API provides clear endpoints for fetching popular books, getting recommendations, and basic status checks.

---

## 🧠 Machine Learning Concepts Used

- **Feature Engineering:**  
  - Combines multiple book attributes (title, author, genres, description) into a single feature set for similarity computation.

- **Vectorization:**  
  - Uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert text features into numerical vectors.

- **Nearest Neighbors Search:**  
  - Applies k-nearest neighbors algorithm to find books most similar to a given book based on vectorized features.

- **Popularity Metrics:**  
  - Aggregates book ratings and other metrics to determine popularity.

- **Similarity Computation:**  
  - Precomputes similarity scores between books for fast retrieval.

- **Model Persistence:**  
  - Stores trained models and preprocessed data using Pickle and Joblib for efficient API serving.

- **(Expandable) Neural Networks & Hybrid Approaches:**  
  - Future expansion can include deep learning methods (neural networks), collaborative filtering, and hybrid recommender systems.

---

## 📊 Algorithms & Techniques Used

- **Content-Based Filtering:**  
  - Vectorizes combined book features (title, author, genres, description) using TF-IDF.
  - Computes book similarity via k-nearest neighbors on the TF-IDF matrix.
  - Returns recommendations based on most similar books.

- **Popularity-Based Filtering:**  
  - Ranks books by popularity metrics.
  - Supports genre filtering and multi-genre matching (any/all).

- **Precomputed Similarity Scores:**  
  Utilizes pre-trained models and pickled similarity data for fast recommendation queries.

- **API Framework:**  
  - Built using Flask and FastAPI for serving recommendations.
  - Error handling for missing books, server errors, and invalid queries.

---

## 🛠️ Data & Dependencies

- **Python Packages:**  
  - Flask, FastAPI
  - Numpy, Pickle, Joblib
  - Scikit-learn (for TF-IDF and neighbors)
- **Data Files:**  
  - `popular.pkl`, `pt.pkl`, `books.pkl`, `similarity_scores.pkl`, and others for model persistence.

---

## 🔮 Future Expansion Ideas

- Integrate neural network-based recommendation algorithms to improve suggestions for larger datasets.
- Add collaborative filtering and hybrid recommendation techniques.
- Enhance context-awareness (e.g., recommendations by time, user location, or device).
- Expand filtering options and recommendation diversity.
- Improve API endpoints for batch recommendations and user personalization.

---

## 🚀 Running Locally

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install flask numpy scikit-learn joblib
   ```
3. Place required data/model files in the working directory.
4. Run the API server:
   ```bash
   python app.py
   ```
5. Access the API at `http://localhost:10000`.

---

_Made by [Mradul-Lakhotiya](https://github.com/Mradul-Lakhotiya) for learning and experimentation._
