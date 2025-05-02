import ast
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pickle
from scipy.sparse import save_npz, load_npz

class PopularityRecommender:
    """
    A faster popularity-based recommender using precomputed indices.

    Usage:
        rec = PopularityRecommender(df, min_ratings=10)
        rec.get_popular_isbns()  # overall top
        rec.get_popular_isbns(genre='History', top=10)
        rec.get_popular_isbns(genre=['History','Civil War'], top=5, match_mode='all')
    """
    def __init__(self, df, min_ratings=10):
        # Copy and preprocess
        self.df = df.copy()
        # Parse genre_list safely
        self.df['genre_list'] = self.df['genre_list'].apply(self._parse_genre_list)
        # Filter by minimum ratings
        self.df = self.df[self.df['totalratings'] > min_ratings]
        # Compute popularity score once
        self.df['popularity_score'] = self.df['rating'] * np.log1p(self.df['totalratings'])
        # Build reverse index: genre -> set of DataFrame indices
        self.genre_index = {}
        for idx, genres in self.df['genre_list'].items():
            for g in genres:
                self.genre_index.setdefault(g, set()).add(idx)

    def _parse_genre_list(self, val):
        """
        Safely parse the genre_list field, handling malformed strings.
        """
        if isinstance(val, list):
            return val
        if not isinstance(val, str):
            return []
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        s = val.strip().lstrip('[').rstrip(']')
        items = []
        for part in s.split(','):
            item = part.strip().strip("'\" ")
            if item and '...' not in item:
                items.append(item)
        return items

    def get_popular_isbns(self, genre=None, top=20, match_mode='all'):
        """
        Retrieve top ISBNs by popularity, optionally filtered by genre(s).

        Parameters:
        - genre: None, str, or list of str
        - top: int
        - match_mode: 'any' or 'all'

        Returns:
        - List of ISBN strings, or None if no matches.
        """
        # Determine candidate indices
        if genre is None:
            candidate_idx = list(self.df.index)
        else:
            genres = [genre] if isinstance(genre, str) else list(genre)
            sets = [self.genre_index.get(g, set()) for g in genres]
            if match_mode == 'any':
                candidate_idx = list(set().union(*sets))
            elif match_mode == 'all':
                candidate_idx = list(set.intersection(*sets)) if sets else []
            else:
                raise ValueError("match_mode must be 'any' or 'all'")

        if not candidate_idx:
            return None

        # Subset and sort by popularity
        sub = self.df.loc[candidate_idx]
        top_df = sub.nlargest(top, 'popularity_score')
        return top_df['isbn'].tolist()

class ContentBasedRecommender:
    def __init__(self, top_k=20):
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.model = NearestNeighbors(n_neighbors=top_k + 1, metric='cosine')
        self.similarity_dict = {}
        self.df_clean = None
        self.tfidf_matrix = None

    def _parse_genre_list(self, val):
        if isinstance(val, list): return val
        if not isinstance(val, str): return []
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list): return parsed
        except: pass
        s = val.strip().lstrip('[').rstrip(']')
        items = []
        for part in s.split(','):
            item = part.strip().strip("'\" ")
            if item and '...' not in item: items.append(item)
        return items

    def _clean_text(self, x):
        if not isinstance(x, str): return ''
        x = x.lower()
        x = re.sub(r'[^\w\s]', ' ', x)
        x = re.sub(r'\s+', ' ', x)
        return x.strip()

    def preprocess(self, df):
        # Check if necessary columns exist
        required_columns = ['title', 'author', 'desc', 'genre_list']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        df = df.copy()
        df['genre_list'] = df['genre_list'].apply(self._parse_genre_list)
        df['title_clean'] = df['title'].apply(self._clean_text)
        df['author_clean'] = df['author'].apply(self._clean_text)
        df['desc_clean'] = df['desc'].apply(self._clean_text)
        df['genres_clean'] = df['genre_list'].apply(lambda genres: ' '.join([self._clean_text(g) for g in genres]))
        df['combined_features'] = df['title_clean'] + ' ' + df['author_clean'] + ' ' + df['desc_clean'] + ' ' + df['genres_clean']
        self.df_clean = df
        return df

    def build_similarity(self):
        if self.df_clean is None:
            raise ValueError("Dataframe is not preprocessed. Call preprocess() first.")
        
        tfidf_matrix = self.vectorizer.fit_transform(self.df_clean['combined_features'])
        self.tfidf_matrix = tfidf_matrix
        self.model.fit(tfidf_matrix)

        distances, indices = self.model.kneighbors(tfidf_matrix)
        self.similarity_dict = {
            i: [idx for idx in neighbors[1:]]  # skip self
            for i, neighbors in enumerate(indices)
        }

    def recommend_by_isbn(self, isbn):
        if isbn not in self.df_clean['isbn'].values:
            return []
        idx = self.df_clean[self.df_clean['isbn'] == isbn].index[0]
        similar_idxs = self.similarity_dict.get(idx, [])
        return self.df_clean.iloc[similar_idxs]['isbn'].tolist()

    def save(self, path_prefix):
        columns_to_keep = ['isbn', 'title_clean', 'author_clean', 'desc_clean', 'genres_clean', 'combined_features']
        self.df_clean = self.df_clean[columns_to_keep]
        
        # Save vectorizer, model, similarity_dict, and tfidf_matrix to files
        with open(f'{path_prefix}_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(f'{path_prefix}_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open(f'{path_prefix}_similarity_dict.pkl', 'wb') as f:
            pickle.dump(self.similarity_dict, f)
        self.df_clean.to_csv(f'{path_prefix}_df_clean.csv', index=False)
        save_npz(f'{path_prefix}_tfidf_matrix.npz', self.tfidf_matrix)

    def load(self, path_prefix):
        # Load vectorizer, model, similarity_dict, and tfidf_matrix from files
        with open(f'{path_prefix}_vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(f'{path_prefix}_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        with open(f'{path_prefix}_similarity_dict.pkl', 'rb') as f:
            self.similarity_dict = pickle.load(f)
        self.df_clean = pd.read_csv(f'{path_prefix}_df_clean.csv')
        self.tfidf_matrix = load_npz(f'{path_prefix}_tfidf_matrix.npz')
