from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException, Query

from App.utils import ContentBasedRecommender, PopularityRecommender

# App setup
app = FastAPI(
    title="Book Recommendation API",
    version="1.0.0",
    description="Content-based and popularity-based book recommendations"
)

# Resolve model paths relative to this file
HERE = Path(__file__).parent
CONTENT_PREFIX = HERE / "model_assets" / "content_model" / "book_recommender"
POPULARITY_PATH = HERE / "model_assets" / "popularity_model" / "good_book_pop_rec.pkl"

# Load content‑based model once at startup
content_rec = ContentBasedRecommender()
content_rec.load(str(CONTENT_PREFIX))

# Load popularity‑based recommender once
try:
    pop_rec = joblib.load(str(POPULARITY_PATH))
except FileNotFoundError:
    pop_rec = None  # you can raise an error on first call if you prefer

@app.get("/", summary="Health Check")
def health():
    return {"status": "ok"}

@app.get(
    "/recommend/content",
    summary="Content-Based Recommendation",
    response_description="A list of ISBNs similar to the query ISBN"
)
def recommend_content(
    isbn: str = Query(..., description="ISBN of the book to base recommendations on"),
    top_k: int = Query(None, description="Override number of neighbors (default from training)")
):
    if top_k is not None:
        content_rec.top_k = top_k

    recs = content_rec.recommend_by_isbn(isbn)
    if not recs:
        raise HTTPException(status_code=404, detail=f"No recommendations found for ISBN {isbn}")

    return {"query_isbn": isbn, "recommendations": recs}

@app.get(
    "/recommend/popular",
    summary="Popularity-Based Recommendation",
    response_description="A list of popular ISBNs"
)
def recommend_popular(
    genre: str = Query(None, description="Filter by single genre (e.g. 'History')"),
    top_k: int = Query(20, description="Number of popular books to return"),
    match_mode: str = Query('all', regex="^(any|all)$", description="For multi-genre: 'any' or 'all'")
):
    if pop_rec is None:
        raise HTTPException(status_code=500, detail="Popularity model not loaded")

    try:
        recs = pop_rec.get_popular_isbns(genre=genre, top=top_k, match_mode=match_mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not recs:
        raise HTTPException(status_code=404, detail="No popular books found for given criteria")

    return {"genre": genre, "recommendations": recs}