# Book Recommendation API Documentation

This API provides book recommendations based on popular books and similarity scores using machine learning models. The API has two main functionalities:

1. **Get Popular Books**
2. **Get Book Recommendations**

## Base URL
The base URL for this API is:
```
https://bookrecommendationsystem-4e5m.onrender.com/
```

---

## Endpoints

### 1. **GET /**

**Description**:  
This is the home route for the API. It returns a simple message indicating that the API is running.

**Response**:
```json
{
  "message": "Book Recommendation API is running."
}
```

---

### 2. **GET /popular**

**Description**:  
This endpoint returns a list of popular books stored in the `popular_df` DataFrame. It includes the book's details such as title, author, or other metadata, depending on what is available in the DataFrame.

**Response**:
```json
[
  {
    "book_name": "Book Title 1",
    "author": "Author 1",
    "other_field": "Field Value"
  },
  {
    "book_name": "Book Title 2",
    "author": "Author 2",
    "other_field": "Field Value"
  },
  ...
]
```

**HTTP Status Codes**:
- **200 OK**: Successfully fetched popular books.
- **500 Internal Server Error**: If there’s a server error.

---

### 3. **POST /recommend**

**Description**:  
This endpoint returns a list of recommended books based on a given book. The recommendations are made by finding similar books from the precomputed `similarity_scores` and `pt` DataFrame. 

**Request Body** (JSON):
```json
{
  "book_name": "Book Title"
}
```

- **book_name**: The title of the book you want recommendations for. It should be present in the `pt.index` for the recommendation to work.

**Response**:
```json
{
  "recommended_books": [
    "Recommended Book 1",
    "Recommended Book 2",
    "Recommended Book 3",
    "Recommended Book 4",
    "Recommended Book 5"
  ]
}
```

**Error Responses**:
- **404 Not Found**: If the book name provided is not found in the dataset.
  ```json
  {
    "error": "Book not found"
  }
  ```
- **500 Internal Server Error**: If an unexpected error occurs while processing the request.
  ```json
  {
    "error": "Detailed error message"
  }
  ```

**Example Request**:
```bash
curl -X POST https://bookrecommendationsystem-4e5m.onrender.com/recommend \
  -H "Content-Type: application/json" \
  -d '{"book_name": "The Catcher in the Rye"}'
```

**Example Response**:
```json
{
  "recommended_books": [
    "Book Title 1",
    "Book Title 2",
    "Book Title 3",
    "Book Title 4",
    "Book Title 5"
  ]
}
```

---

## Error Handling

In case of an error, the API will return an appropriate status code along with a detailed error message. The common error codes are:
- **404 Not Found**: When the requested resource is not found.
- **500 Internal Server Error**: When there is an error in processing the request on the server.

---

## Dependencies

To run this API, you will need the following Python packages:
- Flask
- Numpy
- Pickle

You can install them using:
```bash
pip install flask numpy
```

Ensure you also have the necessary pickled model files:
- `popular.pkl`
- `pt.pkl`
- `books.pkl`
- `similarity_scores.pkl`

---

## Running the API

To run the Flask API locally:
1. Clone or download the project.
2. Navigate to the directory where the script is located.
3. Run the following command to start the Flask development server:
   ```bash
   python app.py
   ```

This will start the server at `http://localhost:10000`.