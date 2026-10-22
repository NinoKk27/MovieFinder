"""Readable Pinecone diagnostics without raw response bodies or secrets."""
from pinecone import PineconeError, PineconeConnectionError, PineconeTimeoutError
from pinecone.exceptions import PineconeException

PINECONE_ERRORS = (PineconeError, PineconeException)

def pinecone_message(error):
    if isinstance(error, PineconeTimeoutError):
        return "Pinecone timed out. Try again shortly."
    if isinstance(error, PineconeConnectionError):
        return "Could not connect to Pinecone. Check your internet connection."
    status = getattr(error, "status_code", None)
    if status == 401:
        return "Pinecone rejected the API key. Check PINECONE_API_KEY in .env."
    if status == 403:
        return "Pinecone denied access. Check your key permissions and account access."
    if status == 404:
        return "Pinecone could not find the requested resource. Check your index and run --index if needed."
    if status == 429:
        return "Pinecone rate limit or quota reached. Check account limits and try again later."
    if status in (400, 422):
        return "Pinecone rejected the request. Check index schema, cloud/region, and metadata."
    return "Pinecone is unavailable or the request failed. Check your settings and try again later."
