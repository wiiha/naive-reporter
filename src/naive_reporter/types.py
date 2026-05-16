"""Shared types for naive-reporter."""

from pydantic import BaseModel


class Document(BaseModel):
    """A document with its pre-generated search queries."""

    stem: str
    queries: list[str]


class SearchResult(BaseModel):
    """One search hit."""

    stem: str
    score: float
