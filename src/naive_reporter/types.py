"""Shared types for naive-reporter."""

from dataclasses import dataclass

from pydantic import BaseModel


class Document(BaseModel):
    """A document with its pre-generated search queries."""

    stem: str
    queries: list[str]


class SearchResult(BaseModel):
    """One search hit."""

    stem: str
    score: float


@dataclass
class MatchedDocInfo:
    """A document matched by one or more synthetic queries."""

    stem: str
    summary: str
    query_scores: list[tuple[str, float]]


@dataclass
class DocContent:
    """Full text content of a document for LLM context."""

    stem: str
    text: str
