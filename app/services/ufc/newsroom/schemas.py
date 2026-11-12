"""Plain data passed between the newsroom stages. Nothing here touches the database."""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field

KINDS = ("storyline", "history", "injury", "camp", "quote", "beef", "report")
VENUES = ("press_conference", "interview", "podcast", "social", "report")
TIERS = ("matchup", "recent", "background")
LAYOUTS = ("feature", "rivalry", "magazine", "data", "standard", "brief")


class NewsItem(BaseModel):
    """A headline from a feed, before the story is fetched."""
    title: str
    outlet: str | None = None
    link: str
    published_at: dt.date | None = None
    query: str = ""
    score: float = 0.0


class Article(BaseModel):
    url: str
    outlet: str | None = None
    title: str | None = None
    author: str | None = None
    published_at: dt.date | None = None
    text: str = ""
    tweets: list[str] = Field(default_factory=list)
    videos: list[str] = Field(default_factory=list)
    db_id: int | None = None


class Claim(BaseModel):
    id: int = 0  # DB id once stored; a run-local index before that
    fighter_id: int
    opponent_id: int | None = None
    tier: str = "recent"
    kind: str = "storyline"
    text: str
    quote: str | None = None
    speaker: str | None = None
    outlet: str | None = None
    show: str | None = None
    reporter: str | None = None
    venue: str | None = None
    source_url: str | None = None
    published_at: dt.date | None = None
    embed_url: str | None = None
    newsworthiness: int = 3
    #: The story's text, for the checker; not stored on the claim.
    source_text: str | None = Field(default=None, exclude=True)


class Plan(BaseModel):
    """The editor's outline."""
    angle: str = ""
    headline_hook: str = ""
    lead_claims: list[int] = Field(default_factory=list)
    background_claims: list[int] = Field(default_factory=list)
    pull_quotes: list[int] = Field(default_factory=list)
    counterpoints: list[int] = Field(default_factory=list)
    embeds: list[str] = Field(default_factory=list)


class CheckReport(BaseModel):
    ok: bool
    issues: list[str] = Field(default_factory=list)
    review: dict = Field(default_factory=dict)
