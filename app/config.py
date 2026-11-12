import json

from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./alocks.db"
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:5174", "http://localhost:5175", "http://localhost:5176", "http://localhost:3000", "https://alocks-admin.pages.dev", "https://kernelmcgregor.github.io"]
    ENV: str = "dev"
    ODDS_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-haiku-4-5-20251001"
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    PREVIEW_MODEL: str = "deepseek-chat"
    # Fights generated concurrently. Each is one blocking HTTPS call, so this is
    # wall-clock divided by roughly this factor. DeepSeek does not publish a hard
    # request cap; 8 has been comfortable and leaves headroom.
    PREVIEW_WORKERS: int = 8
    # Per-request ceiling. The SDK default is long enough that one stalled
    # completion holds a worker slot for minutes.
    PREVIEW_TIMEOUT_SECONDS: float = 180.0
    ADMIN_API_KEY: str = ""
    # Google Images via SerpAPI, for article artwork (services/ufc/article_images.py).
    SERPAPI_KEY: str = ""
    # Newsroom (services/ufc/newsroom): researched articles for the top of each card.
    NEWSROOM_TOP_N: int = 5
    NEWSROOM_DAYS_AHEAD: int = 10
    NEWSROOM_MONTHLY_BUDGET_USD: float = 10.0
    # DeepSeek list prices per million tokens (cache miss), for the budget meter. Kept
    # on the high side so the cap errs early.
    NEWSROOM_PRICE_IN_PER_M: float = 0.30
    NEWSROOM_PRICE_OUT_PER_M: float = 0.50

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
