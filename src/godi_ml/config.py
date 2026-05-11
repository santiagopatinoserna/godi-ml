"""Configuration settings for godi-ml."""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global settings loaded from env vars (prefix GODI_ML_) or .env file."""

    model_config = SettingsConfigDict(
        env_prefix="GODI_ML_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    random_seed: int = 42
    n_jobs: int = -1
    test_size: float = 0.2
    val_size: float = 0.2
    runs_dir: str = "godi-ml-runs"
    lang: Literal["en", "es", "fr"] = "en"
