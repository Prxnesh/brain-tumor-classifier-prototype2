from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "NeuroVision AI API"
    model_path: Path = ROOT_DIR / "brain_tumor_resnet50.pth"
    mri_dataset_path: Path = ROOT_DIR / "archive MRI"
    allowed_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    model_config = SettingsConfigDict(env_prefix="BRAIN_TUMOR_", env_file=".env")


settings = Settings()
