from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "NeuroVision AI API"
    mri_model_path: Path = ROOT_DIR / "brain_tumor_resnet50.pth"
    ct_model_path: Path = ROOT_DIR / "brain_tumor_ct_resnet50.pth"
    mri_dataset_path: Path = ROOT_DIR / "archive MRI"
    ct_dataset_path: Path = ROOT_DIR / "archive ct"
    allowed_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    ollama_enabled: bool = True
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout_seconds: float = 45.0
    model_config = SettingsConfigDict(env_prefix="BRAIN_TUMOR_", env_file=".env")


settings = Settings()
