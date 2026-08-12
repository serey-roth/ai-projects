from pathlib import Path
from dotenv import load_dotenv

def load_env_dev():
    ROOT_PATH = next(p for p in Path(__file__).resolve().parents if (p / ".gitignore").exists() or (p / "requirements.txt").exists())
    load_dotenv(ROOT_PATH / ".env.local")