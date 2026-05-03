import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY    = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
DEBUG         = os.getenv("DEBUG", "True") == "True"
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "localhost 127.0.0.1").split()

INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "channels",
    "predictor",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "isl_web.urls"

TEMPLATES = [{
    "BACKEND": "django.template.backends.django.DjangoTemplates",
    "DIRS": [],
    "APP_DIRS": True,
    "OPTIONS": {"context_processors": [
        "django.template.context_processors.request",
    ]},
}]

ASGI_APPLICATION = "isl_web.asgi.application"
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels.layers.InMemoryChannelLayer",
    }
}

STATIC_URL  = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ── ISL model paths ───────────────────────────────────────────────────────────
# On Render: relative paths (files downloaded by startup.py into word/ sentence/)
# Locally:   set ISL_ROOT in .env to point to your ISL project root

_ISL_ROOT = os.getenv("ISL_ROOT", str(BASE_DIR))

ISL_WORD_MODEL_PATH     = os.getenv("ISL_WORD_MODEL_PATH",
    os.path.join(_ISL_ROOT, "word", "isl_best_model.keras"))
ISL_LABEL_MAP_PATH      = os.getenv("ISL_LABEL_MAP_PATH",
    os.path.join(_ISL_ROOT, "word", "label_map.json"))
ISL_SENTENCE_MODEL_PATH = os.getenv("ISL_SENTENCE_MODEL_PATH",
    os.path.join(_ISL_ROOT, "sentence", "isl_sentence_model.keras"))
ISL_SENTENCE_LABEL_PATH = os.getenv("ISL_SENTENCE_LABEL_PATH",
    os.path.join(_ISL_ROOT, "sentence", "sentence_label_map.json"))
