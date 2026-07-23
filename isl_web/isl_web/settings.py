"""
Django settings for isl_web project.

This file merges standard Django/Channels boilerplate with the ISL-specific
model-path settings from the original project, and loads a local .env file
so ISL_ROOT and the model paths actually take effect (this was the cause
of the "model NOT FOUND" errors when running locally).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from a .env file sitting next to manage.py, if present.
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# ── Core ──────────────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "dev-insecure-secret-key-change-me")
DEBUG = os.getenv("DEBUG", "True") == "True"
ALLOWED_HOSTS = [h.strip() for h in os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",") if h.strip()]

# ── Applications ──────────────────────────────────────────────────────────────
INSTALLED_APPS = [
    "daphne",                       # must come first so `runserver` uses ASGI
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "channels",
    "predictor",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "isl_web.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

ASGI_APPLICATION = "isl_web.asgi.application"

CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels.layers.InMemoryChannelLayer",
    },
}

# ── Database ──────────────────────────────────────────────────────────────────
# Simple local SQLite database. The app itself doesn't store predictions in
# it — this only exists because Django's admin/sessions/auth apps expect one.
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Kolkata"
USE_I18N = True
USE_TZ = True

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ── Static files ──────────────────────────────────────────────────────────────
STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
if not DEBUG:
    STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

# ── Production security ───────────────────────────────────────────────────────
if not DEBUG:
    SECURE_BROWSER_XSS_FILTER = True
    SECURE_CONTENT_TYPE_NOSNIFF = True

# ── ISL model paths ────────────────────────────────────────────────────────────
# ISL_ROOT must point at the folder that directly contains word/ and sentence/
# (the same folder predict.py lives in), so predictor_engine.py can
# `import word.sentence_builder` and `import sentence.sentence_model`.
# Set this in your .env file — see .env.example.
ISL_ROOT = os.getenv("ISL_ROOT", "")
if ISL_ROOT:
    os.environ["ISL_ROOT"] = ISL_ROOT  # predictor_engine.py reads this directly

# Model paths for Render (files land in /opt/render/project/src/) and for
# local runs (set absolute paths in .env if the model files live elsewhere).
ISL_WORD_MODEL_PATH     = os.getenv("ISL_WORD_MODEL_PATH",     "word/isl_best_model.keras")
ISL_LABEL_MAP_PATH      = os.getenv("ISL_LABEL_MAP_PATH",      "word/label_map.json")
ISL_SENTENCE_MODEL_PATH = os.getenv("ISL_SENTENCE_MODEL_PATH", "sentence/isl_sentence_model.keras")
ISL_SENTENCE_LABEL_PATH = os.getenv("ISL_SENTENCE_LABEL_PATH", "sentence/sentence_label_map.json")
