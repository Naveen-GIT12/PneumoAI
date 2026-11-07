
"""
Django settings for Project project.
"""

from pathlib import Path
import os
from dotenv import load_dotenv


# -------------------------------------------------------
# BASE DIRECTORY AND ENVIRONMENT VARIABLES
# -------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from the project-root .env file
load_dotenv(BASE_DIR / ".env")


# -------------------------------------------------------
# SECURITY SETTINGS
# -------------------------------------------------------

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY")

if not SECRET_KEY:
    raise ValueError(
        "DJANGO_SECRET_KEY is missing. Please configure it in the project-root .env file."
    )

DEBUG = True

ALLOWED_HOSTS = []


# -------------------------------------------------------
# APPLICATION DEFINITION
# -------------------------------------------------------

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "social_django",
    "App",
]


# -------------------------------------------------------
# MIDDLEWARE
# -------------------------------------------------------

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]


# -------------------------------------------------------
# URL CONFIGURATION
# -------------------------------------------------------

ROOT_URLCONF = "Project.urls"


# -------------------------------------------------------
# TEMPLATES
# -------------------------------------------------------

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [os.path.join(BASE_DIR, "Templates")],
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


WSGI_APPLICATION = "Project.wsgi.application"


# -------------------------------------------------------
# DATABASE
# -------------------------------------------------------

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}


# -------------------------------------------------------
# PASSWORD VALIDATION
# -------------------------------------------------------

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


# -------------------------------------------------------
# INTERNATIONALIZATION
# -------------------------------------------------------

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True


# -------------------------------------------------------
# STATIC FILES
# -------------------------------------------------------

STATIC_URL = "/static/"

STATICFILES_DIRS = [
    BASE_DIR / "static",
]


# -------------------------------------------------------
# MEDIA FILES
# -------------------------------------------------------

MEDIA_URL = "/media/"

MEDIA_ROOT = BASE_DIR / "media"


# -------------------------------------------------------
# DEFAULT PRIMARY KEY
# -------------------------------------------------------

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# -------------------------------------------------------
# LOGIN REDIRECT
# -------------------------------------------------------

LOGIN_URL = "login"

LOGIN_REDIRECT_URL = "/"


# -------------------------------------------------------
# GEMINI API CONFIGURATION
# -------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


# -------------------------------------------------------
# GMAIL SMTP EMAIL CONFIGURATION
# Used for registration verification and login OTP
# -------------------------------------------------------

EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"

EMAIL_HOST = "smtp.gmail.com"

EMAIL_PORT = 587

EMAIL_USE_TLS = True

EMAIL_HOST_USER = os.getenv("EMAIL_HOST_USER")

EMAIL_HOST_PASSWORD = os.getenv("EMAIL_HOST_PASSWORD")

DEFAULT_FROM_EMAIL = EMAIL_HOST_USER or "webmaster@localhost"

EMAIL_TIMEOUT = 20
