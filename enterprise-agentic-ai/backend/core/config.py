# Configuration settings for the backend

DATABASE_URL = "sqlite:///./database.db"  # Database connection string
SECRET_KEY = "your_secret_key"  # Secret key for authentication
ALGORITHM = "HS256"  # Algorithm used for encoding JWT tokens
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # Token expiration time in minutes

# CORS settings
CORS_ORIGINS = [
    "http://localhost:3000",  # Frontend URL
    "https://yourdomain.com",  # Production URL
]

# Logging settings
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        },
    },
    "loggers": {
        "uvicorn": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}