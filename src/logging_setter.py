import logging.config
import os

abs_path_for_log_file = os.path.abspath("app.log")

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "DEBUG",
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "default",
            "level": "DEBUG",
            "filename": abs_path_for_log_file,
        },
    },
    "root": {"level": "DEBUG", "handlers": ["console", "file"],},
}

logging.config.dictConfig(LOGGING_CONFIG)

logger = logging.getLogger()
