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
            "level": "INFO",
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "default",
            "level": "INFO",
            "filename": abs_path_for_log_file,
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "file"],},
    "loggers": {
        "pdfplumber": {
            "level": "WARNING",
            "handlers": ["console", "file"],
            "propagate": False,
        },
        "pdfminer": {
            "level": "WARNING",
            "handlers": ["console", "file"],
            "propagate": False,
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger()
