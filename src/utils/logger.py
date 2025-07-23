"""
Enhanced logging configuration for enterprise-level application
"""
import logging
import logging.handlers
import os
from datetime import datetime
from config.settings import settings


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        if hasattr(record, 'levelname'):
            color = self.COLORS.get(record.levelname, '')
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging():
    """Setup comprehensive logging configuration"""
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # File handler for all logs
    file_handler = logging.handlers.RotatingFileHandler(
        settings.log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        "logs/errors.log",
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s'
    )
    error_handler.setFormatter(error_formatter)
    root_logger.addHandler(error_handler)
    
    # Audit log handler
    audit_handler = logging.handlers.RotatingFileHandler(
        "logs/audit.log",
        maxBytes=20*1024*1024,  # 20MB
        backupCount=10
    )
    audit_formatter = logging.Formatter(
        '%(asctime)s - AUDIT - %(message)s'
    )
    audit_handler.setFormatter(audit_formatter)
    
    # Create audit logger
    audit_logger = logging.getLogger('audit')
    audit_logger.addHandler(audit_handler)
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False
    
    # Suppress noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)


def audit_log(action: str, user_id: str = None, details: dict = None):
    """Log audit events"""
    audit_logger = logging.getLogger('audit')
    
    audit_data = {
        'action': action,
        'timestamp': datetime.utcnow().isoformat(),
        'user_id': user_id,
        'details': details or {}
    }
    
    audit_logger.info(f"{action} | {audit_data}")


# Setup logging on import
setup_logging()
