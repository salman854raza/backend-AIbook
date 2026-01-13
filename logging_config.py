import logging
import sys
from datetime import datetime
from typing import Callable, Optional

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration for the application
    """
    # Create logger
    logger = logging.getLogger('rag_chatbot')
    logger.setLevel(getattr(logging, level.upper()))

    # Prevent adding multiple handlers if logger already has handlers
    if logger.handlers:
        return logger

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(console_handler)

    # Add file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def log_progress(current: int, total: int, message: str = "", logger: logging.Logger = None):
    """
    Log progress in a consistent format
    """
    if logger is None:
        logger = logging.getLogger('rag_chatbot')

    percentage = (current / total) * 100 if total > 0 else 0
    logger.info(f"Progress: {current}/{total} ({percentage:.1f}%) - {message}")

def create_progress_callback(logger: logging.Logger = None) -> Callable[[int, int, str], None]:
    """
    Create a progress callback function that logs progress
    """
    if logger is None:
        logger = logging.getLogger('rag_chatbot')

    def progress_callback(current: int, total: int, message: str = ""):
        log_progress(current, total, message, logger)

    return progress_callback

# Set up the root logger when this module is imported
app_logger = setup_logging()

# Example usage of the logging configuration
if __name__ == "__main__":
    # Example of how to use the logging
    logger = setup_logging(level="INFO", log_file="rag_chatbot.log")

    logger.info("Logging system initialized")
    logger.debug("This is a debug message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Example of progress logging
    for i in range(10):
        log_progress(i+1, 10, f"Processing item {i+1}")