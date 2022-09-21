import logging
import sys

LOG_FILE_DIR = "."
LOG_FILE_PATH = LOG_FILE_DIR + "\\log.log"

logging.basicConfig(level=logging.INFO)
log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s")
root_logger = logging.getLogger(logging.basicConfig())

file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setFormatter(log_formatter)
root_logger.addHandler(file_handler)
