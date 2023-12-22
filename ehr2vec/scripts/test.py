# from ehr2vec.model import model
from mod import foo
import sys
import logging

class LoggerToFileLike:
    def __init__(self, logger, log_level):
        self.logger = logger
        self.log_level = log_level

    def write(self, message):
        # Avoid log messages from empty strings or just newlines
        if message.strip() != "":
            self.logger.log(self.log_level, message.strip())

    def flush(self):
        pass


logging.basicConfig(level=logging.INFO, filename="test",)
logger = logging.getLogger(__name__)

# Redirect standard output to the logger
sys.stdout = LoggerToFileLike(logger, logging.INFO)

print("This will be logged instead of printed.")
foo()