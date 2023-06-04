class TqdmToLogger(object):
    """File-like object to redirect tqdm to logger"""
    def __init__(self, logger, level=logging.INFO):
       self.logger = logger
       self.level = level

    def write(self, buf):
       for line in buf.rstrip().splitlines():
          self.logger.log(self.level, line)

    def flush(self):
       pass