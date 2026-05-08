import logging

def init_logger(log_path):
    logger = logging.getLogger(__file__)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        chformatter = logging.Formatter("%(asctime)s - %(message)s",datefmt="%Y-%m-%d %H:%M:%S")
        fhformatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",datefmt="%Y-%m-%d %H:%M:%S")
        ch.setFormatter(chformatter)
        fh.setFormatter(fhformatter)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger
