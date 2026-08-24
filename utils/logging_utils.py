import os
import warnings
import logging

# global logging obj
_logger = None

warning_once_op_set = set()

def read_logging_level():
    level = os.environ.get("TORCHDUMP_LOG_LEVEL", "INFO").upper()
    if level not in ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]:
        warnings.warn("The value you set is invalid for TORCHDUMP_LOG_LEVEL, "
                      "will be set to 'INFO'. Please pass value from one of "
                      "['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']")
        level = "INFO"
    return level

def get_logger():
    global _logger
    if _logger is None:
        # create logger
        _logger = logging.getLogger("torchdump")
        _logger.setLevel(read_logging_level())
        # avoid printing twice in subprocess
        _logger.propagate = False

        # create console handler and set formatter
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s [%(name)s](%(levelname)s): %(message)s')
        console_handler.setFormatter(formatter)

        # add handler to logger
        if not _logger.handlers:
            _logger.addHandler(console_handler)

    return _logger

def warning_once(message):
    global warning_once_op_set
    if message not in warning_once_op_set:
        warning_once_op_set.add(message)
        get_logger().warning(message)
