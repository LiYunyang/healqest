import logging
from .log import setup_logger

if not logging.getLogger().hasHandlers():
    setup_logger(verbose=3, verbose_other=2)
