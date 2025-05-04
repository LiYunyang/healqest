import logging

from .startup import setup_logger

# Check if logging has already been configured
# If not, set up the logger at "INFO" level.
if not logging.getLogger().hasHandlers():
    setup_logger(verbose=3)
