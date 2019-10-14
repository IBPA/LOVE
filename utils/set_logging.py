"""
Authors:
    Jason Youn -jyoun@ucdavis.edu

Description:
    Function for handling the logging options.

To-do:
"""
# standard imports
import logging as log

def set_logging(log_file=None, log_level=log.DEBUG):
    """
    Configure logging. By default, log to the console.
    If requested, log to a file specified by the user.
    Inputs:
        log_file: (str) Path to save the log file.
        log_level: (log.level) Log level.
    """
    # create logger
    logger = log.getLogger()
    logger.setLevel(log_level)

    # create formatter
    formatter = log.Formatter('%(asctime)s %(levelname)s %(filename)s: %(message)s')

    # create and set file handler if requested
    if log_file:
        file_handler = log.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # create and set console handler
    stream_handler = log.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # set logging level to WARNING for matplotlib
    matplotlib_logger = log.getLogger('matplotlib')
    matplotlib_logger.setLevel(log.WARNING)
