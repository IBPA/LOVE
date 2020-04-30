"""
Authors:
    Jason Youn -jyoun@ucdavis.edu

Description:
    Utility functions used in the project.

To-do:
"""
import os


def dir_exists(directory):
    """
    Check if directory exists.

    Inputs:
        directory: (str) Directory to check.

    Returns:
        (bool) True if directory exists, False otherwise.
    """
    return os.path.isdir(directory)


def file_exists(filepath):
    """
    Check if file exists.

    Inputs:
        filepath: (str) File to check.

    Returns:
        (bool) True if file exists, False otherwise.
    """
    return os.path.isfile(filepath)
