"""
Authors:
    Jason Youn -jyoun@ucdavis.edu

Description:
    Utility functions used in the project.

To-do:
"""
import os
import pickle


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


def save_pkl(obj, save_to):
    """
    Pickle the object.

    Inputs:
        obj: (object) Object to pickle.
        save_to: (str) Filepath to pickle the object to.
    """
    with open(save_to, 'wb') as fid:
        pickle.dump(obj, fid)


def load_pkl(load_from):
    """
    Load the pickled object.

    Inputs:
        save_to: (str) Filepath to pickle the object to.

    Returns:
        (object) Loaded object.
    """
    with open(load_from, 'rb') as fid:
        obj = pickle.load(fid)

    return obj

