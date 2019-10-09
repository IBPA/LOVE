#!/usr/bin/env bash

# Filename: utilities.sh
#
# Authors:
#   Jason Youn -jyoun@ucdavis.edu
#
# Description:
#   Bash utility functions.

function dir_exists () {
    # Check if directory exists or not.
    #
    # Inputs:
    #   $1: Directory to check.
    #
    # Returns:
    #   0: Directory exists.
    #   1: Directory does not exist.
    if [ -d "$1" ]; then
        return 0
    else
        return 1
    fi
}

function file_exists () {
    # Check if file exists or not.
    #
    # Inputs:
    #   $1: File to check.
    #
    # Returns:
    #   0: File exists.
    #   1: File does not exist.
    if [ -f "$1" ]; then
        return 0
    else
        return 1
    fi
}

function dir_is_empty () {
    # Check if directory is empty or not.
    #
    # Inputs:
    #   $1: Directory to check.
    #
    # Returns:
    #   0: Directory is empty.
    #   1: Directory is not empty.
    if [ -z "$(ls $1)" ]; then
       return 0
    else
       return 1
    fi
}

function dir_exists_and_is_empty () {
    # Check if directory exists and is also empty.
    #
    # Inputs:
    #   $1: Directory to check.
    #
    # Returns:
    #   0: Directory exists and is empty.
    #   1: Otherwise.
    if ! dir_exists $1; then
        return 1
    fi

    if ! dir_is_empty $1; then
        return 1
    fi

    return 0
}

function dir_exists_and_is_not_empty () {
    # Check if directory exists and is not empty.
    #
    # Inputs:
    #   $1: Directory to check.
    #
    # Returns:
    #   0: Directory exists and is not empty.
    #   1: Otherwise.
    if ! dir_exists $1; then
        return 1
    fi

    if dir_is_empty $1; then
        return 1
    fi

    return 0
}

function make_dir () {
    # Make directory using -p option.
    #
    # Inputs:
    #   $1: Directory to make.
    mkdir -p $1
}
