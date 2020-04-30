"""
Authors:
    Jason Youn - jyoun@ucdavis.edu

Description:
    Wrapper class to process the .ini configuration files.

To-do:
"""
# standard imports
import configparser
import logging as log


class ConfigParser:
    """
    Config parser.
    """
    def __init__(self, filepath):
        """
        Class initializer for ConfigParser.

        Inputs:
            filepath: (str) Filepath of the .ini config file.
        """
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        self.config.read(filepath)

    @staticmethod
    def _print_no_key_warning(key):
        """
        (Private) Log warning if the query key does not exist.

        Inputs:
            key: (str) Failed key that does not exist.
        """
        log.warning('Failed to read key \'%s\'. Returning None instead.', key)

    def append(self, section, entry):
        """
        Append to the configuration.

        Inputs:
            section: section name to append
            entry: items to append to specified section
                in dictionary format
        """
        self.config[section] = entry

    def overwrite(self, key, value, section='DEFAULT'):
        self.config[section][key] = value

    def write(self, filepath):
        """
        Write configuration.

        Inputs:
            filepath: (str) File path to save the config file.
        """
        with open(filepath, 'w') as configfile:
            self.config.write(configfile)

    def sections(self):
        """
        Return the sections of the file.

        Returns:
            (list) List of sections, where each section is in string.
        """
        return self.config.sections()

    def options(self, section='DEFAULT'):
        """
        Return the options (keys) under the specified section.

        Inputs:
            section: (str, optional) Section to read the options from.

        Returns:
            (list) List of options, where each option is in string.
        """
        return self.config.options(section)

    def getstr(self, key, section='DEFAULT'):
        """
        Get key from configuration in string.

        Inputs:
            key: (str) Key to fetch.
            section: (str, optional) Section to fetch from.

        Returns:
            (str) Configuration in string format.
        """
        try:
            return self.config[section][key]
        except KeyError:
            ConfigParser._print_no_key_warning(key)
            return None

    def getint(self, key, section='DEFAULT'):
        """
        Get key from configuration in integer.

        Inputs:
            key: (str) Key to fetch.
            section: (str, optional) Section to fetch from.

        Returns:
            (int) Configuration in integer format.
            If key does not exist, return None.
        """
        try:
            return self.config.getint(section, key)
        except (KeyError, configparser.NoOptionError):
            ConfigParser._print_no_key_warning(key)
            return None

    def getbool(self, key, section='DEFAULT'):
        """
        Get key from configuration in boolean.

        Inputs:
            key: (str) Key to fetch.
            section: (str, optional) Section to fetch from.

        Returns:
            (bool) Configuration in boolean format.
            If key does not exist, return None.
        """
        try:
            return self.config.getboolean(section, key)
        except KeyError:
            ConfigParser._print_no_key_warning(key)
            return None

    def getfloat(self, key, section='DEFAULT'):
        """
        Get key from configuration in float.

        Inputs:
            key: (str) Key to fetch.
            section: (str, optional) Section to fetch from.

        Returns:
            (float) Configuration in float format.
            If key does not exist, return None.
        """
        try:
            return self.config.getfloat(section, key)
        except KeyError:
            ConfigParser._print_no_key_warning(key)
            return None

    def get_str_list(self, key, delim=', ', section='DEFAULT'):
        """
        Get key from configuration in list of strings.

        Inputs:
            key: (str) Key to fetch.
            delim: (str, optional) Delimiter to use for splitting.
            section: (str, optional) Section to fetch from.

        Returns:
            (list) Configuration in list of strings.
            If key does not exist, return None.
        """
        try:
            return self.config[section][key].split(delim)
        except KeyError:
            ConfigParser._print_no_key_warning(key)
            return None

    def get_section_as_dict(self, section='DEFAULT', value_delim=','):
        """
        For the specified section, return all its options
        and values as a dictionary.

        Inputs:
            section: (str, optional) Section to fetch from.
            value_delim: (str, optional) Delimiter for splitting the values.
                If None specified, value will not be processed and
                returned as string.

        Returns:
            dictionary: (dict) Dictionary where key is the option
                and value is a string or list of string depending
                on the 'value_delim' parameter.
                ex) dictionary = {
                        'option1' = ['value1', 'value2']
                        'option2' = ['value3']
                    }
        """
        dictionary = {}

        for key in self.options(section):
            value = self.get_str(key, section=section)

            if value_delim:
                value = value.split(value_delim)
                value = [item.strip() for item in value]

            dictionary[key] = value

        return dictionary
