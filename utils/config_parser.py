"""
Authors:
    Jason Youn - jyoun@ucdavis.edu

Description:
    Wrapper class to process the .ini configuration files.

To-do:
"""
# standard imports
import configparser

class ConfigParser:
    """
    Config parser.
    """
    def __init__(self, filepath):
        """
        Class initializer for ConfigParser.

        Inputs:
            filepath: (str) File path of .ini config file.
        """
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        self.config.read(filepath)

    def write(self, filepath):
        """
        Write configuration.

        Inputs:
            filepath: (str) File path to save the config file.
        """
        self.config.write(filepath)

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
            section: (str) Section to read the options from.

        Returns:
            (list) List of options, where each option is in string.
        """
        return self.config.options(section)

    def getstr(self, key, section='DEFAULT'):
        """
        Get key from configuration in string.

        Inputs:
            key: (str) Key to fetch.
            section: (str) Section to fetch from.

        Returns:
            (str) Configuration in string format.
        """
        return self.config[section][key]

    def getint(self, key, section='DEFAULT'):
        """
        Get key from configuration in integer.

        Inputs:
            key: (str) Key to fetch.
            section: (str) Section to fetch from.

        Returns:
            (int) Configuration in integer format.
        """
        return self.config.getint(section, key)

    def getbool(self, key, section='DEFAULT'):
        """
        Get key from configuration in boolean.

        Inputs:
            key: (str) Key to fetch.
            section: (str) Section to fetch from.

        Returns:
            (bool) Configuration in boolean format.
        """
        return self.config.getboolean(section, key)

    def getfloat(self, key, section='DEFAULT'):
        """
        Get key from configuration in float.

        Inputs:
            key: (str) Key to fetch.
            section: (str) Section to fetch from.

        Returns:
            (float) Configuration in float format.
        """
        return self.config.getfloat(section, key)

    def get_section_as_dict(self, section='DEFAULT', value_delim=','):
        """
        For the specified section, return all its options
        and values as a dictionary.

        Inputs:
            section: (str) Section to fetch from.
            value_delim: (str) Delimiter for splitting the values.
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
            value = self.getstr(key, section=section)

            if value_delim:
                value = value.split(value_delim)
                value = [item.strip() for item in value]

            dictionary[key] = value

        return dictionary
