import json


def load_config(config_filepath: str):
    """Loads a .json configuration file given a filepath."""
    try:
        with open(config_filepath) as config_file:
            return json.load(config_file)
    except IOError as e:
        raise e
