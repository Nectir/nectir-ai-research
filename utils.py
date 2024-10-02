import os
import sys


def add_source_root_to_system_path(path):
    """Adds the given path to sys.path if it's not already there."""
    source_root = os.path.abspath(path)
    if source_root not in sys.path:
        sys.path.append(source_root)

def get_configs_path(root_name, config_filename="configs.yaml"):
    """
    Constructs the path to a configs file relative to the source root.

    Parameters:
        root_name (str): The name of the directory added to sys.path (e.g., "lapu-research").
        config_filename (str): The name of the configuration file (default: "configs.yaml").

    Returns:
        str: The absolute path to the configuration file.
    """
    source_root = os.path.abspath(root_name)
    return os.path.join(source_root, config_filename)
