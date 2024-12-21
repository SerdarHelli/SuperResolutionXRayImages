import yaml

def load_config(config_path="configs/config.yaml"):
    """
    Load the configuration from a YAML file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise Exception(f"Configuration file '{config_path}' not found.")
    except yaml.YAMLError as e:
        raise Exception(f"Error parsing configuration file: {e}")
