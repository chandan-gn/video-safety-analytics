import yaml


def load_config(path="pipeline.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
