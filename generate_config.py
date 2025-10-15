import os

import yaml


def read_configs(config_path=None):
    if config_path == None:
        config_file = r'config.yaml'
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), config_file)
    with open(config_path, 'r') as f:
        config_template = yaml.safe_load(f)
    return config_template

def read_dataset_meta(folder_path=None):
    if folder_path == None:
        dataset_meta_file = r'description.yaml'
        dataset_meta_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataset_meta_file)
    with open(dataset_meta_path, 'r') as f:
        dataset_meta = yaml.safe_load(f)
    return dataset_meta


if __name__ == "__main__":
    print(read_dataset_meta('./publish'))