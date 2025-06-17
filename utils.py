import os
import sys
import json
import yaml
import datetime
import datasets

def validate_config_path(file_path: str) -> None:
    if len(sys.argv) != 2:
        raise ValueError("This script requires exactly one argument: the path to a YAML file.")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The specified path does not exist or is not a file: {file_path}")
    if not file_path.lower().endswith(('.yaml', '.yml')):
        raise ValueError(f"The specified file is not a YAML file: {file_path}")

def load_yaml(file_path: str) -> dict:
    validate_config_path(file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"YAML file not found at: {file_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")

def load_txt(file_path: str, encoding: str = 'utf-8') -> str:
    with open(file_path, 'r', encoding=encoding) as file:
        return file.read()

def load_json(file_path: str, encoding: str = 'utf-8') -> list[dict]:
    with open(file_path, 'r', encoding=encoding) as f:
        return json.load(f)

def save_json(file_path: str, data: dict, encoding: str = 'utf-8', indent: int = 4) -> None:
    with open(file_path, 'w', encoding=encoding) as f:
        json.dump(data, f, ensure_ascii = False, indent = indent)
    print(f'\n>>> Records saved here: {file_path}\n')

def save_ds_as_json(file_path: str, dataset: datasets.Dataset, encoding: str = 'utf-8', indent: int = 4) -> None:
    data = dataset.to_list()
    save_json(file_path, data, encoding, indent)

def create_output_path(output_dir: str, dataset_name: str, file_extension: str = "json", add_timestamp: bool = True) -> str:
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    timestamp_suffix = f'_{datetime.datetime.now().strftime("%Y%m%d_%H%M")}' if add_timestamp else ''
    return os.path.join(output_dir, f'records_{dataset_name}{timestamp_suffix}.{file_extension}')

