import os
import sys
import utils
import pprint
import dotenv
import argilla as rg

dotenv.load_dotenv()

config = utils.load_yaml(sys.argv[1])

client = rg.Argilla(api_url = os.getenv('API_URL'), api_key = os.getenv('API_KEY'))

dataset = client.datasets(name = config['DATASET_NAME'], workspace = config['WORKSPACE_NAME'])

pprint.pprint(dataset.progress(with_users_distribution=True))

output_filepath = utils.create_output_path(config['OUTPUT_DIR'], config['DATASET_NAME'])

records = dataset.records.to_datasets()

records = records.map(lambda row: {"guidelines": config['GUIDELINES_PATH']})

utils.save_ds_as_json(output_filepath, records)

