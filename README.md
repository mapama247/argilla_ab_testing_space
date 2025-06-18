# Argilla Spaces for A/B testing

## Steps

### Create HF Space

Deploy Argilla on Hugginface's Spaces following [these steps](https://docs.v1.argilla.io/en/v1.19.0/getting_started/installation/deployments/huggingface-spaces.html).

> <span style="color:red;">**Warning:** HuggingFace Spaces now have persistent storage and this is supported from Argilla 1.11.0 onwards, but you will need to manually activate it via the HuggingFace Spaces settings. Otherwise, unless you’re on a paid space upgrade, after 48 hours of inactivity the space will be shut off and you will lose all the data. To avoid losing data, it is highly recommended to use the persistent storage layer offered by HF.</span>

### Clone this repository
```bash
git clone git@github.com:mapama247/argilla_ab_testing_space.git
cd argilla_ab_testing_space
```

### Install requirements
```bash
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Set API credentials
Create a `.env` file with, at least, two environment variables: `API_URL` and `API_KEY`.
```bash
API_URL=https://mapama247-argilla-test.hf.space
API_KEY=s_wtL8r9tqb3ZsNZz89PqLBkrVotbyymiQx...
ARGILLA_ENABLE_TELEMETRY=0
```

You can easily get your space's credentials by clicking the "Import from Python" button on the top-right corner.

**Note:** You can find some useful environment variables [here](https://docs.argilla.io/v2.0/reference/argilla-server/configuration/#environment-variables).

### Create dataset
You need to format your data (responses from different LLMs) as follows:
```python
[
    {
        "instance_id": 1,
        "lang": "es",
        "model_A": "xxx",
        "model_B": "xxx",
        "prompt": "¿Cuántas B hay en abbsqdjfbbcjd?",
        "answer_A": "En 'abbsqdjfbbcjd' hay cuatro letras 'b'.",
        "answer_B": "Hay 5 letras B."
    },
    {
        "instance_id": 2,
        "lang": "ca",
        "model_A": "yyy",
        "model_B": "xxx",
        "prompt": "Quin és el millor jugador de la història del futbol?",
        "answer_A": "Leo Messi.",
        "answer_B": "Cristiano Ronaldo."
    }
]
```

### Create guidelines

This should be plain text in a `.txt` or `.md` file (e.g. `./data/guidelines.md`).

**Note:** There's a toy dataset in `./data/toy_data.json`.

### Edit configuration file
The configuration files (under `./configs`) should be YAMLs with the following format:
```yaml
WORKSPACE_NAME: "argilla"
DATASET_NAME: "toy_data"
DATASET_PATH: "./data/toy_data.json"
GUIDELINES_PATH: "./data/guidelines.md"
NUM_USERS: 3
OUTPUT_DIR: "./output"
```

Each experiment (a.k.a annotation task) should have a separate config file.

### Upload dataset to HF Hub

```bash
python upload_data.py ./configs/experiment_1.yaml
```

### Annotate data

HuggingFace users can start annotating instances after logging in with their credentials.

### Collect results

```bash
python upload_dataset.py ./configs/experiment_1.yaml
```

This will create a new json file in your `$OUTPUT_DIR`.

## Resources
- [How-to guides (very useful!)](https://docs.argilla.io/latest/how_to_guides)
- [Quickstart Guide](https://docs.argilla.io/latest/getting_started/quickstart/#export-your-dataset-to-the-hub)
- [Create a Feedback Dataset](https://docs.v1.argilla.io/en/v1.10.0/guides/llms/practical_guides/create_dataset.html)
- [Blogpost: Launching Argilla on HF Spaces](https://argilla.io/blog/launching-argilla-huggingface-hub/)
- [HuggingFace Spaces](https://docs.v1.argilla.io/en/v1.19.0/getting_started/installation/deployments/huggingface-spaces.html)
- [HF Space Settings](https://docs.argilla.io/latest/getting_started/how-to-configure-argilla-on-huggingface/)
- [Python SDK Documentation](https://docs.argilla.io/v2.0/reference/argilla/client/)
- [Server Configuration Variables](https://docs.argilla.io/v2.0/reference/argilla-server/configuration/)
- [Dev Mode for Pro Accounts](https://huggingface.co/dev-mode-explorers)
- [Open-Source AI Cookbook](https://huggingface.co/learn/cookbook/en/enterprise_cookbook_argilla)
- [HF Documentation about Argilla](https://huggingface.co/docs/hub/en/datasets-argilla)
- [Official YouTube channel with tutorials](https://www.youtube.com/@argilla-io/videos)
- [Official dockers](https://hub.docker.com/u/argilla)

## TODO
- Creation of users (see [Assign records to annotation team](https://docs.v1.argilla.io/en/latest/tutorials/notebooks/labelling-tokenclassification-basics.html) and [Distribute annotation task among team](https://docs.argilla.io/latest/how_to_guides/distribution))
