import os
import utils
import dotenv
import argilla as rg

""" LOAD API CREDENTIALS """
dotenv.load_dotenv()

""" INIT CLIENT """
client = rg.Argilla(api_url = os.getenv('API_URL'), api_key = os.getenv('API_KEY'))

""" LOAD CONFIG """
config = utils.load_yaml(sys.argv[1])

""" LOAD DATA """
data_file = utils.load_json(config['DATASET_PATH'])

""" LOAD GUIDELINES """
guidelines_markdown = utils.load_txt(config['GUIDELINES_PATH'])

""" DATASET SETTINGS """
settings = rg.Settings(
    distribution=rg.TaskDistribution(min_submitted=config['NUM_USERS']),
    allow_extra_metadata=True,
    guidelines=guidelines_markdown,
    fields=[
            rg.TextField(
                name="prompt",
                title="Prompt",
                use_markdown=True,
                required=True,
                description="Field description"
            ),
            rg.TextField(
                name="answer_a",
                title="Answer A",
                use_markdown=True,
                required=True,
                description="Field description",
            ),
            rg.TextField(
                name="answer_b",
                title="Answer B",
                use_markdown=True,
                required=True,
                description="Field description",
            ),
            rg.TextField(
                name="guidelines",
                title="Guidelines",
                use_markdown=True,
                required=True,
                description="Field description"
            )
    ],
    questions=[
        rg.LabelQuestion(
            name="label",
            title="What is the best response given the prompt?",
            description="Select the one that applies.",
            required=True,
            labels={"answer_a": "Answer A", "answer_b": "Answer B", "both": "Both", "none": "None"}

        ),
        rg.RatingQuestion(
                name="rating",
                values=[1, 2, 3, 4, 5],
                title="How much better is the chosen response with respect to the rejected one?",
                description="1 = very unsatisfied, 5 = very satisfied",
                required=True,
        ),
        rg.TextQuestion(
            name="text",
            title="Copy and modify the response here if there is anything you would like to modify.",
            description="If there is anything you would modify in the response copy and edit the response in this field.",
            use_markdown=True,
            required=False
        )
    ],
)

""" CREATE WORKSPACE """
try:
    workspace = rg.Workspace(name=config['WORKSPACE_NAME'], client=client)
    workspace.create()
except rg._exceptions._api.ConflictError as e:
    print(f"⚠️  Workspace '{config['WORKSPACE_NAME']}' already exists. Skipping creation.")
except Exception as e:
    print(f"❌ Unexpected error creating workspace '{config['WORKSPACE_NAME']}': {str(e)}")

""" CREATE DATASET """
try:
    dataset = rg.Dataset(name=config['DATASET_NAME'], workspace=config['WORKSPACE_NAME'], settings=settings, client=client)
    dataset.create()
except rg._exceptions._api.ConflictError as e:
    print(f"⚠️  Dataset '{config['DATASET_NAME']}' already exists. Skipping creation.")
except Exception as e:
    print(f"❌ Unexpected error creating dataset '{config['DATASET_NAME']}': {str(e)}")

""" GENERATE RECORDS """
records = []
for item in data_file:
    record = rg.Record(
        id=item["instance_id"],
        fields={
            "prompt": item["prompt"],
            "answer_a": item["answer_A"],
            "answer_b": item["answer_B"],
            "guidelines": guidelines_markdown
        },
        metadata={
            "lang": item["lang"],
            "model_A": item["model_A"],
            "model_B": item["model_B"],
        },
    )
    records.append(record)

""" LOG RECORDS """
dataset.records.log(records)
