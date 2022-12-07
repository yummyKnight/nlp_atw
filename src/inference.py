import os.path

from transformers import pipeline
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)


def inference():
    # Loading the pipeline from hub
    # Pipeline handles the preprocessing and post processing steps
    model_checkpoint = os.path.join(root, "bert-base-cased/checkpoint-1000")
    namedEntityRecogniser = pipeline(
        "token-classification", model=model_checkpoint, aggregation_strategy="simple"
    )

    text = "Mark Zuckerberg is one of the founders of Facebook, a company from the United States"
    sample_output = namedEntityRecogniser([text])
    print(sample_output)




if __name__ == '__main__':
    inference()












