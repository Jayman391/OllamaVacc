import pandas as pd
import logging
from typing import List
import subprocess
import threading

## remove all escape sequences from prompt
## print out each request in json format with each request being on a single line
## containerize ollama
## update shell script with bennets patches

## make new version of pipeline.py that just generates the list of reuqests one per line and spits it to stdout
## start multiple ollamas in parallel and hit each port with batch of requests
    ## specify port numbers for each ollama

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class CustomThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, verbose=None):
        # Initializing the Thread class
        super().__init__(group, target, name, args, kwargs)
        self._return = None

    # Overriding the Thread.run function
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        super().join()
        return self._return

def load_data():
    """Load labeled and unlabeled data from CSV files."""
    try:
        labeled = pd.read_csv("data/labeled-data.csv")
        unlabeled = pd.read_csv("data/unlabeled-data.csv")
        logger.info("Data loaded successfully.")
        return labeled, unlabeled
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

def generate_few_shot(labeled: pd.DataFrame, unlabeled: pd.DataFrame) -> List[str]:
    """Generate few-shot prompts for text classification."""
    logger.info("Starting to generate few-shot prompts.")
    try:
        # Extract required columns
        text_with_labels = labeled[['text', 'labels']]

        text_without_labels = unlabeled['text']

        # Remove labels containing commas
        text_with_labels = text_with_labels[
            text_with_labels['labels'].str.contains(',') == False
        ]

        # Clean labels by removing unwanted characters
        text_with_labels['labels'] = (
            text_with_labels['labels']
            .str.replace(r'\[|\]|"', '', regex=True)
        )


        # Remove rows with "Not about vaccines" label
        text_with_labels = text_with_labels[
            text_with_labels['labels'] != 'Not about vaccines'
        ]

        unique_labels = text_with_labels['labels'].unique()

        # Group by labels
        grouped = text_with_labels.groupby('labels')

        prompts = []

        # Iterate through each text in the unlabeled dataset
        for text in text_without_labels:

            # clean text to remove unwanted characters such as \n and "
            text = text.replace("\n", " ").replace('"', '')
            # Randomly sample 4 examples from each group
            sample_docs = grouped.apply(
                lambda x: x.sample(min(len(x), 2))
            ).reset_index(drop=True)

            # Prepare examples for the prompt
            examples = " ".join(
                f"  {row['text'].replace("\n", " ").replace('"', '')} : {row['labels']}" 
                for _, row in sample_docs.iterrows()
            )

            # Construct the prompt
            prompt = f"""Your task is to classify the following text as one of the following categories: {', '.join(unique_labels)}.
Here are several examples of text and their corresponding labels:
{examples}
Return the most likely label for this document: <START> {text} <STOP>
Do not have the <START> or <STOP> tokens in the response.
Do not have anything but the label in the response."""
            


            prompts.append(prompt)

        logger.info(f"Generated {len(prompts)} prompts.")
        return prompts
    except Exception as e:
        logger.error(f"Error during prompt generation: {e}")
        raise

def generate_ollama_requests(input_prompts: List[str]) -> List[str]:
    """Generate JSON requests for the Ollama model."""
    logger.info("Generating Ollama requests.")
    requests = [
        {
            "model": "llama3.2", 
            "messages": [{"role": "user", "content": prompt}], 
            "stream": 'false'
        } for prompt in input_prompts
    ]

    return requests
    # need to 

def main():
    """Main function to orchestrate the pipeline with concurrent batch processing."""
    try:
        logger.info("Starting main pipeline.")
        labeled, unlabeled = load_data()
        labeled = labeled.sample(labeled.shape[0])

        batch_size = 100

        import math

        num_batches = math.ceil(len(unlabeled) / batch_size)

        for i in range(num_batches):
            unlabeled_batch = unlabeled.loc[0 + i * batch_size:batch_size + i * batch_size]

            thread = CustomThread(target = generate_few_shot, args=(labeled, unlabeled_batch))
            thread.start()

            few_shot_prompts = thread.join()

            ollama_requests = generate_ollama_requests(few_shot_prompts)
            # run_ollama_batch(ollama_requests)
            import json
            with open(f"requests.json", "a") as f:
                for request in ollama_requests:
                    # remove \\n and turn \u2019 into `
                    request = json.dumps(request).replace("\\n", "").replace("\\u2019", "`").replace('"false"', 'false')
                    # replace \\ with " "
                    request = request.replace("\\\\", " ")
                    # replace \u201c \u201d and \u2026
                    request = request.replace("\\u201c", '').replace("\\u201d", '').replace("\\u2026", "")
                    # remove \ 
                    request = request.replace("\\", "")
                    

                    f.write(request + "\n")

        logger.info("Pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
