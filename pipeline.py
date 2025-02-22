import pandas as pd
import logging
from typing import List
import subprocess
import threading
import json
import math

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class CustomThread(threading.Thread):
    """
    Custom thread class to retrieve return values from threaded functions.
    """
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        if kwargs is None:
            kwargs = {}
        super().__init__(group=group, target=target, name=name, args=args, kwargs=kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        super().join()
        return self._return

def load_data():
    """
    Load labeled and unlabeled data from CSV files.
    """
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

def generate_prompt(labeled: pd.DataFrame, unlabeled: pd.DataFrame, unique_labels=None) -> List[str]:
    """
    Generate few-shot prompts for text classification.

    For each unlabeled text, this function creates a prompt that:
      - Clearly instructs the LLM to act as an expert text classifier.
      - Provides a list of categories.
      - Shows several examples (sampled from labeled data) with text and corresponding labels.
      - Presents the target text within explicit start and end tokens.
      - Instructs the LLM to return only the predicted label.
    """
    if unique_labels is None:
        unique_labels = {}

    logger.info("Starting to generate few-shot prompts.")
    try:
        # Use only the 'text' and 'labels' columns from labeled data
        text_with_labels = labeled[['text', 'labels']].copy()

        # Filter out rows with commas in labels to avoid ambiguity
        text_with_labels = text_with_labels[
            text_with_labels['labels'].str.contains(',') == False
        ]

        # Clean labels by removing unwanted characters
        text_with_labels['labels'] = text_with_labels['labels'].str.replace(r'\[|\]|"', '', regex=True)

        # Exclude rows with the label "Not about vaccines"
        text_with_labels = text_with_labels[text_with_labels['labels'] != 'Not about vaccines']

        if not unique_labels:
            unique_labels = sorted(text_with_labels['labels'].unique())

        # Group labeled data by label for sampling examples
        grouped = text_with_labels.groupby('labels')

        prompts = []

        for text in unlabeled['text']:
            # Clean the unlabeled text by removing newlines and quotes
            clean_text = text.replace("\n", " ").replace('"', '')

            # For each label group, randomly sample up to 3 examples (or fewer if not available)
            sample_docs = grouped.apply(lambda x: x.sample(min(len(x), 3))).reset_index(drop=True)

            # Construct example string: each example is delineated by markers
            examples = " ".join(
                f"<ExampleStart> {row['text'].replace('\n', ' ').replace('\"', '')} : {row['labels']} <ExampleEnd>"
                for _, row in sample_docs.iterrows()
            )

            # Build the prompt with clear instructions and boundaries.
            prompt = (
                f"You are an expert text classifier. Your job is to look at the following labeld examples and then label an unlabeled document.\n"
                f"Available categories: {', '.join(unique_labels)}.\n\n"
                f"Here are some examples for guidance:\n{examples}\n\n"
                f"Now, classify the following text between the tokens <<<START>>> and <<<END>>>.\n"
                f"<<<START>>>\n{clean_text}\n<<<END>>>\n\n"
                f"Return only the single label (from the available categories) without any additional text or formatting."
            )
            prompts.append(prompt)

        logger.info(f"Generated {len(prompts)} prompts.")
        return prompts
    except Exception as e:
        logger.error(f"Error during prompt generation: {e}")
        raise

def generate_ollama_requests(input_prompts: List[str]) -> List[str]:
    """
    Generate JSON requests for the Ollama model.
    Each request is a JSON object specifying the model, user message content, and streaming setting.
    """
    logger.info("Generating Ollama requests.")
    requests = [
        {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "seed": 101,
                "temperature": 0
            }
        } for prompt in input_prompts
    ]
    return requests

def write_requests_to_file(requests: List[dict], filename: str = "requests.json"):
    """
    Write each JSON request to the specified file, one per line.
    This function also cleans up escape sequences for clarity.
    """
    logger.info(f"Writing {len(requests)} requests to {filename}.")
    with open(filename, "a") as f:
        for request in requests:
            # Dump JSON as a single line and perform cleanup on escape sequences.
            request_str = json.dumps(request)
            # Clean up escape sequences and unwanted characters
            request_str = (
                request_str.replace("\\n", " ")
                           .replace("\\u2019", "`")
                           .replace('"false"', 'false')
                           .replace("\\\\", " ")
                           .replace("\\u201c", "")
                           .replace("\\u201d", "")
                           .replace("\\u2026", "")
                           .replace("\\", "")
                           .replace("(", "")
                           .replace(")", "")
            )
            f.write(request_str + "\n")

def main():
    """
    Main pipeline:
      1. Load labeled and unlabeled data.
      2. Process the unlabeled data in batches.
      3. For each batch, generate few-shot prompts using a separate thread.
      4. Convert these prompts into Ollama JSON requests.
      5. Write each JSON request to a file (one per line).
      
    Notes:
      - Future improvements include containerizing Ollama, applying Bennett's patches to the shell script,
        and running multiple Ollama instances in parallel on specified ports.
    """
    try:
        logger.info("Starting main pipeline.")
        labeled, unlabeled = load_data()

        # Shuffle labeled data to randomize examples
        labeled = labeled.sample(frac=1)

        batch_size = 100
        num_batches = math.ceil(len(unlabeled) / batch_size)

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            unlabeled_batch = unlabeled.iloc[start_idx:end_idx]

            # Run few-shot prompt generation in a separate thread
            thread = CustomThread(target=generate_prompt, args=(labeled, unlabeled_batch))
            thread.start()
            few_shot_prompts = thread.join()

            # Generate Ollama requests from the prompts
            ollama_requests = generate_ollama_requests(few_shot_prompts)

            # Write the requests to file, one per line in JSON format
            write_requests_to_file(ollama_requests)

        logger.info("Pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
