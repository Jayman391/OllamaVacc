import pandas as pd
import logging
from typing import List
import subprocess
import threading


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

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

        # Group by labels
        grouped = text_with_labels.groupby('labels')

        prompts = []

        # Iterate through each text in the unlabeled dataset
        for text in text_without_labels:
            # Randomly sample 4 examples from each group
            sample_docs = grouped.apply(
                lambda x: x.sample(min(len(x), 4))
            ).reset_index(drop=True)

            # Prepare examples for the prompt
            examples = "\n".join(
                f"  {row['text']} : {row['labels']}" 
                for _, row in sample_docs.iterrows()
            )

            # Construct the prompt
            prompt = f"""Your task is to classify the following text as one of the following categories: {', '.join(sample_docs['labels'].unique())}
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
    return [
        str({
            "model": "llama3.2", 
            "messages": [{"role": "user", "content": prompt}], 
            "stream": False
        }) for prompt in input_prompts
    ]

def batch_ollama_requests(requests: List[str], batch_size: int = 8) -> List[List[str]]:
    """Batch the Ollama requests into groups."""
    logger.info(f"Batching requests with batch size {batch_size}.")
    return [requests[i:i + batch_size] for i in range(0, len(requests), batch_size)]

def run_ollama_batch(requests: List[str]):
    """Submit batched requests to Ollama via a shell command."""
    logger.info(f"Submitting a batch of size {len(requests)} to Ollama.")
    try:
        subprocess.run(["sbatch", "run_pipeline.sh", *requests], shell=False)
    except Exception as e:
        logger.error(f"Error running Ollama batch: {e}")
        raise

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

def main():
    """Main function to orchestrate the pipeline with concurrent batch processing."""
    try:
        logger.info("Starting main pipeline.")
        labeled, unlabeled = load_data()
        labeled = labeled.sample(labeled.shape[0])

        batch_size = 100

        num_batches = len(unlabeled) // batch_size

        for i in range(num_batches):
            labeled_batch = labeled.loc[0 + i * batch_size:batch_size + i * batch_size]
            unlabeled_batch = unlabeled.loc[0 + i * batch_size:batch_size + i * batch_size]

            thread = CustomThread(target = generate_few_shot, args=(labeled_batch, unlabeled_batch))
            thread.start()

            few_shot_prompts = thread.join()

            ollama_requests = generate_ollama_requests(few_shot_prompts)
            run_ollama_batch(ollama_requests)

        logger.info("Pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
