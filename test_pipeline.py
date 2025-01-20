import pytest
import pandas as pd
import subprocess
from pipeline import (
    load_data,
    generate_few_shot,
    generate_ollama_requests,
    batch_ollama_requests,
    run_ollama_batch,
)

# Sample data for testing
@pytest.fixture
def sample_labeled_df():
    data = {
        "Input.text": ["Text A", "Text B", "Text C"],
        "Answer.category.labels": ["Category1", "Category2", "Not about vaccines"]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_unlabeled_df():
    data = {"text": ["Unlabeled text 1", "Unlabeled text 2"]}
    return pd.DataFrame(data)

def test_load_data_file_not_found(monkeypatch):
    # Simulate FileNotFoundError on pd.read_csv
    def fake_read_csv(*args, **kwargs):
        raise FileNotFoundError("File not found.")
    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    with pytest.raises(FileNotFoundError):
        load_data()

def test_load_data_success(monkeypatch, sample_labeled_df, sample_unlabeled_df):
    # Simulate successful read_csv calls
    calls = [sample_labeled_df, sample_unlabeled_df]
    def fake_read_csv(filepath, *args, **kwargs):
        # Return first call for first file, second call for second file
        return calls.pop(0)
    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    labeled, unlabeled = load_data()
    pd.testing.assert_frame_equal(labeled, sample_labeled_df)
    pd.testing.assert_frame_equal(unlabeled, sample_unlabeled_df)

def test_generate_few_shot(sample_labeled_df, sample_unlabeled_df):
    # Add additional rows to simulate comma-containing labels and unwanted label removal
    labeled = sample_labeled_df.copy()
    # Row with comma in label should be removed
    labeled = pd.concat([labeled, pd.DataFrame({
        "Input.text": ["Text D"],
        "Answer.category.labels": ["Category1, Category2"]
      })
    ])
    # Test for cleaning labels and filtering out "Not about vaccines"
    prompts = generate_few_shot(labeled, sample_unlabeled_df)

    # Check that prompts is a list and not empty
    assert isinstance(prompts, list)
    assert len(prompts) == len(sample_unlabeled_df)

    # Check that no prompt contains unwanted tokens for each unlabeled text
    for prompt in prompts:
        # Ensure 'Not about vaccines' not in prompt examples
        assert "Not about vaccines" not in prompt
        # Ensure cleaned labels (no brackets or quotes)
        assert "[" not in prompt and "]" not in prompt and '"' not in prompt

def test_generate_ollama_requests():
    sample_prompts = ["Prompt 1", "Prompt 2"]
    requests = generate_ollama_requests(sample_prompts)
    assert isinstance(requests, list)
    assert len(requests) == 2
    # Check that each request string contains expected keys
    for req in requests:
        assert any(prompt in req for prompt in sample_prompts)

def test_batch_ollama_requests():
    # Create a list of dummy requests
    dummy_requests = [f"request_{i}" for i in range(20)]
    batch_size = 8
    batches = batch_ollama_requests(dummy_requests, batch_size)

    # Check that batches are correct
    assert isinstance(batches, list)
    # Each batch except possibly last has size batch_size
    for batch in batches[:-1]:
        assert len(batch) == batch_size
    # Last batch should have remainder
    assert len(batches[-1]) == 20 % batch_size

def test_run_ollama_batch(monkeypatch):
    # Prepare a dummy list of requests
    dummy_requests = ["req1", "req2"]
    # Capture arguments passed to subprocess.run
    called_args = []

    def fake_run(args, shell):
        called_args.append(args)
    monkeypatch.setattr(subprocess, "run", fake_run)

    run_ollama_batch(dummy_requests)
    # Check that subprocess.run was called with expected arguments
    assert called_args, "subprocess.run was not called"
    # Check that the command starts with ["sbatch", "run_pipeline.sh", ...]
    assert called_args[0][:2] == ["sbatch", "run_pipeline.sh"]
    # Also check that our requests are included
    for req in dummy_requests:
        assert req in called_args[0]

