"""
Prompt optimization script using DSPy GEPA for rephrasing low-quality web data.

This script optimizes prompts for rephrasing web text data to improve educational quality,
using the fineweb edu score classifier as the evaluation metric and GEPA optimizer.

Usage:
    python optimize_prompt.py --model-name Qwen/Qwen3-0.6B-FP8 --port 8000 --train-size 20 --val-size 5 --provider deepseek
    python optimize_prompt.py -m Qwen/Qwen3-0.6B-FP8 -p 8001 -t 50 -v 10 --seed 123 --log-dir ./logs --provider openai
    python optimize_prompt.py --model-name meta-llama/Llama-2-7b-hf --train-size 100 --val-size 20 --provider vllm
"""

import random
import dspy
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Optional, Any
import subprocess
import time
import requests
import os
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

USER = os.environ.get('USER')
PROJECT_NAME = "finephrase"

BASE_PATH = f"/fsx/{USER}"
LOG_BASE_PATH = f"{BASE_PATH}/logs/{PROJECT_NAME}/experiments/prompt_optimization"

# Load fineweb edu classifier for scoring
edu_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")
edu_model = AutoModelForSequenceClassification.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")

def calculate_edu_score(text: str) -> float:
    """Calculate fineweb edu score for text."""
    if not text.strip():
        return 0.0
    
    try:
        inputs = edu_tokenizer(text, return_tensors="pt", padding="longest", truncation=True)
        outputs = edu_model(**inputs)
        logits = outputs.logits.squeeze(-1).float().detach().numpy()
        score = logits.item()
        return max(0.0, min(score, 5.0))  # Clamp between 0 and 5
    except Exception:
        return 0.0

# Define the rephrasing module
class Rephraser(dspy.Signature):
    """Rephrase low-quality web text into higher quality, more educational content."""
    
    original_text = dspy.InputField(desc="The original low-quality web text to rephrase")
    rephrased_text = dspy.OutputField(desc="The rephrased, higher-quality version of the text")

class RephraseModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.rephrase = dspy.ChainOfThought(Rephraser)
    
    def forward(self, original_text):
        result = self.rephrase(original_text=original_text)
        return dspy.Prediction(rephrased_text=result.rephrased_text)

# Define the evaluation metric for GEPA
def gepa_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: Optional[Any] = None,
    pred_name: Optional[str] = None,
    pred_trace: Optional[Any] = None,
) -> float:
    """
    Metric function for GEPA that evaluates the educational quality improvement.
    Returns a numeric score for compatibility with DSPy's evaluation system.
    """
    try:
        # Get the original and rephrased text
        original_text = gold.original_text
        rephrased_text = pred.rephrased_text
        
        # Calculate edu scores for both
        original_score = calculate_edu_score(original_text)
        rephrased_score = calculate_edu_score(rephrased_text)
        
        # Return the improvement in educational score (normalized to 0-1)
        # Add a small bonus for any improvement to encourage optimization
        improvement = rephrased_score - original_score
        if improvement > 0:
            score = min(1.0, 0.5 + (improvement / 5.0))  # Base 0.5 + improvement bonus
        else:
            score = max(0.0, 0.5 + (improvement / 5.0))  # Penalty for degradation
        
        # Log feedback for debugging (since we can't return it in dict format)
        logging.debug(f"Evaluation: Original={original_score:.2f}, Rephrased={rephrased_score:.2f}, Improvement={improvement:.2f}, Score={score:.2f}")
        return improvement
    except Exception as e:
        logging.error(f"Error in metric evaluation: {e}")
        return 0.0

class VLLMServerManager:
    """Context manager for VLLM server lifecycle management."""
    
    def __init__(self, model_name: str, port: int = 8000):
        self.model_name = model_name
        self.port = port
        self.process = None
    
    def __enter__(self):
        """Start VLLM server when entering context."""
        logging.info(f"Starting VLLM server for {self.model_name} on port {self.port}...")
        
        cmd = [
            "vllm", "serve", self.model_name,
            "--port", str(self.port),
            "--host", "127.0.0.1"
        ]
        
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        max_retries = 60  # Increased wait time for model loading
        for i in range(max_retries):
            try:
                # Try both health endpoints that VLLM might use
                for endpoint in ["/health", "/v1/models"]:
                    try:
                        response = requests.get(f"http://127.0.0.1:{self.port}{endpoint}")
                        if response.status_code == 200:
                            logging.info(f"VLLM server started successfully on port {self.port}")
                            return self
                    except requests.exceptions.ConnectionError:
                        continue
            except Exception:
                pass
            
            time.sleep(5)  # Increased wait time between retries
            logging.debug(f"Waiting for VLLM server to start... ({i+1}/{max_retries})")
        
        # If we get here, the server failed to start
        self._cleanup()
        raise RuntimeError("Failed to start VLLM server")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up VLLM server when exiting context."""
        self._cleanup()
    
    def _cleanup(self):
        """Terminate the VLLM server process."""
        if self.process:
            logging.info("Shutting down VLLM server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logging.warning("Force killing VLLM server...")
                self.process.kill()
                self.process.wait()
            self.process = None

def load_fineweb_samples(n_samples: int = 100) -> list:
    """Load random samples from FineWeb dataset."""
    logging.info(f"Loading {n_samples} samples from FineWeb dataset...")
    
    # Load a subset of the FineWeb dataset
    # Using the sample-10BT subset for faster loading
    dataset = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
    
    # Sample random examples
    samples = []
    dataset_iter = iter(dataset)
    
    # Skip a smaller random number of examples to get different starting points
    skip_count = random.randint(0, 1000)  # Reduced for faster loading
    for _ in range(skip_count):
        try:
            next(dataset_iter)
        except StopIteration:
            break
    
    # Collect samples
    for i, example in enumerate(dataset_iter):
        if len(samples) >= n_samples:
            break
        
        text = example.get('text', '').strip()
        if text and len(text) > 100:  # Only use non-empty texts with reasonable length
            samples.append(dspy.Example(original_text=text).with_inputs('original_text'))
        
        if i > n_samples * 10:  # Safety limit to avoid infinite loop
            break
    
    logging.info(f"Loaded {len(samples)} valid samples")
    return samples

def prepare_dataset(train_size: int, val_size: int, seed: int = 42) -> tuple[list, list]:
    """
    Prepare training and validation datasets from FineWeb.
    
    Args:
        train_size: Number of training examples to prepare
        val_size: Number of validation examples to prepare
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (trainset, valset) with non-overlapping examples
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Load training and validation examples from FineWeb (non-overlapping sets)
    total_samples_needed = train_size + val_size
    logging.info(f"Loading {total_samples_needed} total samples from FineWeb...")
    all_samples = load_fineweb_samples(n_samples=total_samples_needed)
    
    # Ensure we have enough samples
    if len(all_samples) < total_samples_needed:
        logging.warning(f"Only {len(all_samples)} samples available, but {total_samples_needed} requested")
        logging.info("Adjusting sizes proportionally...")
        ratio = len(all_samples) / total_samples_needed
        adjusted_train_size = int(train_size * ratio)
        adjusted_val_size = len(all_samples) - adjusted_train_size
        logging.info(f"  New training size: {adjusted_train_size}")
        logging.info(f"  New validation size: {adjusted_val_size}")
    else:
        adjusted_train_size = train_size
        adjusted_val_size = val_size
    
    # Split into non-overlapping train and validation sets
    trainset = all_samples[:adjusted_train_size]
    valset = all_samples[adjusted_train_size:adjusted_train_size + adjusted_val_size]
    
    logging.info("Dataset split completed:")
    logging.info(f"  Training examples: {len(trainset)}")
    logging.info(f"  Validation examples: {len(valset)}")
    logging.info(f"  No overlap: {len(set(id(x) for x in trainset) & set(id(x) for x in valset)) == 0}")
    
    return trainset, valset

def optimize_rephraser(trainset: list, valset: list, reflection_lm: Any, seed: int = 42):
    """
    Optimize the rephraser module using GEPA.
    
    Args:
        trainset: Training examples for optimization
        valset: Validation examples for evaluation
        reflection_lm: Language model for reflection
        seed: Random seed for reproducibility
    
    Returns:
        Optimized rephraser module
    """
    # Initialize the student module
    student = RephraseModule()
    
    # Configure GEPA optimizer
    gepa = dspy.GEPA(
        metric=gepa_metric,
        #auto="light",  # Light budget for quick experimentation
        max_full_evals=5,
        #max_metric_calls=10,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=1,
        candidate_selection_strategy="pareto",
        skip_perfect_score=True,
        track_stats=True,
        seed=seed
    )
    
    logging.info("Starting GEPA optimization...")
    
    # Compile the optimized module
    optimized_module = gepa.compile(
        student=student,
        trainset=trainset,
        valset=valset
    )
    
    logging.info("Optimization completed!")
    
    # Test the optimized module on a few examples
    logging.info("Testing optimized module:")
    test_examples = valset[:min(3, len(valset))]

    example_length = 2000
    
    for i, example in enumerate(test_examples):
        logging.info(f"--- Example {i+1} ---")
        original_score = calculate_edu_score(example.original_text)
        logging.info(f"Original (edu score: {original_score:.2f}):")
        original_preview = example.original_text[:example_length] + "..." if len(example.original_text) > example_length else example.original_text
        logging.info(f"Original text: {original_preview}")
        
        result = optimized_module(original_text=example.original_text)
        rephrased_score = calculate_edu_score(result.rephrased_text)
        
        rephrased_preview = result.rephrased_text[:example_length] + "..." if len(result.rephrased_text) > example_length else result.rephrased_text
        logging.info(f"Rephrased (edu score: {rephrased_score:.2f}):")
        logging.info(f"Rephrased text: {rephrased_preview}")
        
        improvement = rephrased_score - original_score
        logging.info(f"Improvement: {improvement:.2f}")
    
    # Print optimization statistics if available
    if hasattr(optimized_module, 'detailed_results'):
        results = optimized_module.detailed_results
        logging.info("Optimization Results:")
        logging.info(f"- Total candidates evaluated: {len(results.candidates)}")
        logging.info(f"- Best validation score: {max(results.val_aggregate_scores):.3f}")
        logging.info(f"- Total metric calls: {results.total_metric_calls}")
        logging.info(f"- Best candidate: {results.best_candidate}")

        logging.info(results)

    # Print optimization results
    logging.info(optimized_module)
    for name, pred in optimized_module.named_predictors():
        logging.info("================================")
        logging.info(f"Predictor: {name}")
        logging.info("================================")
        logging.info("Prompt:")
        logging.info(pred.signature.instructions)
        logging.info("*********************************")
    
    return optimized_module

def setup_logging(log_dir: str) -> None:
    """
    Set up logging configuration with file and console handlers.
    
    Args:
        log_dir: Directory to save log files
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"optimize_prompt_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also output to console
        ]
    )
    
    logging.info(f"Logging initialized. Log file: {log_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimize prompts for rephrasing low-quality web data using DSPy GEPA"
    )
    
    parser.add_argument(
        "--model-name", "-m",
        type=str,
        default="deepseek-chat",
        help="Model name for rephrasing (default: deepseek-chat)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="deepseek",
        choices=["deepseek", "openrouter", "vllm"],
        help="LLM provider to use (default: deepseek)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port for VLLM server (default: 8000)"
    )
    parser.add_argument(
        "--train-size", "-t",
        type=int,
        default=50,
        help="Number of training examples (default: 50)"
    )
    parser.add_argument(
        "--val-size", "-v",
        type=int,
        default=20,
        help="Number of validation examples (default: 20)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=LOG_BASE_PATH,
        help=f"Directory to save log files (default: {LOG_BASE_PATH})"
    )
    
    return parser.parse_args()

def main():
    """Main optimization function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_dir)
    
    logging.info("Starting prompt optimization with DSPy GEPA...")
    logging.info("Configuration:")
    logging.info(f"  Model: {args.model_name}")
    logging.info(f"  Provider: {args.provider}")
    logging.info(f"  Port: {args.port}")
    logging.info(f"  Training examples: {args.train_size}")
    logging.info(f"  Validation examples: {args.val_size}")
    logging.info(f"  Random seed: {args.seed}")
    logging.info(f"  Log directory: {args.log_dir}")
    
    # Configure provider-specific settings
    if args.provider == "vllm":
        # Configure DSPy with local VLLM server (explicit OpenAI provider)
        # TODO: Currently there are some issues with this: potentially a deadlock
        os.environ['OPENAI_API_BASE'] = f'http://127.0.0.1:{args.port}/v1'
        lm_model_name = f"openai/{args.model_name}"
    elif args.provider == "openrouter":
        os.environ['OPENAI_API_BASE'] = f'https://openrouter.ai/api/v1'
        lm_model_name = f"openrouter/{args.model_name}"
    elif args.provider == "deepseek":
        os.environ['OPENAI_API_BASE'] = f'https://api.deepseek.com/v1'
        lm_model_name = f"deepseek/{args.model_name}"
    else:
        raise ValueError(f"Unsupported provider: {args.provider}")
    
    logging.info(f"Using model: {lm_model_name}")
    
    lm = dspy.LM(model=lm_model_name, max_tokens=2048, temperature=0.7)
    dspy.settings.configure(lm=lm)
    
    # Configure reflection model (using the same server)
    reflection_lm = dspy.LM(model=lm_model_name, max_tokens=1024, temperature=0.8)

    # Prepare training and validation datasets
    trainset, valset = prepare_dataset(args.train_size, args.val_size, args.seed)

    if args.provider == "vllm":
        # Use context manager for VLLM server
        with VLLMServerManager(args.model_name, args.port):
            # Test the connection with a simple call
            logging.info("Testing VLLM server connection...")
            try:
                test_response = lm("Test message", max_tokens=10)
                logging.info(f"Server test successful: {test_response}")
            except Exception as e:
                logging.error(f"Server test failed: {e}")
                raise
            optimized_module = optimize_rephraser(trainset, valset, reflection_lm, args.seed)
    else:
        optimized_module = optimize_rephraser(trainset, valset, reflection_lm, args.seed)

    optimized_module.save(f"{LOG_BASE_PATH}/optimized_module_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")


if __name__ == "__main__":
    main()
