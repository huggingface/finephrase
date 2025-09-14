"""
Prompt optimization script using DSPy GEPA for rephrasing low-quality web data.

This script optimizes prompts for rephrasing web text data to improve educational quality,
using the fineweb edu score classifier as the evaluation metric and GEPA optimizer.

Usage:
    optimize-prompt --rephrasing-model deepseek/deepseek-chat --reflection-model deepseek/deepseek-chat --train-size 50 --val-size 10 --seed 123 --log-dir ./logs
    optimize-prompt --rephrasing-model openrouter/meta-llama/llama-3.1-8b-instruct --reflection-model deepseek/deepseek-chat --train-size 100 --val-size 20
    optimize-prompt --rephrasing-model huggingface/google/gemma-3-27b-it --reflection-model deepseek/deepseek-chat --train-size 100 --val-size 20
"""

import random
import dspy
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Optional, Any
import os
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

from utils import LOG_BASE_PATH

# Load environment variables from .env file
load_dotenv()

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

def optimize_rephraser(trainset: list, valset: list, reflection_lm: Any, seed: int = 42, budget: int = 5):
    """
    Optimize the rephraser module using GEPA.
    
    Args:
        trainset: Training examples for optimization
        valset: Validation examples for evaluation
        reflection_lm: Language model for reflection
        seed: Random seed for reproducibility
        budget: Budget for GEPA optimization (max_full_evals)
    
    Returns:
        Optimized rephraser module
    """
    # Initialize the student module
    student = RephraseModule()
    
    # Configure GEPA optimizer
    gepa = dspy.GEPA(
        metric=gepa_metric,
        max_full_evals=budget,
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

def parse_model_string(model_string: str) -> tuple[str, str]:
    """
    Parse model string in format {provider}/{model_name} and return provider and model name.
    
    Args:
        model_string: Model string in format "provider/model_name"
    
    Returns:
        Tuple of (provider, model_name)
    """
    if "/" not in model_string:
        raise ValueError(f"Model string must be in format 'provider/model_name', got: {model_string}")
    
    parts = model_string.split("/", 1)  # Split only on the first '/'
    provider = parts[0]
    model_name = parts[1]
    
    return provider, model_name

def configure_model_provider(provider: str, model_name: str, max_tokens: int = 4096, temperature: float = None, top_p: float = None, top_k: int = None) -> dspy.LM:
    """
    Configure and create a DSPy model for the specified provider.
    
    Args:
        provider: Provider name (openrouter, deepseek, huggingface)
        model_name: Model name
        max_tokens: Maximum tokens for the model
        temperature: Temperature for the model (optional)
        top_p: Top-p for the model (optional)
        top_k: Top-k for the model (optional)
    
    Returns:
        Configured DSPy LM instance
    """
    if provider == "openrouter":
        api_base = f'https://openrouter.ai/api/v1'
        dspy_model_name = f"openrouter/{model_name}"
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required for OpenRouter provider")
    elif provider == "deepseek":
        api_base = f'https://api.deepseek.com/v1'
        dspy_model_name = f"deepseek/{model_name}"
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is required for DeepSeek provider")
    elif provider == "huggingface":
        api_base = f'https://router.huggingface.co/v1'
        dspy_model_name = f"huggingface/{model_name}"
        api_key = os.getenv("HF_TOKEN")
        if not api_key:
            raise ValueError("HF_TOKEN environment variable is required for HuggingFace provider")
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    # Build kwargs for dspy.LM, only including optional parameters if they are set
    lm_kwargs = {
        "model": dspy_model_name,
        "api_key": api_key,
        "api_base": api_base,
        "max_tokens": max_tokens,
        "default_headers": {"X-HF-Bill-To": "huggingface"} if provider == "huggingface" else None,
    }
    
    if temperature is not None:
        lm_kwargs["temperature"] = temperature
    if top_p is not None:
        lm_kwargs["top_p"] = top_p
    if top_k is not None:
        lm_kwargs["top_k"] = top_k
    
    return dspy.LM(**lm_kwargs)


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



parser = argparse.ArgumentParser(description="Optimize prompts for rephrasing low-quality web data using DSPy GEPA")
parser.add_argument(
    "--rephrasing-model",
    type=str,
    default="huggingface/google/gemma-3-27b-it",
    help="Model name for rephrasing in format {provider}/{model_name} (default: huggingface/google/gemma-3-27b-it)"
)
parser.add_argument(
    "--reflection-model",
    type=str,
    default="huggingface/deepseek-ai/DeepSeek-V3.1",
    help="Model name for reflection in format {provider}/{model_name} (default: huggingface/deepseek-ai/DeepSeek-V3.1)."
)
parser.add_argument(
    "--train-size",
    type=int,
    default=50,
    help="Number of training examples (default: 50)"
)
parser.add_argument(
    "--val-size",
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
    default=f"{LOG_BASE_PATH}/prompt_optimization",
    help=f"Directory to save log files (default: {LOG_BASE_PATH}/prompt_optimization)"
)
parser.add_argument(
    "--budget",
    type=int,
    default=5,
    help="Budget for GEPA optimization (max_full_evals, default: 5)"
)

def main():
    """Main optimization function."""
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_dir)
    
    logging.info("Starting prompt optimization with DSPy GEPA...")
    logging.info("Configuration:")
    logging.info(f"  Rephrasing model: {args.rephrasing_model}")
    logging.info(f"  Reflection model: {args.reflection_model}")
    logging.info(f"  Training examples: {args.train_size}")
    logging.info(f"  Validation examples: {args.val_size}")
    logging.info(f"  Random seed: {args.seed}")
    logging.info(f"  Budget (max_full_evals): {args.budget}")
    logging.info(f"  Log directory: {args.log_dir}")
    
    # Parse model strings
    rephrasing_provider, rephrasing_model_name = parse_model_string(args.rephrasing_model)
    reflection_provider, reflection_model_name = parse_model_string(args.reflection_model)
    
    # Create models directly
    lm = configure_model_provider(
        rephrasing_provider, 
        rephrasing_model_name, 
        max_tokens=4096, # we only train on 4096 tokens
    )
    dspy.settings.configure(lm=lm)
    
    reflection_lm = configure_model_provider(
        reflection_provider, 
        reflection_model_name, 
        max_tokens=8192, 
    )

    # Test the rephrasing model connection
    logging.info("Testing models connection...")
    test_response = lm("Test message for rephrasing model", max_tokens=10)
    logging.info(f"Rephrasing model test successful: {test_response}")
    test_response = reflection_lm("Test message for reflection model", max_tokens=10)
    logging.info(f"Reflection model test successful: {test_response}")


    # Prepare training and validation datasets
    trainset, valset = prepare_dataset(args.train_size, args.val_size, args.seed)

    optimized_module = optimize_rephraser(trainset, valset, reflection_lm, args.seed, args.budget)

    optimized_module.save(f"{args.log_dir}/optimized_module_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")


if __name__ == "__main__":
    main()
