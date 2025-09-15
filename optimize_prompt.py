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
import json
import litellm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Optional, Any
import os
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

from utils import LOG_BASE_PATH

# Load environment variables from .env file
load_dotenv()

# Ensure LiteLLM does not forward unsupported params (e.g., max_retries) to providers
litellm.drop_params = True
litellm.set_verbose = False

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
    generated_text = dspy.OutputField(desc="The rephrased, higher-quality version of the text")

class Summarizer(dspy.Signature):
    """Summarize the input text into a concise, clear summary that preserves key points."""

    original_text = dspy.InputField(desc="The original text to summarize")
    generated_text = dspy.OutputField(desc="A concise summary of the original text")

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
        # Get the original and generated text
        original_text = gold.original_text
        generated_text = pred.generated_text
        
        # Calculate edu scores for both
        original_score = calculate_edu_score(original_text)
        generated_score = calculate_edu_score(generated_text)
        
        # Return the improvement in educational score
        improvement = generated_score - original_score
        
        # Log feedback for debugging (since we can't return it in dict format)
        logging.debug(f"Evaluation: Original={original_score:.2f}, Output={generated_score:.2f}, Improvement={improvement:.2f}")
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

def optimize_task_module(trainset: list, valset: list, reflection_lm: Any, seed: int = 42, budget: int = 5, signature_cls=Rephraser):
    """
    Optimize the rephraser module using GEPA.
    
    Args:
        trainset: Training examples for optimization
        valset: Validation examples for evaluation
        reflection_lm: Language model for reflection
        seed: Random seed for reproducibility
        budget: Budget for GEPA optimization (max_full_evals)
    
    Returns:
        Optimized task module
    """
    # Initialize the student module
    student = dspy.Predict(signature_cls)
    
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
        output_text = result.generated_text
        output_score = calculate_edu_score(output_text)
        
        output_preview = output_text[:example_length] + "..." if len(output_text) > example_length else output_text
        logging.info(f"Output (edu score: {output_score:.2f}):")
        logging.info(f"Output text: {output_preview}")
        
        improvement = output_score - original_score
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
        # Route via OpenAI adapter to avoid forwarding liteLLM-only params to HF router
        dspy_model_name = f"openai/{model_name}"
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
    
    # Create log filename with timestamp under slurm_logs (to match other modules)
    slurm_logs_path = log_path / "slurm_logs"
    slurm_logs_path.mkdir(parents=True, exist_ok=True)
    log_file = slurm_logs_path / f"optimize_prompt.log"
    
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



parser = argparse.ArgumentParser(description="Optimize prompts for web-text tasks (rephrase, summarize) using DSPy GEPA")
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
parser.add_argument(
    "--name",
    type=str,
    required=True,
    help="Run name; used as the per-run directory inside --log-dir",
)
parser.add_argument(
    "--run_local",
    action="store_true",
    help="Run locally instead of using Slurm",
)
parser.add_argument(
    "--time",
    type=str,
    default="20:00:00",
    help="Slurm time limit (default: 20:00:00)",
)
parser.add_argument(
    "--partition",
    type=str,
    default="hopper-cpu",
    help="Slurm partition (default: hopper-cpu)",
)
parser.add_argument(
    "--qos",
    type=str,
    default="normal",
    help="Slurm QoS (default: normal)",
)
parser.add_argument(
    "--dep_job_id",
    type=str,
    default=None,
    help="Optional Slurm dependency job id",
)
parser.add_argument(
    "--task",
    type=str,
    default="rephrase",
    choices=["rephrase", "summarize"],
    help="Task to optimize: rephrase or summarize (default: rephrase)",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Enable debug mode: set train-size=2, val-size=2, budget=1, and run_local=true",
)

def run_optimization(args) -> None:
    """Execute the prompt optimization end-to-end."""
    setup_logging(args.log_dir)
    
    logging.info("Starting prompt optimization with DSPy GEPA...")
    logging.info("Configuration:")
    logging.info(f"  Task: {args.task}")
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
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )
    dspy.settings.configure(lm=lm)
    
    reflection_lm = configure_model_provider(
        reflection_provider, 
        reflection_model_name, 
        max_tokens=16384, 
        temperature=0.6, 
        top_p=0.95, 
        top_k=64
    )

    # Test the rephrasing model connection
    logging.info("Testing models connection...")
    test_response = lm("Test message for rephrasing model", max_tokens=10)
    logging.info(f"Rephrasing model test successful: {test_response}")
    test_response = reflection_lm("Test message for reflection model", max_tokens=10)
    logging.info(f"Reflection model test successful: {test_response}")

    # Prepare training and validation datasets
    trainset, valset = prepare_dataset(args.train_size, args.val_size, args.seed)

    signature_cls = Rephraser if args.task == "rephrase" else Summarizer
    optimized_module = optimize_task_module(trainset, valset, reflection_lm, args.seed, args.budget, signature_cls=signature_cls)

    optimized_module.save(f"{args.log_dir}/optimized_module.pkl")

    # Save the last LM interaction
    try:
        if getattr(lm, "history", None):
            last_entry = lm.history[-1]
            logging.info(f"Last LM interaction: {last_entry}")
            # Keep only JSON-serializable fields
            record = {
                "timestamp": last_entry.get("timestamp"),
                "uuid": last_entry.get("uuid"),
                "model": last_entry.get("model"),
                "response_model": last_entry.get("response_model"),
                "model_type": last_entry.get("model_type"),
                "messages": last_entry.get("messages"),
                "outputs": last_entry.get("outputs"),
                "kwargs": last_entry.get("kwargs"),
            }
            with open(Path(args.log_dir) / "last_lm_interaction.json", "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

            # Write system.md and user.md
            messages = last_entry.get("messages") or []
            system_content = ""
            user_content = ""
            for m in messages:
                if m.get("role") == "system" and isinstance(m.get("content"), str):
                    system_content = m["content"]
                if m.get("role") == "user" and isinstance(m.get("content"), str):
                    user_content = m["content"]

            # Extract user prompt segment per requested structure: split by two newlines and take last part
            if user_content:
                parts = user_content.split("\n\n")
                user_extracted = parts[-1] if parts else user_content
            else:
                user_extracted = ""

            prompts_dir = Path(args.log_dir) / "prompts"
            prompts_dir.mkdir(parents=True, exist_ok=True)

            with open(prompts_dir / "system.md", "w", encoding="utf-8") as f:
                f.write(system_content)
            # Compose user prompt to include the original_text placeholder and the trailing instruction
            user_prompt_template = "[[ ## original_text ## ]]\n[TEXT]\n\n" + user_extracted
            with open(prompts_dir / "user.md", "w", encoding="utf-8") as f:
                f.write(user_prompt_template)
    except Exception as e:
        logging.warning(f"Failed to save last LM interaction: {e}", exc_info=True)


def main():
    """Main optimization entrypoint with a per-run directory; unified logging."""
    args = parser.parse_args()

    # Debug mode overrides for quick local runs
    if getattr(args, "debug", False):
        args.train_size = 2
        args.val_size = 2
        args.budget = 1
        args.run_local = True

    # Per-run directory inside prompt_optimization: use --name
    run_name = f"{args.task}-budget-{args.budget}-{args.name}"
    run_dir = f"{args.log_dir}/{run_name}"
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    # Ensure downstream uses the per-run directory
    args.log_dir = run_dir

    # Build a simple pipeline step that runs the optimization once
    def _run_step(_data, rank: int = 0, world_size: int = 1):
        run_optimization(args)
        return []

    from datatrove.executor.local import LocalPipelineExecutor
    from datatrove.executor.slurm import SlurmPipelineExecutor

    if args.run_local:
        executor = LocalPipelineExecutor(pipeline=[_run_step], logging_dir=run_dir)
    else:
        executor = SlurmPipelineExecutor(
            pipeline=[_run_step],
            tasks=1,
            time=args.time,
            partition=args.partition,
            cpus_per_task=20,
            mem_per_cpu_gb=4,
            qos=args.qos,
            logging_dir=run_dir,
            job_name=f"optimize-prompt-{run_name}",
            env_command="sleep $((RANDOM % 30))",
            depends_job_id=args.dep_job_id,
        )

    executor.run()


if __name__ == "__main__":
    main()
