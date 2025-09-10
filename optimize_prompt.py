"""
Prompt optimization script using DSPy GEPA for rephrasing low-quality web data.

This script optimizes prompts for rephrasing web text data to improve educational quality,
using the fineweb edu score classifier as the evaluation metric and GEPA optimizer.

Usage:
    optimize-prompt --rephrasing-model sglang/Qwen/Qwen3-0.6B-FP8 --reflection-model deepseek/deepseek-chat --port 8000 --train-size 20 --val-size 5
    optimize-prompt --rephrasing-model deepseek/deepseek-chat --reflection-model deepseek/deepseek-chat --train-size 50 --val-size 10 --seed 123 --log-dir ./logs
    optimize-prompt --rephrasing-model vllm/meta-llama/Llama-2-7b-hf --reflection-model openrouter/meta-llama/llama-3.1-8b-instruct --train-size 100 --val-size 20
    optimize-prompt --rephrasing-model sglang/meta-llama/Llama-2-7b-hf --reflection-model deepseek/deepseek-chat --train-size 100 --val-size 20
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

from abc import ABC, abstractmethod

class ServerManager(ABC):
    """Abstract base class for server lifecycle management."""
    
    def __init__(self, model_name: str, port: int = 8000):
        self.model_name = model_name
        self.port = port
        self.process = None
    
    def __enter__(self):
        """Start server when entering context."""
        logging.info(f"Starting {self.__class__.__name__} for {self.model_name} on port {self.port}...")
        
        cmd = self._build_command()
        logging.info(f"Running command: {' '.join(cmd)}")
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info(f"Server process started with PID: {self.process.pid}")
        
        # Wait for server to start
        max_retries = 60  # Increased wait time for model loading
        for i in range(max_retries):
            # Check if process is still running
            if self.process.poll() is not None:
                # Process has terminated, get the output
                stdout, stderr = self.process.communicate()
                logging.error(f"Server process terminated unexpectedly!")
                logging.error(f"STDOUT: {stdout.decode('utf-8')}")
                logging.error(f"STDERR: {stderr.decode('utf-8')}")
                raise RuntimeError(f"Server process terminated with code {self.process.returncode}")
            
            if self._check_server_ready():
                logging.info(f"{self.__class__.__name__} started successfully on port {self.port}")
                return self
            
            time.sleep(5)  # Increased wait time between retries
            logging.info(f"Waiting for {self.__class__.__name__} to start... ({i+1}/{max_retries})")
        
        # If we get here, the server failed to start
        self._cleanup()
        raise RuntimeError(f"Failed to start {self.__class__.__name__}")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up server when exiting context."""
        self._cleanup()
    
    @abstractmethod
    def _build_command(self) -> list:
        """Build the command to start the server."""
        pass
    
    @abstractmethod
    def _check_server_ready(self) -> bool:
        """Check if the server is ready to accept requests."""
        pass
    
    def _cleanup(self):
        """Terminate the server process."""
        if self.process:
            logging.info(f"Shutting down {self.__class__.__name__}...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logging.warning(f"Force killing {self.__class__.__name__}...")
                self.process.kill()
                self.process.wait()
            self.process = None

class VLLMServerManager(ServerManager):
    """Context manager for VLLM server lifecycle management."""
    
    def _build_command(self) -> list:
        """Build the VLLM server command."""
        return [
            "vllm", "serve", self.model_name,
            "--port", str(self.port),
            "--host", "127.0.0.1"
        ]
    
    def _check_server_ready(self) -> bool:
        """Check if the VLLM server is ready."""
        try:
            # Try both health endpoints that VLLM might use
            for endpoint in ["/health", "/v1/models"]:
                try:
                    response = requests.get(f"http://127.0.0.1:{self.port}{endpoint}")
                    if response.status_code == 200:
                        return True
                except requests.exceptions.ConnectionError:
                    continue
            return False
        except Exception:
            return False

class SGLangServerManager(ServerManager):
    """Context manager for SGLang server lifecycle management."""
    
    def _build_command(self) -> list:
        """Build the SGLang server command."""
        return [
            "python3", "-m", "sglang.launch_server",
            "--model-path", self.model_name,
            "--allow-auto-truncate",
            "--context-length", "8192",  # Default context length
            "--port", str(self.port),
            "--log-level-http", "warning"
        ]
    
    def _check_server_ready(self) -> bool:
        """Check if the SGLang server is ready."""
        try:
            # SGLang typically uses the /v1/models endpoint
            response = requests.get(f"http://127.0.0.1:{self.port}/v1/models")
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False
        except Exception:
            return False

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

def configure_model_provider(provider: str, model_name: str, max_tokens: int = 1024, temperature: float = 0.7, top_p: float = 0.8, port: int = 8000) -> dspy.LM:
    """
    Configure and create a DSPy model for the specified provider.
    
    Args:
        provider: Provider name (vllm, sglang, openrouter, deepseek, huggingface)
        model_name: Model name
        max_tokens: Maximum tokens for the model
        temperature: Temperature for the model
        port: Port for local servers (VLLM/SGLang)
    
    Returns:
        Configured DSPy LM instance
    """
    if provider in ["vllm", "sglang"]:
        # Configure DSPy with local VLLM/SGLang server (explicit OpenAI provider)
        api_base = f'http://127.0.0.1:{port}/v1'
        dspy_model_name = f"openai/{model_name}"
        api_key = os.getenv("OPENAI_API_KEY", "dummy-key")  # VLLM doesn't need a real key
    elif provider == "openrouter":
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
    
    return dspy.LM(
        model=dspy_model_name,
        api_key=api_key,
        api_base=api_base,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=20,
        presence_penalty=1.5 if "Qwen3" in model_name else None,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}} if "Qwen3" in model_name else {},
        default_headers={"X-HF-Bill-To": "huggingface"} if provider == "huggingface" else None,
    )

def get_server_manager(provider: str, model_name: str, port: int) -> ServerManager:
    """
    Get the appropriate server manager for the given provider.
    
    Args:
        provider: Provider name (vllm, sglang)
        model_name: Model name
        port: Port for the server
    
    Returns:
        Server manager instance
    """
    if provider == "vllm":
        return VLLMServerManager(model_name, port)
    elif provider == "sglang":
        return SGLangServerManager(model_name, port)
    else:
        raise ValueError(f"No server manager available for provider: {provider}")

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
    default="huggingface/Qwen/Qwen3-0.6B",
    help="Model name for rephrasing in format {provider}/{model_name} (default: vllm/Qwen/Qwen3-0.6B-FP8)"
)
parser.add_argument(
    "--reflection-model",
    type=str,
    default="deepseek/deepseek-chat",
    help="Model name for reflection in format {provider}/{model_name} (default: deepseek/deepseek-chat). Currently no vllm/sglang models support reflection."
)
parser.add_argument(
    "--port",
    type=int,
    default=8000,
    help="Port for VLLM server (default: 8000)"
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
    default=LOG_BASE_PATH,
    help=f"Directory to save log files (default: {LOG_BASE_PATH})"
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
    logging.info(f"  Port: {args.port}")
    logging.info(f"  Training examples: {args.train_size}")
    logging.info(f"  Validation examples: {args.val_size}")
    logging.info(f"  Random seed: {args.seed}")
    logging.info(f"  Log directory: {args.log_dir}")
    
    # Parse model strings
    rephrasing_provider, rephrasing_model_name = parse_model_string(args.rephrasing_model)
    reflection_provider, reflection_model_name = parse_model_string(args.reflection_model)
    
    # Create models directly
    lm = configure_model_provider(
        rephrasing_provider, 
        rephrasing_model_name, 
        max_tokens=8192, 
        temperature=0.7, 
        top_p=0.8,
        port=args.port
    )
    dspy.settings.configure(lm=lm)
    
    reflection_lm = configure_model_provider(
        reflection_provider, 
        reflection_model_name, 
        max_tokens=2048, 
        temperature=0.8,
        top_p=0.95,
    )

    # Prepare training and validation datasets
    trainset, valset = prepare_dataset(args.train_size, args.val_size, args.seed)

    if rephrasing_provider in ["vllm", "sglang"]:
        # Use context manager for local server (VLLM or SGLang)
        server_manager = get_server_manager(rephrasing_provider, rephrasing_model_name, args.port)
        with server_manager:
            # Test the connection with a simple call
            logging.info(f"Testing {rephrasing_provider} server connection...")
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
