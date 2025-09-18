"""
Prompt optimization script using DSPy GEPA for rephrasing and summarizing low-quality web data.

This script optimizes prompts for rephrasing and summarizing web text data to improve educational quality,
using the fineweb edu score classifier as the evaluation metric and GEPA optimizer.
After optimization, it evaluates the optimized prompt on a separate set of new FineWeb samples
and reports edu scores and their differences.

Usage:
    optimize-prompt --generation-model deepseek/deepseek-chat --reflection-model deepseek/deepseek-chat --train-size 50 --val-size 10 --test-size 1000 --seed 123 --log-dir ./logs
    optimize-prompt --generation-model openrouter/meta-llama/llama-3.1-8b-instruct --reflection-model deepseek/deepseek-chat --train-size 100 --val-size 20 --test-size 500
    optimize-prompt --generation-model huggingface/google/gemma-3-27b-it --reflection-model deepseek/deepseek-chat --train-size 100 --val-size 20 --test-size 1000
"""

import random
import dspy
import json
import litellm
from datasets import load_dataset
from typing import Optional, Any
import os
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv
from statistics import mean
from tqdm import tqdm

from utils import LOG_BASE_PATH, calculate_edu_score

from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline

# Load environment variables from .env file
load_dotenv()

# Ensure LiteLLM does not forward unsupported params (e.g., max_retries) to providers
litellm.drop_params = True
litellm.set_verbose = False

class Rephraser(dspy.Signature):
    """Rephrase low-quality web text into higher quality, more educational content."""
    
    original_text = dspy.InputField(desc="The original low-quality web text to rephrase")
    generated_text = dspy.OutputField(desc="The rephrased, higher-quality version of the text")

class Summarizer(dspy.Signature):
    """Summarize the input text into a concise, clear summary that preserves key points."""

    original_text = dspy.InputField(desc="The original text to summarize")
    generated_text = dspy.OutputField(desc="A concise summary of the original text")


class DspyGepaOptimizer(PipelineStep):
    """Pipeline step for DSPy GEPA optimization of prompts for text rephrasing/summarization."""
    
    def __init__(
        self,
        generation_model: str = "huggingface/google/gemma-3-27b-it",
        reflection_model: str = "huggingface/deepseek-ai/DeepSeek-V3.1",
        train_size: int = 50,
        val_size: int = 20,
        test_size: int = 1000,
        seed: int = 42,
        budget: int = 5,
        task: str = "rephrase",
        name: str = "",
        log_dir: str = f"{LOG_BASE_PATH}/prompt_optimization"
    ):
        super().__init__()
        self.generation_model = generation_model
        self.reflection_model = reflection_model
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed
        self.budget = budget
        self.task = task
        self.name = name
        self.log_dir = log_dir
        
        # Create hierarchical directory structure: task/model/train-val-size/budget
        generation_model_name = self.generation_model.split("/")[-1]  # Extract last part after /
        train_val_size = f"train-{self.train_size}-val-{self.val_size}"
        run_name = f"budget-{self.budget}-{self.name}"
        
        self.run_dir = f"{self.log_dir}/{self.task}/{generation_model_name}/{train_val_size}/{run_name}"
        Path(self.run_dir).mkdir(parents=True, exist_ok=True)
        
        # Parse model strings and create models
        generation_provider, generation_model_name = self.parse_model_string(self.generation_model)
        reflection_provider, reflection_model_name = self.parse_model_string(self.reflection_model)
        
        # Create models
        self.lm = self.configure_model_provider(
            generation_provider, 
            generation_model_name, 
            max_tokens=4096,  # we only train on 4096 tokens
            temperature=1.0,
            top_p=0.95,
            top_k=64,
        )
        
        self.reflection_lm = self.configure_model_provider(
            reflection_provider, 
            reflection_model_name, 
            max_tokens=16384, 
            temperature=0.6, 
            top_p=0.95, 
            top_k=64
        )

        # Set signature class
        self.signature_cls = Rephraser if self.task == "rephrase" else Summarizer
        

    def calculate_edu_score(self, text: str) -> float:
        return calculate_edu_score(text, self.edu_tokenizer, self.edu_model)

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """
        Run DSPy GEPA optimization for prompt optimization.
        This pipeline step doesn't process input data but performs optimization and saves results.
        """
        with self.track_time():
            dspy.settings.configure(lm=self.lm)

            # Create slurm logs directory
            slurm_logs_path = Path(self.run_dir) / "slurm_logs"
            slurm_logs_path.mkdir(parents=True, exist_ok=True)

            # Prepare datasets
            self.trainset, self.valset, self.testset = self.prepare_datasets(self.train_size, self.val_size, self.test_size, self.seed)

            # Load edu classifier
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.edu_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceFW/fineweb-edu-classifier")
            self.edu_model = AutoModelForSequenceClassification.from_pretrained("HuggingFaceFW/fineweb-edu-classifier").eval()
        
            # Run optimization
            optimized_module = self.optimize_task_module()

            # Save results
            optimized_module.save(f"{self.run_dir}/optimized_module.pkl")
            self.save_lm_interaction_history()
            self.evaluate_optimized_module(optimized_module)
            
            # Track statistics
            self.stat_update("optimization_completed", value=1)
            self.stat_update("train_size", value=self.train_size)
            self.stat_update("val_size", value=self.val_size)
            self.stat_update("test_size", value=self.test_size)
            self.stat_update("budget", value=self.budget)
        
        # This pipeline step doesn't yield any documents as it's purely for optimization
        return
        yield  # This line is unreachable but satisfies the generator requirement

    def gepa_metric(self,
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
            original_score = self.calculate_edu_score(original_text)
            generated_score = self.calculate_edu_score(generated_text)
            
            # Return the improvement in educational score
            improvement = generated_score - original_score
            
            # Log feedback for debugging (since we can't return it in dict format)
            logging.debug(f"Evaluation: Original={original_score:.2f}, Output={generated_score:.2f}, Improvement={improvement:.2f}")
            return improvement
        except Exception as e:
            logging.error(f"Error in metric evaluation: {e}")
            return 0.0


    def load_fineweb_samples(self, n_samples: int = 100) -> list:
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


    def prepare_datasets(self, train_size: int, val_size: int, test_size: int, seed: int = 42) -> tuple[list, list, list]:
        """
        Prepare non-overlapping training, validation, and test datasets from FineWeb.
        
        Args:
            train_size: Number of training examples to prepare
            val_size: Number of validation examples to prepare
            test_size: Number of test examples to prepare
            seed: Random seed for reproducibility
        
        Returns:
            Tuple of (trainset, valset, testset) with non-overlapping examples
        """
        # Set random seed for reproducibility
        random.seed(seed)

        total_samples_needed = train_size + val_size + test_size
        logging.info(f"Loading {total_samples_needed} total samples from FineWeb (train/val/test)...")
        all_samples = self.load_fineweb_samples(n_samples=total_samples_needed)

        # Ensure we have enough samples
        if len(all_samples) < total_samples_needed:
            logging.warning(f"Only {len(all_samples)} samples available, but {total_samples_needed} requested")
            logging.info("Adjusting sizes proportionally...")
            ratio = len(all_samples) / max(total_samples_needed, 1)
            adjusted_train_size = int(train_size * ratio)
            adjusted_val_size = int(val_size * ratio)
            adjusted_test_size = max(len(all_samples) - adjusted_train_size - adjusted_val_size, 0)
            logging.info(f"  New training size: {adjusted_train_size}")
            logging.info(f"  New validation size: {adjusted_val_size}")
            logging.info(f"  New test size: {adjusted_test_size}")
        else:
            adjusted_train_size = train_size
            adjusted_val_size = val_size
            adjusted_test_size = test_size

        # Split into non-overlapping train, validation, and test sets
        trainset = all_samples[:adjusted_train_size]
        valset = all_samples[adjusted_train_size:adjusted_train_size + adjusted_val_size]
        testset = all_samples[adjusted_train_size + adjusted_val_size:adjusted_train_size + adjusted_val_size + adjusted_test_size]

        logging.info("Dataset split completed:")
        logging.info(f"  Training examples: {len(trainset)}")
        logging.info(f"  Validation examples: {len(valset)}")
        logging.info(f"  Test examples: {len(testset)}")

        return trainset, valset, testset

    def optimize_task_module(self):
        """
        Optimize the task module using GEPA and return the optimized module.
        
        Returns:
            Optimized task module
        """
        # Initialize the student module
        student = dspy.Predict(self.signature_cls)
        
        # Configure GEPA optimizer
        gepa = dspy.GEPA(
            metric=self.gepa_metric,
            max_full_evals=self.budget,
            reflection_lm=self.reflection_lm,
            reflection_minibatch_size=1,
            candidate_selection_strategy="pareto",
            skip_perfect_score=True,
            track_stats=True,
            seed=self.seed
        )
        
        logging.info("Starting GEPA optimization...")
        
        # Compile the optimized module
        optimized_module = gepa.compile(
            student=student,
            trainset=self.trainset,
            valset=self.valset
        )
        
        logging.info("Optimization completed!")
        
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

    def parse_model_string(self, model_string: str) -> tuple[str, str]:
        """
        Parse model string in format {provider}/{model_name} and return provider and model name.
        
        Args:
            model_string: Model string in format "provider/model_name"
        
        Returns:
            Tuple of (provider, model_name)
        """
        if "/" not in model_string:
            raise ValueError(f"Model string must be in format 'provider/model_name', got: {model_string}")
        
        return model_string.split("/", 1)

    def configure_model_provider(self, provider: str, model_name: str, max_tokens: int = 4096, temperature: float = None, top_p: float = None, top_k: int = None) -> dspy.LM:
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
            # Use my own deployed inference endpoint at: https://endpoints.huggingface.co/huggingface/endpoints/dedicated
            # NVIDIA T4 errors, NVIDIA L4 and A10G work
            if model_name == "google/gemma-3-270m-it": # A10G, vllm
                api_base = "https://y7ftany7ifv3hpqh.us-east-1.aws.endpoints.huggingface.cloud/v1/"
            elif model_name == "google/gemma-3-1b-it": # L4, TGI
                api_base = "https://xk759aj0m12gfnz4.us-east-1.aws.endpoints.huggingface.cloud/v1/"
            elif model_name == "google/gemma-3-4b-it": # L4, TGI
                api_base = "https://raa4dhdnusvum3o5.us-east-1.aws.endpoints.huggingface.cloud/v1/"
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


    def save_lm_interaction_history(self) -> None:
        """
        Save the last language model interaction history and extract prompts.
        """
        try:
            if getattr(self.lm, "history", None):
                last_entry = self.lm.history[-1]
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
                with open(Path(self.run_dir) / "last_lm_interaction.json", "w", encoding="utf-8") as f:
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

                prompts_dir = Path(self.run_dir) / "prompts"
                prompts_dir.mkdir(parents=True, exist_ok=True)

                with open(prompts_dir / "system.md", "w", encoding="utf-8") as f:
                    f.write(system_content)
                # Compose user prompt to include the original_text placeholder and the trailing instruction
                user_prompt_template = "[[ ## original_text ## ]]\n[TEXT]\n\n" + user_extracted
                with open(prompts_dir / "user.md", "w", encoding="utf-8") as f:
                    f.write(user_prompt_template)
        except Exception as e:
            logging.warning(f"Failed to save last LM interaction: {e}", exc_info=True)


    def evaluate_optimized_module(self, optimized_module: Any) -> dict:
        """
        Evaluate the optimized module on the test set and compute statistics.
        
        Args:
            optimized_module: The optimized DSPy module to evaluate
        
        Returns:
            Dictionary containing evaluation statistics
        """
        logging.info("Evaluating optimized module on the held-out test set...")
        
        n_test = len(self.testset)
        if n_test == 0:
            logging.warning("Test set is empty; skipping evaluation and stats computation.")
            stats = {
                "n": 0,
                "mean_original_edu_score": 0.0,
                "mean_generated_edu_score": 0.0,
                "mean_edu_score_improvement": 0.0,
            }
        else:
            example_length = 2000
            # Show a single example
            example = self.testset[0]
            original_score = self.calculate_edu_score(example.original_text)
            original_preview = example.original_text[:example_length] + "..." if len(example.original_text) > example_length else example.original_text
            logging.info("--- Test Example 1 ---")
            logging.info(f"Original (edu score: {original_score:.2f}):")
            logging.info(f"Original text: {original_preview}")

            result = optimized_module(original_text=example.original_text)
            output_text = result.generated_text
            output_score = self.calculate_edu_score(output_text)
            output_preview = output_text[:example_length] + "..." if len(output_text) > example_length else output_text
            logging.info(f"Output (edu score: {output_score:.2f}):")
            logging.info(f"Output text: {output_preview}")
            logging.info(f"Improvement: {output_score - original_score:.2f}")

            # Compute statistics across full test set
            original_scores = []
            generated_scores = []
            improvements = []
            
            for ex in tqdm(self.testset, desc="Evaluating test examples"):
                orig = self.calculate_edu_score(ex.original_text)
                gen_text = optimized_module(original_text=ex.original_text).generated_text
                gen = self.calculate_edu_score(gen_text)
                original_scores.append(orig)
                generated_scores.append(gen)
                improvements.append(gen - orig)

            stats = {
                "n": n_test,
                "mean_original_edu_score": float(mean(original_scores)) if original_scores else 0.0,
                "mean_generated_edu_score": float(mean(generated_scores)) if generated_scores else 0.0,
                "mean_edu_score_improvement": float(mean(improvements)) if improvements else 0.0,
            }

        stats_path = Path(self.run_dir) / "edu_score_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved edu score statistics to {stats_path}")
        
        return stats


parser = argparse.ArgumentParser(description="Optimize prompts for web-text tasks (rephrase, summarize) using DSPy GEPA")
parser.add_argument(
    "--generation-model",
    type=str,
    default="huggingface/google/gemma-3-27b-it",
    help="Model name for generation in format {provider}/{model_name} (default: huggingface/google/gemma-3-27b-it)"
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
    default=500,
    help="Number of training examples (default: 50)"
)
parser.add_argument(
    "--val-size",
    type=int,
    default=100,
    help="Number of validation examples (default: 20)"
)
parser.add_argument(
    "--test-size",
    type=int,
    default=100,
    help="Number of new FineWeb samples for post-optimization evaluation (default: 1000)"
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility (default: 42)"
)
parser.add_argument(
    "--name",
    type=str,
    default="",
    help="Name of the optimization run",
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
    "--run-local",
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
    "--qos",
    type=str,
    default="normal",
    help="Slurm QoS (default: normal)",
)
parser.add_argument(
    "--dep-job-id",
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
    help="Enable debug mode: set train-size=2, val-size=2, budget=1, and run_local=true (default: False)",
)

def main():
    args = parser.parse_args()

    # Debug mode overrides for quick local runs
    if args.debug:
        args.train_size = 2
        args.val_size = 2
        args.test_size = 2
        args.budget = 1
        args.run_local = True

    # Create the DspyGepaOptimizer pipeline step
    optimizer_step = DspyGepaOptimizer(
        generation_model=args.generation_model,
        reflection_model=args.reflection_model,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        budget=args.budget,
        task=args.task,
        name=args.name,
        log_dir=args.log_dir
    )

    from datatrove.executor.local import LocalPipelineExecutor
    from datatrove.executor.slurm import SlurmPipelineExecutor

    if args.run_local:
        executor = LocalPipelineExecutor(pipeline=[optimizer_step], logging_dir=optimizer_step.run_dir)
    else:
        # Create job name using hierarchical structure: task-model-train-val-budget
        generation_model_name = args.generation_model.split("/")[-1]
        job_name = f"optimize-{args.task}-{generation_model_name}-train-{args.train_size}-val-{args.val_size}-budget-{args.budget}-{args.name}"
        executor = SlurmPipelineExecutor(
            pipeline=[optimizer_step],
            tasks=1,
            time=args.time,
            partition="hopper-prod",
            cpus_per_task=10,
            mem_per_cpu_gb=4,
            qos=args.qos,
            logging_dir=optimizer_step.run_dir,
            job_name=job_name,
            env_command="sleep $((RANDOM % 30))",
            depends_job_id=args.dep_job_id,
            sbatch_args={"gres": f"gpu:1"},
            srun_args={"cpu-bind": "none"},
        )

    executor.run()


if __name__ == "__main__":
    main()
