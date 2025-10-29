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
from tqdm import tqdm
import pandas as pd

from utils import ENV_COMMAND, LOG_BASE_PATH
from quality_scores import calculate_edu_score, calculate_dclm_score

SCORE_WEIGHTS = {
    "edu_improvement": 0.40,
    "dclm_improvement": 0.40,
    "length_similarity": 0.15,
    "prompt_efficiency": 0.05,
}

from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline

load_dotenv() # Load environment variables from .env file

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
        
        # Parse model strings and store provider/model info; defer actual LM creation to run() to speed submission
        self.generation_provider, self.generation_model_name = self.parse_model_string(self.generation_model)
        self.reflection_provider, self.reflection_model_name = self.parse_model_string(self.reflection_model)
        self.lm = None
        self.reflection_lm = None

        # Set signature class
        self.signature_cls = Rephraser if self.task == "rephrase" else Summarizer
        

    def calculate_edu_score(self, text: str) -> float:
        return calculate_edu_score(text)

    def _ensure_models(self) -> None:
        if self.lm is None:
            self.lm = self.configure_model_provider(
                self.generation_provider,
                self.generation_model_name,
                max_tokens=4096, # we only train on 4096 tokens
                temperature=1.0,
                top_p=0.95,
                top_k=64,
            )
        if self.reflection_lm is None:
            self.reflection_lm = self.configure_model_provider(
                self.reflection_provider,
                self.reflection_model_name,
                max_tokens=16384,
                temperature=0.6,
                top_p=0.95,
                top_k=64,
            )

    # Make pickling/lightweight submission faster by dropping heavy runtime-only attributes
    def __getstate__(self):
        state = self.__dict__.copy()
        for attr in ("lm", "reflection_lm", "edu_tokenizer", "edu_model"):
            if attr in state:
                state[attr] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def calculate_prompt_efficiency_score(self, prompt_tokens: int) -> float:
        max_reasonable_prompt_tokens = 500  # Keep existing behavior; can be tuned
        return max(0.0, 1.0 - (prompt_tokens / max_reasonable_prompt_tokens))

    def compute_length_similarity(self, original_text: str, generated_text: str) -> float:
        input_token_count = len(original_text.split())
        output_token_count = len(generated_text.split())
        token_diff = abs(input_token_count - output_token_count)
        return max(0.0, 1.0 - (token_diff / max(input_token_count, 1)))

    def compute_edu_improvement(self, original_text: str, generated_text: str) -> tuple[float, float, float, float]:
        orig = self.calculate_edu_score(original_text)
        gen = self.calculate_edu_score(generated_text)
        improvement = gen - orig
        normalized = 0.0 if improvement <= 0 else min(1.0, improvement / 5.0)
        return orig, gen, improvement, normalized

    def compute_dclm_improvement(self, original_text: str, generated_text: str) -> tuple[float, float, float, float]:
        orig = calculate_dclm_score(original_text)
        gen = calculate_dclm_score(generated_text)
        improvement = gen - orig
        normalized = 0.0 if improvement <= 0 else min(1.0, improvement)
        return orig, gen, improvement, normalized

    def get_prompt_tokens(self) -> int:
        prompt_tokens = 0
        if hasattr(self, 'lm') and hasattr(self.lm, 'history') and self.lm.history:
            prompt_text = self.lm.history[-1]['messages'][0]['content']
            prompt_tokens = len(prompt_text.split())
        return prompt_tokens

    def compute_final_score(
        self,
        normalized_edu_improvement_score: float,
        normalized_dclm_improvement_score: float,
        length_similarity_score: float,
        prompt_efficiency_score: float,
    ) -> float:
        return (
            SCORE_WEIGHTS["edu_improvement"] * normalized_edu_improvement_score
            + SCORE_WEIGHTS["dclm_improvement"] * normalized_dclm_improvement_score
            + SCORE_WEIGHTS["length_similarity"] * length_similarity_score
            + SCORE_WEIGHTS["prompt_efficiency"] * prompt_efficiency_score
        )

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """
        Run DSPy GEPA optimization for prompt optimization.
        This pipeline step doesn't process input data but performs optimization and saves results.
        """
        with self.track_time():
            # Lazily create models on the worker to avoid slowing submission
            self._ensure_models()
            dspy.settings.configure(lm=self.lm)

            # Create slurm logs directory
            slurm_logs_path = Path(self.run_dir) / "slurm_logs"
            slurm_logs_path.mkdir(parents=True, exist_ok=True)

            # Prepare datasets
            self.trainset, self.valset, self.testset = self.prepare_datasets(self.train_size, self.val_size, self.test_size, self.seed)

        
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
    ) -> dspy.Prediction:
        """
        GEPA metric with four components:
        1. Educational quality improvement (maximize) - weight {SCORE_WEIGHTS['edu_improvement']}
        2. DCLM quality improvement (maximize) - weight {SCORE_WEIGHTS['dclm_improvement']}
        3. Length similarity between input/output (minimize absolute difference) - weight {SCORE_WEIGHTS['length_similarity']}
        4. Prompt efficiency (minimize prompt tokens) - weight {SCORE_WEIGHTS['prompt_efficiency']}
        Returns a Prediction with score and feedback. The improvement components are normalized to [0, 1].
        """
        try:
            # Get the original and generated text
            original_text, generated_text = gold.original_text, pred.generated_text
            
            # Component 1: Educational score improvement (maximize)
            original_edu_score, generated_edu_score, edu_improvement, normalized_edu_improvement_score = self.compute_edu_improvement(original_text, generated_text)

            # Component 2: DCLM score improvement (maximize)
            original_dclm_score, generated_dclm_score, dclm_improvement, normalized_dclm_improvement_score = self.compute_dclm_improvement(original_text, generated_text)

            # Component 3: Length similarity (minimize absolute difference)
            length_similarity_score = self.compute_length_similarity(original_text, generated_text)

            # Component 4: Prompt efficiency (minimize prompt tokens)
            prompt_tokens = self.get_prompt_tokens()
            prompt_efficiency_score = self.calculate_prompt_efficiency_score(prompt_tokens)

            # Combine components with weights
            final_score = self.compute_final_score(
                normalized_edu_improvement_score,
                normalized_dclm_improvement_score,
                length_similarity_score,
                prompt_efficiency_score,
            )
            
            # Create detailed feedback
            feedback_text = f"Evaluation Results:\n"
            feedback_text += (
                f"• Educational Quality (improvement, w={SCORE_WEIGHTS['edu_improvement']:.2f}): Original={original_edu_score:.3f}, "
                f"Generated={generated_edu_score:.3f}, Improvement={edu_improvement:.3f} "
                f"(normalized: {normalized_edu_improvement_score:.3f})\n"
            )
            feedback_text += (
                f"• DCLM Quality (improvement, w={SCORE_WEIGHTS['dclm_improvement']:.2f}): Original={original_dclm_score:.3f}, "
                f"Generated={generated_dclm_score:.3f}, Improvement={dclm_improvement:.3f} "
                f"(normalized: {normalized_dclm_improvement_score:.3f})\n"
            )
            feedback_text += (
                f"• Length Similarity (w={SCORE_WEIGHTS['length_similarity']:.2f}): Similarity Score={length_similarity_score:.3f}\n"
            )
            feedback_text += f"• Prompt Efficiency (w={SCORE_WEIGHTS['prompt_efficiency']:.2f}): {prompt_tokens} tokens, Efficiency Score={prompt_efficiency_score:.3f}\n"
            feedback_text += f"• Final Score: {final_score:.3f} (weighted sum)\n"
            
            # Add improvement suggestions
            if normalized_edu_improvement_score < 0.2:
                feedback_text += "→ Enhance the educational quality beyond the original text to improve the score.\n"
            if normalized_dclm_improvement_score < 0.2:
                feedback_text += "→ Enhance the DCLM quality beyond the original text to improve the score.\n"
            if length_similarity_score < 0.8:
                feedback_text += "→ Keep the generated text closer to the original length to improve the score.\n"
            if prompt_efficiency_score < 0.5:
                feedback_text += "→ Reduce prompt length to improve efficiency.\n"
            
            # Log for debugging
            logging.debug(
                "GEPA Metric - Edu: %.3f, DCLM: %.3f, Length: %.3f, Prompt: %.3f, Final: %.3f",
                normalized_edu_improvement_score,
                normalized_dclm_improvement_score,
                length_similarity_score,
                prompt_efficiency_score,
                final_score,
            )
            logging.debug(feedback_text)

            return dspy.Prediction(score=final_score, feedback=feedback_text)
            
        except Exception as e:
            error_msg = f"Error in metric evaluation: {e}"
            logging.error(error_msg)
            return dspy.Prediction(score=0.0, feedback=f"Evaluation failed: {error_msg}")


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
            elif model_name == "google/gemma-3-1b-it": # L4, vllm
                api_base = "https://qbmmldvdpornox7v.us-east-1.aws.endpoints.huggingface.cloud/v1/"
            elif model_name == "google/gemma-3-4b-it": # L4, vllm
                api_base = "https://tts5j24rpbz9xoc4.us-east-1.aws.endpoints.huggingface.cloud/v1/"
            elif model_name == "google/gemma-3-12b-it": # L4S, vllm
                api_base = "https://i2urnyf73c9e5p3d.us-east-1.aws.endpoints.huggingface.cloud/v1/"
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
            **({"temperature": temperature} if temperature is not None else {}),
            **({"top_p": top_p} if top_p is not None else {}),
            **({"top_k": top_k} if top_k is not None else {}),
        }
        
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
                    json.dump(record, f, ensure_ascii=False, indent=2, default=str)

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
            return {"n": 0, "weights": SCORE_WEIGHTS, "means": {}}

        # Show a single example
        example_length = 2000
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

        # Evaluate all test examples and collect results
        results = []
        for ex in tqdm(self.testset, desc="Evaluating test examples"):
            original_text = ex.original_text
            result = optimized_module(original_text=original_text)
            generated_text = result.generated_text

            # Compute all scores
            orig_edu, gen_edu, edu_impr, norm_edu_impr = self.compute_edu_improvement(original_text, generated_text)
            orig_dclm, gen_dclm, dclm_impr, norm_dclm_impr = self.compute_dclm_improvement(original_text, generated_text)
            length_sim = self.compute_length_similarity(original_text, generated_text)
            prompt_tokens = self.get_prompt_tokens()
            prompt_eff = self.calculate_prompt_efficiency_score(prompt_tokens)
            final = self.compute_final_score(norm_edu_impr, norm_dclm_impr, length_sim, prompt_eff)

            results.append({
                "original_text": original_text,
                "generated_text": generated_text,
                "original_edu": orig_edu,
                "generated_edu": gen_edu,
                "edu_improvement": edu_impr,
                "normalized_edu_improvement": norm_edu_impr,
                "original_dclm": orig_dclm,
                "generated_dclm": gen_dclm,
                "dclm_improvement": dclm_impr,
                "normalized_dclm_improvement": norm_dclm_impr,
                "length_similarity": length_sim,
                "prompt_efficiency": prompt_eff,
                "final_score": final,
            })

        # Create DataFrame and compute statistics
        df = pd.DataFrame(results)
        
        # Save test examples as parquet
        test_examples_path = Path(self.run_dir) / "test_examples.parquet"
        df.to_parquet(test_examples_path, index=False)
        logging.info(f"Saved {len(df)} test examples to {test_examples_path}")

        # Compute mean statistics
        score_columns = [col for col in df.columns if col not in ["original_text", "generated_text"]]
        means = df[score_columns].mean().to_dict()
        
        stats = {
            "n": n_test,
            "weights": SCORE_WEIGHTS,
            "means": means,
        }

        stats_path = Path(self.run_dir) / "score_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved score statistics to {stats_path}")
        
        return stats


parser = argparse.ArgumentParser(description="Optimize prompts for web-text tasks (rephrase, summarize) using DSPy GEPA")
parser.add_argument(
    "--generation-model",
    type=str,
    default="huggingface/google/gemma-3-4b-it",
    help="Model name for generation in format {provider}/{model_name} (default: huggingface/google/gemma-3-4b-it)"
)
parser.add_argument(
    "--reflection-model",
    type=str,
    default="huggingface/deepseek-ai/DeepSeek-V3.1-Terminus",
    help="Model name for reflection in format {provider}/{model_name} (default: huggingface/deepseek-ai/DeepSeek-V3.1-Terminus)."
)
# NOTE: The paper uses 150, 300, 300 in most tasks
parser.add_argument(
    "--train-size",
    type=int,
    default=500,
    help="Number of training examples (default: 500)"
)
parser.add_argument(
    "--val-size",
    type=int,
    default=100,
    help="Number of validation examples (default: 100)"
)
parser.add_argument(
    "--test-size",
    type=int,
    default=500,
    help="Number of new FineWeb samples for post-optimization evaluation (default: 500)"
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
    default=10,
    help="Budget for GEPA optimization (max_full_evals, default: 10)"
)
parser.add_argument(
    "--run-local",
    action="store_true",
    help="Run locally instead of using Slurm",
)
parser.add_argument(
    "--time",
    type=str,
    default="3-00:00:00",
    help="Slurm time limit (default: 3-00:00:00)",
)
parser.add_argument(
    "--qos",
    type=str,
    default="high",
    help="Slurm QoS (default: high)",
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
        executor = LocalPipelineExecutor(pipeline=[optimizer_step], logging_dir=optimizer_step.run_dir, skip_completed=not args.debug)
    else:
        # Create job name using hierarchical structure: task-model-train-val-budget
        generation_model_name = args.generation_model.split("/")[-1]
        job_name = f"optimize-{args.task}-{generation_model_name}-train-{args.train_size}-val-{args.val_size}-budget-{args.budget}-{args.name}"
        executor = SlurmPipelineExecutor(
            pipeline=[optimizer_step],
            tasks=1,
            time=args.time,
            partition="hopper-prod", # Still on GPU to speed up quality score computation
            cpus_per_task=11,
            mem_per_cpu_gb=22,
            qos=args.qos,
            logging_dir=optimizer_step.run_dir,
            job_name=job_name,
            env_command=ENV_COMMAND,
            depends_job_id=args.dep_job_id,
            sbatch_args={"gres": f"gpu:1"},
        )

    executor.run()


if __name__ == "__main__":
    main()
