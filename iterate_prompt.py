"""
Interactive script to iterate through dataset examples with quick prompt changes.

Supports Gemma models via LiteLLM (HF inference endpoints) or optional local
transformers generation. Allows changing prompts and re-running on same or new
examples interactively.
"""

import argparse
import os
import random
from pathlib import Path
from typing import Optional, Tuple

import litellm
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

torch.set_float32_matmul_precision('high')

from utils import calculate_edu_score_dict, print_debug_output
from datatrove.data import Document

# Load environment variables
load_dotenv()

# Suppress LiteLLM verbose output
litellm.drop_params = True
litellm.set_verbose = False


GEMMA_MODELS = {
    "270m": {
        "name": "huggingface/google/gemma-3-270m-it",
        "endpoint": "https://y7ftany7ifv3hpqh.us-east-1.aws.endpoints.huggingface.cloud/v1/",
    },
    "1b": {
        "name": "huggingface/google/gemma-3-1b-it",
        "endpoint": "https://qbmmldvdpornox7v.us-east-1.aws.endpoints.huggingface.cloud/v1/",
    },
    "4b": {
        "name": "huggingface/google/gemma-3-4b-it",
        "endpoint": "https://tts5j24rpbz9xoc4.us-east-1.aws.endpoints.huggingface.cloud/v1/",
    },
    "12b": {
        "name": "huggingface/google/gemma-3-12b-it",
        "endpoint": "https://i2urnyf73c9e5p3d.us-east-1.aws.endpoints.huggingface.cloud/v1/",
    },
}


def get_model_config(model_size: str) -> tuple[str, str]:
    """Get the model name and endpoint for a given model size.
    
    Returns:
        Tuple of (model_name, api_endpoint)
    """
    if model_size not in GEMMA_MODELS:
        raise ValueError(
            f"Unsupported model size: {model_size}. "
            f"Supported sizes: {', '.join(GEMMA_MODELS.keys())}"
        )
    config = GEMMA_MODELS[model_size]
    return config["name"], config["endpoint"]


def load_dataset_examples(data_path: str, n_examples: int, text_key: str = "text") -> list:
    """Load examples from dataset."""
    print(f"📂 Loading {n_examples} examples from {data_path}...")
    
    if data_path.startswith("hf://"):
        # Parse hf://datasets/owner/dataset or hf://datasets/owner/dataset/subset
        path_parts = data_path.replace("hf://datasets/", "").split("/")
        if len(path_parts) >= 2:
            dataset_name = "/".join(path_parts[:2])
            subset = path_parts[2] if len(path_parts) > 2 else None
        else:
            raise ValueError(f"Invalid hf:// path format: {data_path}")
        
        dataset = load_dataset(dataset_name, subset, split="train", streaming=True)
        
        # Skip random number of examples for variety
        skip_count = random.randint(0, 1000)
        dataset_iter = iter(dataset)
        for _ in range(skip_count):
            try:
                next(dataset_iter)
            except StopIteration:
                break
        
        examples = []
        for i, example in enumerate(dataset_iter):
            if len(examples) >= n_examples:
                break
            text = example.get(text_key, '').strip()
            if text and len(text) > 100:
                examples.append(text)
            if i > n_examples * 10:
                break
    else:
        raise ValueError(f"Only hf:// paths supported, got: {data_path}")
    
    print(f"✓ Loaded {len(examples)} examples")
    return examples


def load_prompt_template(template_path: str) -> str:
    """Load a prompt template from prompts directory."""
    if not template_path:
        return None
    
    script_dir = Path(__file__).parent
    full_path = script_dir / "prompts" / template_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {full_path}")
    
    return full_path.read_text(encoding='utf-8').strip()


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text."""
    if text and tokenizer:
        return len(tokenizer.encode(text))
    return 0


LocalModelBundle = Tuple[PreTrainedTokenizer, PreTrainedModel, torch.device]


def call_model(
    prompt: str,
    api_base: str,
    api_key: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    run_local: bool = False,
    local_model_bundle: Optional[LocalModelBundle] = None,
) -> str:
    """Generate a response either remotely (LiteLLM) or locally (transformers)."""
    if run_local:
        if not local_model_bundle:
            raise ValueError("Local model bundle is required when run_local=True")

        tokenizer, model, device = local_model_bundle

        return run_local_generation(
            tokenizer,
            model,
            device,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    messages = [
        {"role": "user", "content": prompt}
    ]
    
    response = litellm.completion(
        model=model_name,
        messages=messages,
        api_base=api_base,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        extra_headers={"X-HF-Bill-To": "huggingface"},
    )
    
    return response.choices[0].message.content


def run_local_generation(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: torch.device,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    inputs = tokenizer(prompt, return_tensors="pt")
    if device:
        inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the newly generated continuation
    generated_only = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_only, skip_special_tokens=True)


def process_example(
    text: str,
    prompt: str,
    api_base: str,
    api_key: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    tokenizer,
    edu_tokenizer,
    edu_model,
    *,
    run_local: bool,
    local_model_bundle: Optional[LocalModelBundle],
) -> Document:
    """Process a single example and return Document with metadata."""
    # Run rephrasing
    generated_text = call_model(
        prompt.replace("[TEXT]", text),
        api_base,
        api_key,
        model_name,
        max_tokens,
        temperature,
        top_p,
        top_k,
        run_local=run_local,
        local_model_bundle=local_model_bundle,
    )
    
    # Calculate metrics
    input_token_count = count_tokens(text, tokenizer)
    output_token_count = count_tokens(generated_text, tokenizer)
    
    input_edu_scores = calculate_edu_score_dict(text, edu_tokenizer, edu_model)
    output_edu_scores = calculate_edu_score_dict(generated_text, edu_tokenizer, edu_model)
    
    # Create document with metadata
    doc = Document(id="", text=generated_text)
    doc.metadata = {
        "input": {
            "text": text,
            "token_count": input_token_count,
            **input_edu_scores
        },
        "thinking": {
            "text": "",
            "token_count": 0,
        },
        "token_count": output_token_count,
        **output_edu_scores,
        "token_reduction": input_token_count - output_token_count,
        "edu_score_difference": output_edu_scores["score"] - input_edu_scores["score"],
        "edu_score_improvement": 1 if output_edu_scores["score"] > input_edu_scores["score"] else 0,
    }
    
    return doc


def run_examples(
    examples: list,
    prompt: str,
    api_base: str,
    api_key: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    tokenizer,
    edu_tokenizer,
    edu_model,
    *,
    run_local: bool,
    local_model_bundle: Optional[LocalModelBundle],
):
    """Run rephrasing on examples and display results."""
    print(f"\n{'='*100}")
    print(f"🚀 Running {len(examples)} examples...")
    print(f"{'='*100}\n")
    
    for i, text in enumerate(examples, 1):
        print(f"\n{'='*100}")
        print(f"Example {i}/{len(examples)}")
        print(f"{'='*100}")
        
        try:
            doc = process_example(
                text, prompt, api_base, api_key, model_name,
                max_tokens, temperature, top_p, top_k,
                tokenizer, edu_tokenizer, edu_model,
                run_local=run_local,
                local_model_bundle=local_model_bundle,
            )
            print_debug_output(doc, "", doc.text)
        except Exception as e:
            print(f"❌ Error processing example {i}: {e}")
            import traceback
            traceback.print_exc()


def warmup_model(api_base: str, api_key: str, model_name: str) -> None:
    """Send a test request to wake up the model endpoint."""
    print(f"🔥 Warming up {model_name}...")
    try:
        # Only perform warmup for remote usage. Local run should not call remote.
        response = call_model(
            prompt="You are a helpful assistant.",
            api_base=api_base,
            api_key=api_key,
            model_name=model_name,
            max_tokens=10,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            run_local=False,
        )
        print(f"✓ {model_name} ready! (Response: {response[:50]}...)")
    except Exception as e:
        print(f"⚠️  Warmup request failed, but continuing: {e}")


def setup_model(model_size: str) -> tuple[str, str, str]:
    """Setup model configuration and warmup.
    
    Returns:
        Tuple of (model_name, api_base, api_key)
    """
    model_name, api_base = get_model_config(model_size)
    api_key = os.getenv("HF_TOKEN")
    if not api_key:
        raise ValueError("HF_TOKEN environment variable is required")
    
    print(f"\n🎯 Setting up model: {model_size} ({model_name})")
    print(f"   Endpoint: {api_base}")
    
    # Warmup the model
    warmup_model(api_base, api_key, model_name)
    
    return model_name, api_base, api_key


def setup_local_model(model_name: str) -> LocalModelBundle:
    """Download and load a local causal LM for streaming inference."""
    print(f"\n🎯 Setting up local model: {model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"   Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("   Using CPU (this may be slow!) Run `srun --gpus=1 --qos=high --time=\"04:00:00\" --pty bash` to speed up generation.")

    if device.type == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
    elif device.type == "cuda":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    generation_tokenizer = AutoTokenizer.from_pretrained(model_name)
    if generation_tokenizer.pad_token_id is None:
        generation_tokenizer.pad_token_id = generation_tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()

    print("✓ Local model ready!")
    return generation_tokenizer, model, device


def interactive_loop(args):
    """Main interactive loop."""
    # Setup initial model
    current_model_size = args.model_size

    local_model_bundle = None
    if args.run_local:
        remote_friendly_name = GEMMA_MODELS[current_model_size]["name"]
        model_name = remote_friendly_name.replace("huggingface/", "")
        api_base = ""
        api_key = ""
        local_model_bundle = setup_local_model(model_name)
    else:
        model_name, api_base, api_key = setup_model(current_model_size)
    
    # Load tokenizer for counting
    print(f"📊 Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # Load edu classifier
    print("📚 Loading fineweb-edu classifier...")
    edu_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceFW/fineweb-edu-classifier")
    edu_model = AutoModelForSequenceClassification.from_pretrained("HuggingFaceFW/fineweb-edu-classifier").eval()
    
    # Load initial prompt
    if not args.prompt:
        print("❌ Error: --prompt argument is required")
        return
    
    prompt_path = args.prompt
    try:
        current_prompt = load_prompt_template(prompt_path)
        print(f"\n{'='*100}")
        print(f"📝 LOADED PROMPT: {prompt_path}")
        print(f"{'='*100}")
        print(current_prompt)
        print(f"{'='*100}\n")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    # Initial setup
    current_examples = None
    
    print(f"\n{'='*100}")
    print("🎮 INTERACTIVE PROMPT ITERATION")
    print(f"{'='*100}\n")
    
    while True:
        print("\n" + "="*100)
        print("OPTIONS:")
        print("  1) Load new examples and run")
        print("  2) Rerun on same examples")
        print(f"  3) Change model (current: {current_model_size})")
        print("="*100)
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            # Load new examples
            current_examples = load_dataset_examples(args.data_path, args.n_examples, args.text_key)
            
            # Reload prompt from file
            try:
                current_prompt = load_prompt_template(prompt_path)
                print(f"\n{'='*100}")
                print(f"📝 RELOADED PROMPT: {prompt_path}")
                print(f"{'='*100}")
                print(current_prompt)
                print(f"{'='*100}\n")
            except FileNotFoundError as e:
                print(f"❌ Error reloading prompt: {e}")
                continue
            
            # Run with current prompt
            run_examples(
                current_examples, current_prompt,
                api_base, api_key, model_name,
                args.max_tokens, args.temperature, args.top_p, args.top_k,
                tokenizer, edu_tokenizer, edu_model,
                run_local=args.run_local,
                local_model_bundle=local_model_bundle,
            )
            
        elif choice == "2":
            # Rerun on same examples
            if current_examples is None:
                print("❌ No examples loaded yet. Choose option 1 first.")
                continue
            
            # Reload prompt from file
            try:
                current_prompt = load_prompt_template(prompt_path)
                print(f"\n{'='*100}")
                print(f"📝 RELOADED PROMPT: {prompt_path}")
                print(f"{'='*100}")
                print(current_prompt)
                print(f"{'='*100}\n")
            except FileNotFoundError as e:
                print(f"❌ Error reloading prompt: {e}")
                continue
            
            # Run with current prompt
            run_examples(
                current_examples, current_prompt,
                api_base, api_key, model_name,
                args.max_tokens, args.temperature, args.top_p, args.top_k,
                tokenizer, edu_tokenizer, edu_model,
                run_local=args.run_local,
                local_model_bundle=local_model_bundle,
            )
            
        elif choice == "3":
            # Change model
            print(f"\nAvailable models: {', '.join(GEMMA_MODELS.keys())}")
            new_model_size = input(f"Enter model size (current: {current_model_size}): ").strip()
            
            if new_model_size in GEMMA_MODELS:
                current_model_size = new_model_size
                if args.run_local:
                    remote_friendly_name = GEMMA_MODELS[current_model_size]["name"]
                    model_name = remote_friendly_name.replace("huggingface/", "")
                    local_model_bundle = setup_local_model(model_name)
                    api_base = ""
                    api_key = ""
                else:
                    model_name, api_base, api_key = setup_model(current_model_size)
            else:
                print(f"❌ Invalid model size. Supported: {', '.join(GEMMA_MODELS.keys())}")
            
        else:
            print("❌ Invalid choice. Please enter 1-3.")


parser = argparse.ArgumentParser("Interactive prompt iteration with dataset examples.")

parser.add_argument(
    "--data-path", type=str, default="hf://datasets/HuggingFaceFW/fineweb/sample-10BT", help="Dataset path (hf://datasets/...)"
)
parser.add_argument(
    "--prompt", type=str, required=True, help="Prompt template path (relative to prompts/)"
)
parser.add_argument(
    "--model-size",
    type=str,
    default="4b",
    choices=list(GEMMA_MODELS.keys()),
    help="Gemma model size (default: 4b)"
)
parser.add_argument(
    # TODO: In this version the model often goes on for a long time before stopping,
    #  we might need to add stop sequences or switch to vllm
    "--run-local",
    action="store_true",
    help="Run inference locally using transformers instead of remote endpoint",
)
parser.add_argument(
    "--n-examples", type=int, default=5, help="Number of examples to process"
)
parser.add_argument(
    "--temperature", type=float, default=1.0, help="Temperature for inference"
)
parser.add_argument(
    "--top-p", type=float, default=0.95, help="Top-p for inference"
)
parser.add_argument(
    "--top-k", type=int, default=64, help="Top-k for inference"
)
parser.add_argument(
    "--max-tokens", type=int, default=4096, help="Maximum tokens per request"
)
parser.add_argument(
    "--tokenizer", type=str, default="hynky/Llama-3.2-1B-no-bos", help="Tokenizer for counting"
)
parser.add_argument(
    "--text-key", type=str, default="text", help="Text key in dataset"
)


def main():
    args = parser.parse_args()
    interactive_loop(args)


if __name__ == "__main__":
    main()
