import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from engine.kv_cache import KVCacheEngine
from engine.batch_decoder import BatchDecoder
from engine.sampler import Sampler
from benchmark.benchmark import benchmark


def main():

    model_name = "gpt2"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model with 8-bit quantization...")
    quant_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto"
    )

    model.eval()

    print("Initializing inference engine...")

    kv_engine = KVCacheEngine(model, tokenizer)
    sampler = Sampler()
    decoder = BatchDecoder(model, tokenizer, kv_engine, sampler)

    prompts = [
        "Explain machine learning in simple terms.",
        "What is the future of artificial intelligence?",
        "Write a short story about space exploration."
    ]

    print("Running benchmark...")

    outputs = benchmark(decoder, prompts)

    for o in outputs:
        print("\n---\n")
        print(o)


if __name__ == "__main__":
    main()
