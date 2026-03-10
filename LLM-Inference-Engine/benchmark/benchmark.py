
import time
import torch

def benchmark(decoder, prompts, max_new_tokens=100):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    start = time.time()

    outputs = decoder.generate(prompts, max_new_tokens)

    end = time.time()

    latency = end - start
    tokens = len(prompts) * max_new_tokens
    throughput = tokens / latency

    print("Latency:", latency, "seconds")
    print("Throughput:", throughput, "tokens/sec")

    return outputs
