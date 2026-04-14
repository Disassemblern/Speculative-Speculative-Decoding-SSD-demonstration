# SSD Demo — Speculative Speculative Decoding

A clean Python demonstration of three LLM inference strategies, implemented based on the paper:

> **Speculative Speculative Decoding**  
> Tanishq Kumar, Tri Dao, Avner May — arXiv:2603.03251 (2025)

### The Problem with Standard LLM Generation (AutoRegressive Model Architecture)

Large language models generate text one token at a time. Each token requires a full forward pass through the model — a sequential process that leaves most of the GPU's parallel compute unused.

### Speculative Decoding (SD)

Speculative decoding speeds this up by using a small, fast "draft" model to guess the next K tokens, then having the large "target" model verify all K guesses in a single parallel pass. Accepted guesses are kept; the first rejected one is replaced with a corrected token. The key property: the output is **mathematically identical** to running the target model alone (lossless).

### Speculative Speculative Decoding (SSD)

SD still has a hidden sequential bottleneck: the draft model waits idle while the target model is verifying. SSD eliminates this wait.

While the target model verifies round T's tokens, the draft model simultaneously:
1. Predicts what the verification outcome is likely to be
2. Pre-computes next-round speculations for each likely outcome
3. Stores them in a "speculation cache"

When verification finishes, the draft checks the cache. If the actual outcome was predicted correctly (a "cache hit"), the next speculation is returned instantly — zero additional wait. The whole system runs faster because drafting and verification now happen in parallel.

## Reference

Kumar, T., Dao, T., & May, A. (2025). *Speculative Speculative Decoding*. arXiv:2603.03251.