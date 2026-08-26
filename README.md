# train-dspark-draft-models

Train a **DSpark draft model** for speculative decoding, with
[`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) as the verifier — in **online** mode
(hidden states pulled from a live vLLM server, nothing cached to disk).

A drafter guesses several tokens ahead, the verifier checks them in one forward pass and keeps what
it agrees with — same output, faster. [DSpark](https://arxiv.org/abs/2607.05147) drafts a whole
block at a time and conditions on the verifier's hidden states, which is why a vLLM server runs
during training. Built on [vllm-project/speculators](https://github.com/vllm-project/speculators).

## Run it

[`notebooks/train-dspark-drafter-qwen-3-0-6b-online.ipynb`](notebooks/train-dspark-drafter-qwen-3-0-6b-online.ipynb)
— clone `speculators` → regenerate + tokenize data → serve the verifier → train → push to the Hub.

**Needs:** 2× T4 (tested on Kaggle; GPU 0 serves the verifier, GPU 1 trains), `vllm>=0.22.0`, and an
HF write token for the last step.

**Main knob:** `--limit` and `--max-samples` (1600 here) must move together — drop both to ~200 to
smoke-test, raise them well beyond 1600 for a drafter you actually plan to deploy.

**Output:** `output/checkpoints/checkpoint_best`, uploaded to
[`yosefw/Qwen3-0.6B-DSpark`](https://huggingface.co/yosefw/Qwen3-0.6B-DSpark). Load it in vLLM as
the speculative model and check the acceptance rate.

Based on the official
[speculators training tutorial](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train/).
