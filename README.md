# train-dspark-draft-models

Train and evaluate a [DSpark](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dspark/) draft model for `Llama-3.2-1B-Instruct`, using the [`speculators`](https://github.com/vllm-project/speculators) library and vLLM.

Speculative decoding speeds up inference by letting a small drafter propose several tokens at once, then having the full model verify the whole block in a single forward pass and keep the longest prefix it would have produced itself. Output is identical to running the verifier alone — the win is fewer verifier forward passes per token, not a different distribution. The metric that matters is *acceptance length*, the mean number of tokens kept per verification round.

DSpark builds on DFlash: instead of predicting the block autoregressively the way EAGLE-3 does, it predicts the entire block in one forward pass using anchored block diffusion, conditioned on hidden states read from selected verifier layers (`--target-layer-ids 2 5 8 11 14` here). Pure block-parallel drafting leaves no dependency between tokens inside a block, so acceptance decays toward the block's end; DSpark restores that dependency with a *Markov head* — a low-rank logit bias conditioned on the previous token — and adds a *confidence head* that estimates per-position acceptance probability. The `pos_0`…`pos_3` decay in the [Evaluation](#evaluation) table below is exactly this within-block effect.

- **Drafter** — 2 layers, ~0.25B params, proposes 4 tokens per cycle.
- **Verifier** — the full `Llama-3.2-1B-Instruct`, checks each block in one forward pass, so output is identical to running it alone.
- **Result** — **2.404** tokens accepted per verification round, 2.927 on HumanEval.
- **Speedup** — **2.17x** wall-clock decode throughput (68.3 → 148.2 tok/s on a T4 GPU), up to 2.77x on HumanEval.
- **Scope** — two notebooks: regenerate the data, train online against a live vLLM verifier, then serve and benchmark.

## Contents

- [What's in here](#whats-in-here)
- [Model](#model)
- [Usage](#usage)
- [Pipeline](#pipeline)
- [Run it](#run-it)
- [Evaluation](#evaluation)
- [Credits and license](#credits-and-license)

## What's in here

```
.
├── notebooks/
│   ├── data-preparation.ipynb
│   ├── train-llama-3.2-1b-instruct-dspark-online.ipynb
│   └── qwen-3-0.6b/
├── LICENSE
└── README.md
```

Everything runs inside the notebooks. Nothing to install; `speculators` is cloned at run time.

`notebooks/qwen-3-0.6b/` is an earlier drafter trained against `Qwen/Qwen3-0.6B`, in both offline and online modes — see [its own README](notebooks/qwen-3-0.6b/README.md).

## Model

- **Drafter** — [`rasyosef/Llama-3.2-1B-Instruct-speculator.dspark`](https://huggingface.co/rasyosef/Llama-3.2-1B-Instruct-speculator.dspark). Proposes 4 tokens per cycle for the verifier to check.
- **Verifier** — [`unsloth/Llama-3.2-1B-Instruct`](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct). Validates each proposed block in a single forward pass, so output is identical to running the verifier alone.

**Architecture:** 2 Qwen3 layers, ~0.25B params, bfloat16, block size 4, draft vocabulary reduced to 32,000 tokens.

**Training:** 32,000 regenerated Magpie samples, 6 epochs, 96/4 split, AdamW at 6e-4, loss weights `{"ce": 0.1, "tv": 0.9}`, sequence length 2048, up to 128 anchors per sample.

## Usage

```bash
vllm serve rasyosef/Llama-3.2-1B-Instruct-speculator.dspark \
  --port 8000 \
  --gpu-memory-utilization 0.75
```

Query the OpenAI-compatible endpoint at `http://localhost:8000/v1`.

The drafter is not a standalone model — it only works paired with its verifier.

## Pipeline

| Notebook | What it does |
|----------|--------------|
| `data-preparation.ipynb` | Regenerates 32,000 Magpie prompts with the verifier and pushes the JSONL to [`yosefw/magpie-llama-3.2-1b-instruct`](https://huggingface.co/datasets/yosefw/magpie-llama-3.2-1b-instruct) |
| `train-llama-3.2-1b-instruct-dspark-online.ipynb` | Tokenizes that data, trains the drafter **online** against a live vLLM verifier, pushes `checkpoint_best`, then serves and evaluates it |

Online training keeps the verifier resident and fetches hidden states per batch instead of caching them to disk. That needs two GPUs — verifier on GPU 0, training on GPU 1.

The data step is a one-off; its output is cached on the Hub, so retraining never re-runs generation.

## Run it

**Prerequisites:** `vllm>=0.22.0`, two GPUs, and a Hugging Face write token.

The notebooks read the token via `kaggle_secrets`. Off Kaggle that import fails; replace those cells with a direct `os.environ["HF_TOKEN"] = ...`.

Three things to keep in sync:

- **Sample count** — `--limit` on response regeneration and `--max-samples` on `prepare_data.py`. Both default to 32,000; drop to ~200 to smoke-test.
- **Sequence length** — `--seq-length` on `prepare_data.py` and `--total-seq-len` on `train.py`. Both are `2048` here; a mismatch silently truncates or wastes the tokenized data.
- **`--target-layer-ids`** — `2 5 8 11 14` here. It must match in the vLLM launch and training cells, or the drafter trains against hidden states the server isn't exporting.

Budget for a long session: the published model takes roughly **10–11 hours** to train on Kaggle's 2× T4.

## Evaluation

Measured with `evaluate.py throughput` in the training notebook, against the drafter served in vLLM.

| subset | acceptance_length | pos_0 | pos_1 | pos_2 | pos_3 |
| --- | --- | --- | --- | --- | --- |
| HumanEval | 2.927 | 75.4% | 54.0% | 37.6% | 25.8% |
| tool_call | 2.601 | 67.7% | 45.6% | 28.9% | 17.9% |
| math_reasoning | 2.585 | 69.6% | 45.7% | 27.8% | 15.4% |
| translation | 2.347 | 57.4% | 37.2% | 28.9% | 11.2% |
| writing | 2.336 | 58.7% | 35.8% | 22.9% | 16.1% |
| question | 2.149 | 57.2% | 31.3% | 17.0% | 9.4% |
| rag | 1.976 | 52.9% | 26.1% | 12.8% | 5.7% |
| qa | 1.834 | 46.4% | 22.6% | 9.9% | 4.4% |
| summarization | 1.736 | 46.0% | 18.6% | 6.7% | 2.3% |

`acceptance_length` is the mean tokens kept per verification round. `pos_N` is the percentage of blocks whose slot N survives verification, decaying across the block as intended. The two use different denominators, so `pos_N` does not sum to `acceptance_length`.

Acceptance is highest on structured tasks (code, math, tool calls) and lowest on translation and summarization.

Weighted over 117,372 verification steps, acceptance length is **2.404** — an upper bound on single-stream speedup, since it does not charge for the drafter's own forward pass.

### Wall-clock throughput

Single-stream decode, same vLLM server with and without the drafter, 77 prompts, T4 GPU.

| category | n | base tok/s | spec tok/s | ratio | median |
| --- | --- | --- | --- | --- | --- |
| humaneval | 7 | 62.1 | 172.3 | **2.77x** | 2.58x |
| math_reasoning | 10 | 64.8 | 177.4 | **2.74x** | 2.76x |
| writing | 10 | 64.5 | 171.4 | **2.66x** | 2.62x |
| question | 10 | 64.8 | 170.9 | **2.63x** | 2.62x |
| tool_call | 8 | 72.3 | 165.1 | **2.28x** | 2.34x |
| qa | 9 | 65.1 | 124.0 | **1.90x** | 1.91x |
| translation | 8 | 67.8 | 113.9 | **1.68x** | 1.69x |
| rag | 5 | 79.5 | 123.1 | **1.55x** | 1.59x |
| summarization | 10 | 78.0 | 104.6 | **1.34x** | 1.32x |
| **OVERALL** | **77** | **68.3** | **148.2** | **2.17x** | **2.08x** |

`ratio` is the mean per-prompt speedup, `median` its median across prompts. The ordering tracks acceptance length closely — the tasks that draft well (code, math) convert their accepted tokens into real speedup, while summarization and rag barely clear 1.3–1.6x. Overall throughput lands at **2.17x**, below the 2.404 acceptance-length bound, which is the cost of the drafter's own forward pass.

## Credits and license

**Built on**

- [`speculators`](https://github.com/vllm-project/speculators) — the DSpark trainer, converter, and `evaluate.py` used throughout, plus the official [training](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train/) and [performance evaluation](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/evaluating_performance/) tutorials these notebooks follow.
- [vLLM](https://github.com/vllm-project/vllm) — serves the verifier during online training and runs the speculative-decoding benchmark.
- [Magpie-Align](https://huggingface.co/Magpie-Align) — the seed prompts behind `speculators`' built-in `--dataset magpie`, regenerated with the verifier in `data-preparation.ipynb`.
- [`unsloth/Llama-3.2-1B-Instruct`](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct) — the verifier, a mirror of Meta's `Llama-3.2-1B-Instruct`.

Trained on Kaggle's free 2× T4 accelerators.

**License**

- This repository — Apache-2.0, see [`LICENSE`](LICENSE), matching `speculators`.
- The published drafter weights — subject to the [Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE), inherited from the verifier they are trained against and can only run with. Meta's [Acceptable Use Policy](https://www.llama.com/llama3_2/use-policy/) applies, as does the "Built with Llama" attribution requirement.
- The regenerated dataset — Llama 3.2 outputs, so the same Llama 3.2 terms apply on top of the source dataset's own license.
