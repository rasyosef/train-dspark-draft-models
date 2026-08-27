# train-dspark-draft-models

Train a **DSpark draft model** for speculative decoding, with
[`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) as the verifier. Two notebooks do the
training job in two ways — **offline** and **online** — and differ only in where the verifier's
hidden states come from. A third notebook evaluates the drafter that comes out.

A drafter guesses several tokens ahead, the verifier checks them in one forward pass and keeps what
it agrees with — same output, faster. [DSpark](https://arxiv.org/abs/2607.05147) drafts a whole
block at a time and conditions on the verifier's hidden states, which is why a vLLM server runs
during training. Built on [vllm-project/speculators](https://github.com/vllm-project/speculators).

## Which notebook

`speculators` can source hidden states three ways — **online** (fetched from a live vLLM server on
demand and discarded), **offline** (all pre-generated to disk before training), and **hybrid**
(generated during epoch 0, cached, reused). These notebooks cover the first two.

Both were tested on Kaggle with **2× NVIDIA T4**, and the table describes that setup. Data
generation uses both T4s either way; the modes diverge once training starts.

| | offline | online |
| --- | --- | --- |
| Hidden states | extracted up front into `./output/hidden_states`, read via `--hidden-states-path` | fetched from the endpoint per batch (`--on-missing generate --on-generate delete`) |
| vLLM server | shut down *before* training starts | must stay up for the entire training run |
| Training | both T4s (`CUDA_VISIBLE_DEVICES=0,1`, `--nproc_per_node 2`) | GPU 1 only (`--nproc_per_node 1`) — GPU 0 is hosting the verifier |
| Runs on 1 GPU? | yes | not really — the server would have to share the card with the trainer |
| Cost | one extra extraction pass, and a lot of disk | no disk overhead, but the verifier occupies a GPU for the whole run |
| Samples | 1200 | 1600 |
| Wall clock (2× T4) | 5 h 53 min (1200 samples) | 8 h 54 min (1600 samples) |

**Offline is strongly recommended if you have a single GPU** — it materializes the hidden states,
frees the card, and then gives the whole thing to training, whereas online has the server and the
trainer competing for the same memory the entire run. With two or more GPUs both work: offline
still trains twice as wide, so pick online mainly when disk, not GPU count, is the tighter
constraint.

The wall-clock row is whole-notebook time — clone to Hub upload — at each notebook's own sample
count, so it is not a clean head-to-head. The three-hour gap runs the same direction as the
recommendation anyway: offline finishes sooner *despite* paying for an extra extraction pass,
because training gets both cards. Runtime is roughly linear in the sample count, so scale these
figures with the knob below.

## Run it

**Prerequisites:** `vllm>=0.22.0`, a GPU (two if you want the online notebook), and a Hugging Face
write token for the upload step. Everything else installs from inside the notebooks.

### 1. Train a drafter — pick one

[**`[offline] train-dspark-drafter-qwen-3-0.6b.ipynb`**](notebooks/%5Boffline%5D%20train-dspark-drafter-qwen-3-0.6b.ipynb)

clone `speculators` → regenerate + tokenize data → serve the verifier → extract hidden states →
**stop the server** → train on both GPUs → push to
[`yosefw/Qwen3-0.6B-DSpark-v2`](https://huggingface.co/yosefw/Qwen3-0.6B-DSpark-v2).

[**`[online] train-dspark-drafter-qwen-3-0.6b.ipynb`**](notebooks/%5Bonline%5D%20train-dspark-drafter-qwen-3-0.6b.ipynb)

clone `speculators` → regenerate + tokenize data → serve the verifier → train against it live →
push to [`yosefw/Qwen3-0.6B-DSpark`](https://huggingface.co/yosefw/Qwen3-0.6B-DSpark).

Both write `output/checkpoints/checkpoint_best` and upload it to the Hub. Budget for a long
session: end to end on 2× T4 the offline notebook took **5 h 53 min** and the online one
**8 h 54 min** — both fit a 12 h Kaggle session, neither by a comfortable margin. The sample count
below is the lever if you need them shorter.

**The one knob that matters** is the sample count, and it appears in more than one place: `--limit`
on response regeneration and `--max-samples` on `prepare_data.py` must always move together
(offline has a third one to match, on `data_generation_offline.py`). The defaults are 1200 offline
and 1600 online; drop them to ~200 to smoke-test the pipeline, and raise them well beyond the
defaults for a drafter you actually plan to deploy.

### 2. Evaluate it

[**`evaluate-dspark-qwen-3-0.6b.ipynb`**](notebooks/evaluate-dspark-qwen-3-0.6b.ipynb)

clone `speculators` → serve the drafter in vLLM, which pulls in its verifier automatically →
`evaluate.py throughput` → acceptance metrics in `acceptance.csv`. Its `sweep` subcommand runs the
full performance benchmark instead, across all 9
[`RedHatAI/speculator_benchmarks`](https://huggingface.co/datasets/RedHatAI/speculator_benchmarks)
subsets.

Point it at either the Hub id or the local `checkpoint_best` path from step 1. Takes about
**40 min** on 2× T4, most of it model download and server startup.

## Trained models

The drafters these notebooks produced, on the Hugging Face Hub:

| Model | Notebook | Verifier |
| --- | --- | --- |
| [`yosefw/Qwen3-0.6B-DSpark-v2`](https://huggingface.co/yosefw/Qwen3-0.6B-DSpark-v2) | offline, 1200 samples | [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) |
| [`yosefw/Qwen3-0.6B-DSpark`](https://huggingface.co/yosefw/Qwen3-0.6B-DSpark) | online, 1600 samples | [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) |

Both are trained on small sample counts, so treat them as pipeline demonstrations rather than
deployment-ready drafters.

### Evaluation

Per-subset results for [`yosefw/Qwen3-0.6B-DSpark`](https://huggingface.co/yosefw/Qwen3-0.6B-DSpark)
(the online-mode drafter, trained on 1600 samples), produced by
[`evaluate-dspark-qwen-3-0.6b.ipynb`](notebooks/evaluate-dspark-qwen-3-0.6b.ipynb) —
`evaluate.py throughput` against a vLLM server running the drafter on 2× T4:

| subset | num_drafts | num_draft_tokens | num_accepted_tokens | acceptance_length | acceptance_at_pos_0 | acceptance_at_pos_1 | acceptance_at_pos_2 | acceptance_at_pos_3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HumanEval | 225107 | 900428 | 196043 | 1.8709 | 0.2504 | 0.1168 | 0.0501 | 0.0204 |
| math_reasoning | 59956 | 239824 | 58168 | 1.9702 | 0.2673 | 0.1351 | 0.0647 | 0.0281 |
| qa | 20925 | 83700 | 16763 | 1.8011 | 0.2681 | 0.1258 | 0.0396 | 0.0188 |
| question | 56063 | 224252 | 44305 | 1.7903 | 0.2753 | 0.1265 | 0.0543 | 0.0217 |
| rag | 22824 | 91296 | 14539 | 1.637 | 0.2114 | 0.0898 | 0.039 | 0.0158 |
| summarization | 20684 | 82736 | 10040 | 1.4854 | 0.1568 | 0.0569 | 0.0224 | 0.009 |
| tool_call | 57005 | 228020 | 41188 | 1.7225 | 0.2036 | 0.0868 | 0.0384 | 0.0152 |
| translation | 21638 | 86552 | 9425 | 1.4356 | 0.1433 | 0.0519 | 0.0182 | 0.0063 |
| writing | 55683 | 222732 | 44927 | 1.8068 | 0.2074 | 0.0953 | 0.0435 | 0.019 |

`acceptance_length` is the mean number of tokens kept per verification round (`1 + accepted /
drafts`), capped at 5 here because the drafter is trained with `--block-size 4`.
`acceptance_at_pos_N` is how often the Nth drafted token in a block survives verification; it decays
across the block, which is exactly what DSpark's semi-autoregressive head exists to slow down.

Acceptance is highest on `math_reasoning` (1.970) and `HumanEval` (1.871) and lowest on
`translation` (1.436) and `summarization` (1.485) — the drafter does best where the verifier's next
token is most predictable. Across all subsets, weighted by `num_drafts`, acceptance length is
**1.806** (435,398 accepted tokens over 539,885 drafts).

Based on the official speculators
[training](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train/) and
[evaluating performance](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/evaluating_performance/)
tutorials.

## License

Apache-2.0 — see [`LICENSE`](LICENSE). Builds on
[`speculators`](https://github.com/vllm-project/speculators), also Apache-2.0.
