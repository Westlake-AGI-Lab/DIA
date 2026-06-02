# Dynamic Infilling Anchors (DIA)

> **Dynamic Infilling Anchors for Format-Constrained Generation in Diffusion Large Language Models**
>
> The 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)

## 1. Introduction

Diffusion large language models (dLLMs) offer bidirectional attention and parallel generation, enabling them to exploit global context and naturally support format-constrained tasks like parseable JSON or reasoning templates. While straightforward fixed anchors can enforce such constraints, they often impose rigid spans, leading to truncated reasoning or redundant content. To overcome this, we propose **Dynamic Infilling Anchors (DIA)**, a training-free method that dynamically estimates end-anchor positions to adjust generation length before iterative infilling. This flexible mechanism ensures structural correctness and semantic coherence, avoiding the inefficiencies of fixed-span methods. Experiments on reasoning benchmarks demonstrate that DIA substantially improves format compliance and answer accuracy, achieving significant zero-shot gains on GSM8K and MATH. These results establish DIA as a robust pathway toward reliable, structure-aware generation.

DIA is built on [Dream-v0-Base-7B](https://huggingface.co/Dream-org/Dream-v0-Base-7B) (HKUNLP Group) and is evaluated in two settings: **reasoning** (`<think>...</think><answer>...</answer>` templates, via `lm-evaluation-harness`) and **JSON generation** (structured information extraction on Wikibio).

### Main Results

**Reasoning benchmarks** — format adherence (`S_format`, %) and accuracy (`Acc.`, %), 0-shot:

| Method | GSM8K `S_format` | GSM8K `Acc.` | MATH-500 `S_format` | MATH-500 `Acc.` |
|---|:--:|:--:|:--:|:--:|
| Dream-7B-Base | 0.00 | 68.99 | 0.00 | 25.14 |
| Dream-7B-Instruct | 0.00 | 15.01 | 0.00 | 25.28 |
| Infilling Baseline | 58.83 | 14.86 | 29.10 | 21.52 |
| **DIA (Ours)** | **72.63** | **46.78** | **76.82** | 20.08 |

**JSON generation (Wikibio)** — valid-format rate (`S_format`↑, %) and hallucination rate (`S_Hal`↓, %):

| Method | Regex `S_format`↑ | Regex `S_Hal`↓ | Raw `S_format`↑ | Raw `S_Hal`↓ |
|---|:--:|:--:|:--:|:--:|
| Dream-7B-Base | 40.72 | 12.35 | 0.00 | – |
| Dream-7B-Instruct | 66.74 | 5.10 | 52.80 | 4.81 |
| Infilling Baseline | 0.01 | 0.00 | 0.01 | 0.00 |
| **DIA (Ours)** | **79.84** | **0.15** | **79.84** | **0.15** |

## 2. Inference

Tested with Python 3.10, CUDA, a single ≥24 GB GPU (Dream-7B in bf16), and **`transformers==4.46.2`** (see the note in Step 1).

### Step 1 — Environment

DIA's reasoning evaluation runs inside `lm-evaluation-harness`. Clone and install it first, **then** install our `requirements.txt` so that the pinned `transformers==4.46.2` takes precedence.

```bash
git clone https://github.com/HKUNLP/DIA.git      # this repository
cd DIA

# Install lm-evaluation-harness (editable). This also pulls a recent transformers.
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
pip install -e lm-evaluation-harness

# Install our requirements LAST — this pins transformers back to 4.46.2.
pip install -r requirements.txt
```

### Step 2 — Model weights + DIA code injection

The DIA reasoning path is *invasive*: it must load the DIA model code via `trust_remote_code`. We assemble a local model directory that contains the **official Dream weights** plus our **DIA code** (`dia/`). Download the weights, then overwrite the model code:

```bash
# Download official weights + tokenizer into a local dir
hf download Dream-org/Dream-v0-Base-7B --local-dir resource/dream_dia

# Inject the DIA code (overwrites stock modeling, adds the DIA generation mixin)
cp dia/modeling_dream.py        resource/dream_dia/modeling_dream.py
cp dia/generation_utils_dia.py  resource/dream_dia/generation_utils_dia.py
```

`config.json`'s `auto_map` already points to `modeling_dream.DreamModel`, and DIA's `modeling_dream.py` imports `from .generation_utils_dia import ...` — so after the copy, `AutoModel.from_pretrained("resource/dream_dia", trust_remote_code=True)` loads the real DIA `_block_sample` path. (`configuration_dream.py` and `tokenization_dream.py` are identical to the official ones, so they need not be copied.)

> Skipping this injection silently falls back to stock Dream: the hub's `generation_utils.py` has no `_block_sample` and quietly ignores DIA's `format_ids`, so generation "looks fine" but **DIA is never applied**.

All evaluation scripts default to `resource/dream_dia` (resolved relative to the repo root). Override with `--model <path>` / `pretrained=<path>` if you place the model elsewhere.

### Step 3 — Run the evaluations

**Reasoning DIA** (`lm-evaluation-harness`, GSM8K by default):

```bash
bash reasoning/run_dia.sh
```

Edit the `tasks` / `lengths` / `temperatures` and the `--limit` flag at the top of `reasoning/run_dia.sh` to change benchmarks or run the full set. Then score format compliance:

```bash
python reasoning/format_eval.py --results_dir reasoning/evals_results
```

**JSON DIA** (Wikibio). First prepare the dataset, build the separator-token cache, then evaluate.

Download the [Wikipedia Biography Dataset](https://github.com/DavidGrangier/wikipedia-biography-dataset) (stored as split-zip archives — reassemble before unzipping) and convert its `test` split into our JSONL format:

```bash
# Clone + reassemble the split archives into the train/valid/test directories
git clone https://github.com/DavidGrangier/wikipedia-biography-dataset.git
( cd wikipedia-biography-dataset && zip -F wikipedia-biography-dataset.zip --out tmp.zip && unzip tmp.zip )

# Convert the test split -> data/biography_dataset_10000.jsonl
python json/process_biography_dataset.py \
    --data-dir wikipedia-biography-dataset/wikipedia-biography-dataset/test \
    --output data/biography_dataset_10000.jsonl \
    --max-samples 10000
```

Then build the separator-token cache and run the evaluation:

```bash
python json/build_separator_tokens.py --model_path resource/dream_dia --output separator_tokens.json

# Generate (single or multi-GPU is auto-detected); --max-samples for a quick run
bash json/run_json_eval.sh --max-samples 100

# Score S_fmt / S_Hal (expects each run in its own subdirectory under --base_dir)
python json/json_eval.py --base_dir eval_results
```

Key generation parameters (initial/max tokens per field, diffusion steps, expansion threshold, etc.) are exposed as flags on both `reasoning/run_dia.sh` and `json/run_json_eval.sh` — run the latter with `-h` for the full list.

---

Licensed under Apache 2.0 (see [LICENSE](LICENSE)). Builds on [Dream](https://github.com/HKUNLP/Dream) (HKUNLP Group) and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (EleutherAI).
