# LLM From First Principles

An implementation and experimental study of modern
Large Language Models from first principles.

The goal is not to build a production-ready LLM,
but to understand the mathematical, architectural
and computational foundations behind modern language models.

Starting from:
- tokenization
- embeddings
- self-attention

and progressively implementing:
- RoPE
- RMSNorm
- SwiGLU
- GQA
- Transformer blocks
- autoregressive training
- distributed training
- inference optimization

Everything is written from scratch in NumPy — no PyTorch, no autograd —
so that every gradient and every matrix multiplication is visible.

## Roadmap

- [x] Project setup
- [x] Tokenization — character-level baseline and byte-level BPE
- [x] Embeddings — lookup table, dot product, cosine similarity
- [ ] Self-Attention
- [ ] Multi-Head Attention
- [ ] RoPE
- [ ] RMSNorm
- [ ] SwiGLU
- [ ] Transformer Block
- [ ] GPT-style model
- [ ] Training
- [ ] Evaluation
- [ ] Generation
- [ ] GQA
- [ ] Llama-style model

## Getting started

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                # create the venv and install everything
uv run pytest -q       # 86 tests
```

Anything else runs through `uv run`:

```bash
uv run python                       # REPL with numpy and the llm package
uv run python experiments/...       # run an experiment
uv add <package>                    # add a dependency
```

## Layout

```
src/llm/
  tokenizer/     char.py      character-level baseline
                 bpe.py       byte-level Byte Pair Encoding
  embeddings/    embedding.py the (V, d) lookup table
                 similarity.py norm, dot product, cosine

docs/            study notes — the math behind each step (pt-BR)
experiments/     runnable scripts that produce the numbers in the docs
tests/           one file per component
data/            tinyshakespeare.txt (en), domcasmurro.txt (pt)
```

## Study notes

The reasoning behind each implementation lives in [`docs/`](docs/README.md),
written as the project progresses:

| Doc | Topic |
|-----|-------|
| [01 — Tokenização](docs/01_tokenizacao.md) | Why tokenize, the `V` × `n` trade-off, character vs byte level |
| [02 — BPE](docs/02_bpe.md) | The training loop step by step, measured compression curves |
| [03 — Embeddings](docs/03_embeddings.md) | From integer id to vector in `R^d`, similarity, high-dimensional geometry |

## Experiments

```bash
uv run python experiments/01_tokenization/measure.py      # char tokenizer baseline
uv run python experiments/01_tokenization/explore_bpe.py  # BPE merge by merge
uv run python experiments/01_tokenization/curva_v_n.py domcasmurro.txt
```

The last one trains BPE at increasing vocabulary sizes and measures
bytes-per-token on held-out text — the diminishing-returns curve that
justifies every tokenizer design decision.
