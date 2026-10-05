"""Treina um tokenizer BPE num corpus real e grava em disco.

Esta e a etapa CARA, feita uma vez. O resultado e um arquivo que pode ser
carregado instantaneamente para sempre -- veja carregar_e_usar.py.

Rode com:
    uv run python experiments/01_tokenization/treinar_e_salvar.py
    uv run python experiments/01_tokenization/treinar_e_salvar.py domcasmurro.txt 4000
"""

import sys
import time
from pathlib import Path

from llm.tokenizer.bpe import BPETokenizer

RAIZ = Path(__file__).resolve().parents[2]

NOME = sys.argv[1] if len(sys.argv) > 1 else "domcasmurro.txt"
NUM_MERGES = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

CORPUS = RAIZ / "data" / NOME
DESTINO = RAIZ / "tokenizers" / f"{CORPUS.stem}-{NUM_MERGES}.bpe"
DESTINO.parent.mkdir(exist_ok=True)

texto = CORPUS.read_text(encoding="utf-8")
n_bytes = len(texto.encode("utf-8"))

print(f"corpus : {NOME}  ({n_bytes:,} bytes)")
print(f"alvo   : V = 256 + {NUM_MERGES} = {256 + NUM_MERGES}")
print()

print("treinando...", end="", flush=True)
t0 = time.perf_counter()
tok = BPETokenizer()
tok.train(texto, vocab_size=256 + NUM_MERGES)
tempo_treino = time.perf_counter() - t0
print(f" {tempo_treino:.1f}s")

tok.save(DESTINO)

# ------------------------------------------------------------------ relatorio

ids = tok.encode(texto)
tamanho_arquivo = DESTINO.stat().st_size

print()
print(f"salvo em        : {DESTINO.relative_to(RAIZ)}")
print(f"tamanho         : {tamanho_arquivo:,} bytes")
print(f"V               : {len(tok.vocab):,}")
print(f"merges          : {len(tok.merges):,}")
print(f"bytes/token     : {n_bytes / len(ids):.2f}")
print(f"tempo de treino : {tempo_treino:.1f}s")
print()
print("As tres primeiras linhas do arquivo (so inteiros -- o vocab nao e salvo):")
for linha in DESTINO.read_text(encoding="utf-8").splitlines()[:4]:
    print(f"  {linha}")
print("  ...")
print()
print(f"{tamanho_arquivo:,} bytes de pares de inteiros reconstroem "
      f"{len(tok.vocab):,} tokens.")
print("Agora rode: uv run python experiments/01_tokenization/carregar_e_usar.py")
