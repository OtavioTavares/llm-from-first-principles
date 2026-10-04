"""Mede o custo do CharTokenizer no corpus real.

Rode com:  uv run python experiments/01_tokenization/measure.py
"""

from pathlib import Path

from llm.tokenizer.char import CharTokenizer

# Caminho relativo a este arquivo, nao ao diretorio de onde voce chamou.
# Assim o script funciona de qualquer lugar.
RAIZ = Path(__file__).resolve().parents[2]
CORPUS = RAIZ / "data" / "tinyshakespeare.txt"

texto = CORPUS.read_text(encoding="utf-8")

tok = CharTokenizer(texto)
ids = tok.encode(texto)

n_bytes = len(texto.encode("utf-8"))
n_tokens = len(ids)

print(f"V (vocab_size)  = {tok.vocab_size}")
print(f"bytes do corpus = {n_bytes}")
print(f"tokens gerados  = {n_tokens}")
print(f"bytes por token = {n_bytes / n_tokens:.4f}")

# Sanidade: o roundtrip tem que sobreviver ao corpus inteiro, nao so aos testes.
assert tok.decode(ids) == texto
print("\nroundtrip OK em 1.1 MB")
