"""A curva de retornos decrescentes: quanto vale cada merge?

Treina o BPE numa fatia do corpus e mede a compressao em DUAS fatias:
  - a de treino  (o tokenizer viu)
  - a de teste   (o tokenizer nunca viu)   <- a metrica honesta

A diferenca entre as duas e o overfitting do tokenizer.

Rode com:
    uv run python experiments/01_tokenization/curva_v_n.py
    uv run python experiments/01_tokenization/curva_v_n.py domcasmurro.txt
"""

import sys
import time
from pathlib import Path

from llm.tokenizer.bpe import BPETokenizer, merge

RAIZ = Path(__file__).resolve().parents[2]
NOME = sys.argv[1] if len(sys.argv) > 1 else "tinyshakespeare.txt"
CORPUS = RAIZ / "data" / NOME

TREINO_BYTES = 300_000
TESTE_BYTES = 80_000
MAX_MERGES = 2000
CHECKPOINTS = [0, 50, 100, 250, 500, 1000, 1500, 2000]

texto = CORPUS.read_text(encoding="utf-8")
treino = texto[:TREINO_BYTES]
teste = texto[-TESTE_BYTES:]  # fim do corpus: o treino nunca chegou ate aqui

n_bytes = len(texto.encode("utf-8"))
n_chars = len(texto)

print(f"corpus : {NOME}")
print(f"treino : {len(treino.encode('utf-8')):,} bytes")
print(f"teste  : {len(teste.encode('utf-8')):,} bytes  (nunca visto)")
print(f"utf-8  : {n_bytes / n_chars:.4f} bytes por caractere "
      f"({n_bytes - n_chars:,} caracteres multi-byte)")
print(f"treinando {MAX_MERGES} merges...", end="", flush=True)

t0 = time.perf_counter()
tok = BPETokenizer()
tok.train(treino, vocab_size=256 + MAX_MERGES)
print(f" {time.perf_counter() - t0:.1f}s\n")

# Reaplica os merges um a um, medindo nos dois conjuntos a cada checkpoint.
ids_tr = list(treino.encode("utf-8"))
ids_te = list(teste.encode("utf-8"))
bytes_tr, bytes_te = len(ids_tr), len(ids_te)

linhas = []


def registra(k: int) -> None:
    linhas.append((k, 256 + k, bytes_tr / len(ids_tr), bytes_te / len(ids_te)))


registra(0)
for k, (par, novo) in enumerate(tok.merges.items(), start=1):
    ids_tr = merge(ids_tr, par, novo)
    ids_te = merge(ids_te, par, novo)
    if k in CHECKPOINTS:
        registra(k)

# ---------------------------------------------------------------- resultados

base = linhas[0][3]  # bytes/token no teste com V=256 (bytes crus)

print(f"{'merges':>7} {'V':>6} {'b/tok treino':>13} {'b/tok teste':>12} "
      f"{'gap':>6} {'ganho':>7} {'atencao':>9}")
print("-" * 68)

anterior = None
for k, v, btr, bte in linhas:
    ganho = "" if anterior is None else f"+{bte - anterior:.2f}"
    gap = f"{(btr / bte - 1) * 100:+.1f}%"
    # custo da atencao ~ n^2, relativo ao tokenizer de bytes puro
    atencao = f"{(base / bte) ** 2:.2f}x"
    print(f"{k:>7} {v:>6} {btr:>13.2f} {bte:>12.2f} {gap:>6} {ganho:>7} {atencao:>9}")
    anterior = bte

# ------------------------------------------------------- o que ele aprendeu


def mostra(bs: bytes) -> str:
    """Tenta exibir como texto; cai para hex se os bytes nao formarem utf-8."""
    try:
        return repr(bs.decode("utf-8"))
    except UnicodeDecodeError:
        return bs.hex(" ")


print("\ntokens aprendidos em cada faixa:")
for ini in (0, 250, 750, 1950):
    amostra = [tok.vocab[256 + i] for i in range(ini, ini + 10) if 256 + i in tok.vocab]
    print(f"  merge {ini:>4}-{ini + 10:<4}: {' '.join(mostra(b) for b in amostra)}")

# ------------------------------------------------- utf-8 redescoberto sozinho

multibyte = {}
for ch in sorted(set(texto)):
    if not ch.isascii():
        par = tuple(ch.encode("utf-8"))
        if len(par) == 2 and par in tok.merges:
            multibyte[ch] = tok.merges[par] - 256

if multibyte:
    print(f"\ncaracteres multi-byte que o BPE reconstituiu (merge em que aconteceu):")
    for ch, k in sorted(multibyte.items(), key=lambda kv: kv[1]):
        par = list(ch.encode("utf-8"))
        print(f"  merge {k:>4}: {par} -> {ch!r}")
    print(f"\n  {len(multibyte)} caracteres reconstituidos. "
          f"Ninguem ensinou utf-8 ao algoritmo -- ele so contou frequencias.")
