"""Carrega um tokenizer gravado em disco e usa.

Esta e a etapa BARATA, feita toda vez que o modelo roda. O arquivo contem
apenas pares de inteiros -- todo o vocabulario e RECONSTRUIDO na carga.

Rode com (depois de treinar_e_salvar.py):
    uv run python experiments/01_tokenization/carregar_e_usar.py
    uv run python experiments/01_tokenization/carregar_e_usar.py domcasmurro-4000.bpe
"""

import sys
import time
from pathlib import Path

from llm.tokenizer.bpe import BPETokenizer

RAIZ = Path(__file__).resolve().parents[2]
PASTA = RAIZ / "tokenizers"

if len(sys.argv) > 1:
    ARQUIVO = PASTA / sys.argv[1]
else:
    disponiveis = sorted(PASTA.glob("*.bpe")) if PASTA.exists() else []
    if not disponiveis:
        print("nenhum tokenizer encontrado em tokenizers/")
        print("rode antes: uv run python experiments/01_tokenization/treinar_e_salvar.py")
        raise SystemExit(1)
    ARQUIVO = disponiveis[0]

# ------------------------------------------------------------------- a carga

t0 = time.perf_counter()
tok = BPETokenizer.load(ARQUIVO)
tempo_carga = time.perf_counter() - t0

print(f"arquivo      : {ARQUIVO.relative_to(RAIZ)}  "
      f"({ARQUIVO.stat().st_size:,} bytes)")
print(f"tempo de carga: {tempo_carga * 1000:.1f} ms")
print(f"V reconstruido: {len(tok.vocab):,}")
print()

# ------------------------------------------- o vocab foi RECONSTRUIDO, nao lido


def mostra(bs: bytes) -> str:
    try:
        return repr(bs.decode("utf-8"))
    except UnicodeDecodeError:
        return bs.hex(" ")


print("O arquivo so tem pares de inteiros. Mesmo assim, o vocab tem conteudo:")
for i in (256, 257, 258, 300, 500, 1000):
    if i in tok.vocab:
        par = next((p for p, novo in tok.merges.items() if novo == i), None)
        print(f"  id {i:>4}  <- par {str(par):<12}  = {mostra(tok.vocab[i])}")
print()

# ------------------------------------------------------------------- usando

FRASES = [
    "Uma noite destas, vindo da cidade para o Engenho Novo",
    "coração",
    "🙂",
    "palavra-que-ele-nunca-viu-42",
]

print(f"{'texto':<56} {'bytes':>6} {'tokens':>7} {'b/tok':>6}")
print("-" * 80)
for frase in FRASES:
    ids = tok.encode(frase)
    n_bytes = len(frase.encode("utf-8"))
    assert tok.decode(ids) == frase, "roundtrip falhou!"
    print(f"{frase:<56} {n_bytes:>6} {len(ids):>7} {n_bytes / len(ids):>6.2f}")

print()
print("Todos os roundtrips passaram, inclusive emoji e texto nunca visto --")
print("cobertura universal vem de operar em bytes, nao de ter visto o caractere.")
print()

# ------------------------------------------------------- treinar vs carregar

ids = tok.encode(FRASES[0])
print(f"exemplo de ids: {ids[:12]}{' ...' if len(ids) > 12 else ''}")
print()
print("Compare com o tempo de treino reportado por treinar_e_salvar.py:")
print(f"  carregar levou {tempo_carga * 1000:.1f} ms.")
print("  E por isso que o tokenizer e um artefato que acompanha o modelo:")
print("  treina-se uma vez, carrega-se para sempre.")
