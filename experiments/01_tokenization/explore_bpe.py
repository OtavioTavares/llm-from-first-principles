"""Mostra o BPE construindo o vocabulario, merge a merge.

Rode com:  uv run python experiments/01_tokenization/explore_bpe.py
"""

from llm.tokenizer.bpe import BPETokenizer, get_stats, merge

TEXTO = "banana em um caixo de bananeira"
VOCAB_SIZE = 260  # 256 bytes + 4 merges

tok = BPETokenizer()
tok.train(TEXTO, vocab_size=VOCAB_SIZE)

# Reaplica os merges na ordem aprendida para ver a sequencia encolhendo.
ids = list(TEXTO.encode("utf-8"))
print(f"texto  : {TEXTO!r}")
print(f"bytes  : {ids}   ({len(ids)} tokens)")
print()

for (esq, dir_), novo in tok.merges.items():
    antes = len(ids)
    ids = merge(ids, (esq, dir_), novo)
    print(
        f"merge {novo}: ({esq:>3},{dir_:>3})  "
        f"{tok.vocab[esq]!r:>9} + {tok.vocab[dir_]!r:<9} = {tok.vocab[novo]!r:<10}"
        f"  ids={str(ids):<22} {antes} -> {len(ids)} tokens"
    )

print()
print(f"V final        : {len(tok.vocab)}")
print(f"merges         : {len(tok.merges)}")
print(f"bytes/token    : {len(TEXTO.encode('utf-8')) / len(ids):.2f}")
print()
print("vocabulario aprendido (so os ids >= 256):")
for i, bs in tok.vocab.items():
    if i >= 256:
        print(f"  {i} -> {bs!r}")

# Por que nao da pra pedir mais merges aqui:
print()
print(f"sobrou  : {ids}  -- sem par adjacente, get_stats devolve {get_stats(ids)}")
