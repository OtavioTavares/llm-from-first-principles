# Documentação

Notas de estudo do projeto. Cada arquivo acompanha uma etapa do roadmap e
explica a matemática antes do código.

| Doc | Assunto | Código correspondente |
|-----|---------|-----------------------|
| [01 — Tokenização](01_tokenizacao.md) | Por que tokenizar, o trade-off `V` × `n`, nível de caractere vs nível de byte | `src/llm/tokenizer/char.py` |
| [02 — BPE](02_bpe.md) | Byte Pair Encoding: o algoritmo de treino passo a passo | `src/llm/tokenizer/bpe.py` |
| [03 — Embeddings](03_embeddings.md) | De id inteiro para vetor em `R^d`: de onde vem o `d`, de onde vêm os números | `src/llm/embeddings/embedding.py` |

## Convenções usadas nas notas

| Símbolo | Significado |
|---------|-------------|
| `V` | tamanho do vocabulário (número de ids distintos) |
| `n` | comprimento da sequência em tokens |
| `d` | dimensão do embedding |
| `ids` | a sequência de inteiros que o modelo realmente consome |
