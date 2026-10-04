# 01 — Tokenização

> Código: `src/llm/tokenizer/char.py` · Testes: `tests/test_char_tokenizer.py`

## O problema

Um modelo de linguagem não vê texto. Ele vê uma sequência de inteiros:

```
"banana"  →  [98, 97, 110, 97, 110, 97]  →  modelo
```

O tokenizer é a ponte nos dois sentidos — `encode` (texto → ids) e `decode`
(ids → texto). A exigência mínima é que a ida e a volta preservem o original:

```
decode(encode(x)) == x    para todo x
```

A pergunta de design é **quais** inteiros, e isso se resume a dois números:

- `V` — quantos símbolos distintos existem no vocabulário
- `n` — quantos tokens o texto ocupa depois de codificado

Eles são inversamente ligados. Comprimir mais texto em cada token diminui `n`,
mas exige um `V` maior para acomodar os símbolos novos.

## Por que o trade-off é assimétrico

`V` e `n` não entram no custo do modelo da mesma forma:

| Componente | Custo |
|---|---|
| Tabela de embeddings | `V · d` parâmetros |
| Projeção final para logits | `n · d · V` operações |
| Self-attention | **`n² · d`** operações |

`V` aparece sempre **linear**. `n` aparece **quadrático** na atenção.

Essa assimetria decide tudo: dobrar `V` dobra o custo da tabela de embeddings,
mas se isso reduzir `n` pela metade, a atenção fica **4× mais barata**. É por
isso que ninguém treina LLM em caracteres, mesmo sendo a opção mais simples.

## A unidade de medida honesta

A loss de um LLM é cross-entropy medida em **nats por token**. Essa unidade não
é comparável entre tokenizers diferentes: um tokenizer que engole mais texto por
token naturalmente tem loss maior por token, e ainda assim pode ser melhor.

A conversão para uma unidade neutra:

```
bits_por_byte  =  (loss_em_nats_por_token / ln 2)  ×  (nº de tokens / nº de bytes)
                   └──── conversão nat → bit ────┘     └──── taxa de compressão ────┘
```

Byte é unidade física, igual em qualquer língua ou alfabeto. "Caractere" não é —
veja a seção sobre UTF-8 abaixo.

**Consequência prática:** trocar o tokenizer muda a unidade da própria loss. Não
dá para comparar dois modelos pela loss sem antes normalizar para bits/byte.

---

## Nível de caractere

A abordagem mais direta: um id por caractere distinto do corpus.

```python
chars = sorted(set(text))                      # os distintos, em ordem fixa
stoi  = {ch: i for i, ch in enumerate(chars)}  # caractere → id
itos  = {i: ch for ch, i in stoi.items()}      # id → caractere
```

O `sorted` não é estética. Sem ele, `set` não garante ordem, e o mesmo corpus
produziria vocabulários diferentes entre execuções — um modelo treinado hoje não
conseguiria ler os ids de amanhã.

### Medição no Tiny Shakespeare

`experiments/01_tokenization/measure.py`:

```
V (vocab_size)  = 65
bytes do corpus = 1115394
tokens gerados  = 1115394
bytes por token = 1.0000
```

O `V = 65` se decompõe em:

```
 1  newline        '\n'
 1  espaço         ' '
10  pontuações     ! $ & ' , - . : ; ?
 1  dígito         '3'
52  letras         A-Z (26) + a-z (26)
──
65
```

Note o que o inventário revela sobre o corpus:

- `'$'` aparece **uma vez** em 1,1 milhão de caracteres. Ganha uma linha inteira
  na tabela de embeddings para receber gradiente uma vez por época.
- `'3'` é o único dígito. O modelo é incapaz de escrever o número 7 — o id não
  existe, o softmax não tem essa saída.
- Zero acentos. `encode("coração")` levanta `KeyError`.

### As duas falhas fatais

**1. `V` não é uma decisão, é uma consequência.**

```
V = len(set(corpus))
```

Você não escolhe. Trocar de corpus troca o `V` silenciosamente e reembaralha
todos os ids. Dois tokenizers com `V = 65` treinados em corpora diferentes são
incompatíveis, porque `'a'` pode ser `39` num e `41` no outro.

**2. Cobertura incompleta.** O vocabulário só contém o que apareceu no treino.
Qualquer caractere novo é um `KeyError`.

Dá para resolver aumentando o vocabulário até cobrir Unicode? Unicode tem ~150
mil codepoints atribuídos e **cresce a cada versão**. Você teria `V = 150.000`
com a esmagadora maioria dos embeddings nunca vendo um exemplo de treino —
parâmetros mortos pagando custo de memória e de softmax.

---

## Caractere ≠ byte

O `1.0000` da medição esconde uma coincidência: **Tiny Shakespeare é ASCII puro**
(maior codepoint = 122, a letra `z`). Em ASCII, um caractere ocupa exatamente um
byte. Fora do ASCII isso quebra:

| texto | caracteres | bytes | bytes/token |
|---|---|---|---|
| `'banana'` | 6 | 6 | 1.00 |
| `'coracao'` | 7 | 7 | 1.00 |
| `'coração'` | 7 | **9** | 1.29 |
| `'não é'` | 5 | **7** | 1.40 |
| `'🙂'` | 1 | **4** | 4.00 |

UTF-8 é um código de comprimento variável: 1 a 4 bytes por caractere.

```
'ç'  → [195, 167]              2 bytes
'🙂' → [240, 159, 153, 130]    4 bytes
```

Repare que `'coração'` tem **1.29 bytes/token** — parece melhor que o inglês, mas
o tokenizer não ficou mais esperto. Continua 1 token por caractere; é o texto que
ficou mais caro em bytes.

**É por isso que a métrica é bytes/token e não caracteres/token.** Comparar um
modelo em inglês contra um em japonês usando caracteres/token compara coisas
diferentes.

---

## Nível de byte

A saída para o problema da cobertura é descer um nível: trabalhar com os bytes,
não com os caracteres.

```
todo texto, em qualquer língua, é uma sequência de bytes
todo byte está em 0..255
→ V = 256, fixo, por definição
→ nenhum texto no universo é impossível de representar
→ o conceito de "token desconhecido" deixa de existir
```

### Por que exatamente 256

Um byte são 8 bits, cada bit com 2 estados independentes:

```
1 bit  →  2¹ =   2 valores
2 bits →  2² =   4 valores
8 bits →  2⁸ = 256 valores     0 a 255
```

Não é um ajuste nem uma escolha. É a **enumeração completa** do que um byte pode
ser. Não existe um 257º valor.

**Por que não menos?** Você poderia usar só os bytes presentes no corpus, mas aí
perde exatamente a garantia que motivou descer para bytes. Qual byte cortaria?
Omita o 195 e perdeu `'ç'`, `'ã'`, `'é'` — todo o português. E a economia é
irrelevante: num `V = 50.257`, os 256 bytes são 0,5% da tabela.

**Por que não mais?** Porque não há o que colocar lá. Ids ≥ 256 não correspondem
a byte nenhum — a menos que você **defina** o que significam. É exatamente isso
que o BPE faz.

### A diferença conceitual

```
V = 65    ← inventário:  o que por acaso apareceu neste corpus
V = 256   ← enumeração:  o que pode existir, em qualquer corpus, para sempre
```

Um é empírico e frágil. O outro é fechado por definição: 256 hoje, 256 daqui a
cem anos, 256 em japonês, 256 num arquivo binário corrompido.

*(Nota histórica: o byte ter 8 bits não é lei matemática — máquinas antigas usaram
6, 7 e 9 bits. Convergiu para 8 com o IBM System/360 nos anos 60. Então 256 é
constante da realidade computacional, não da matemática.)*

---

## O dilema que sobra

Nível de byte resolve a cobertura, mas piora a compressão:

```
'coração'   em caracteres  →  7 tokens
'coração'   em bytes       →  9 tokens
```

A sequência ficou **mais longa**, e `n` é o termo quadrático.

| abordagem | `V` | bytes/token | cobertura |
|---|---|---|---|
| caractere | 65 | 1.0 | só o corpus de treino |
| byte | 256 | 1.0 | **universal** |
| BPE (GPT-2) | 50.257 | ~4.0 | **universal** |

O BPE resolve os dois ao mesmo tempo: começa com os 256 bytes (cobertura
garantida) e **compra compressão** fundindo pares frequentes, subindo `V` de
forma controlada.

Esses `~4.0` do GPT-2 são o prêmio: 4× menos tokens que bytes crus, logo **16×
menos trabalho na atenção**.

→ Continua em [02 — BPE](02_bpe.md).
