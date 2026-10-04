# 02 — Byte Pair Encoding

> Código: `src/llm/tokenizer/bpe.py`
> Testes: `tests/test_bpe.py`, `tests/test_bpe_train.py`, `tests/test_bpe_codec.py`
> Experimentos: `experiments/01_tokenization/explore_bpe.py`, `curva_v_n.py`
> Corpora: `data/domcasmurro.txt` (pt), `data/tinyshakespeare.txt` (en)
> Pré-requisito: [01 — Tokenização](01_tokenizacao.md)

## A ideia em uma frase

> Comece com os 256 bytes. Repetidamente, ache o par de símbolos adjacentes mais
> frequente e funda os dois num símbolo novo.

É só isso. Toda a sofisticação do GPT-2 em tokenização é esse loop rodando 50 mil
vezes.

## A aritmética que justifica a escolha gulosa

Fundir um par reduz o comprimento da sequência em exatamente o número de
ocorrências desse par, e aumenta `V` em exatamente 1:

```
cada merge:   n  diminui em (nº de ocorrências do par)
              V  aumenta em 1
```

Então "pegue o par mais frequente" significa literalmente **"compre a maior
redução de `n` possível por esta unidade de `V`"**. Cada merge é uma transação de
preço fixo (`+1` no `V`) e retorno variável. Você ordena por retorno.

---

## O estado do treino

Esta é a parte que confunde. Durante o treino existem **três** coisas vivas, e
elas se comportam de formas diferentes:

| Estrutura | Tipo | O que acontece a cada rodada |
|---|---|---|
| `ids` | `list[int]` | é **substituída inteira** por uma versão mais curta |
| `merges` | `dict[(int,int), int]` | **ganha uma entrada**: `par → id novo` |
| `vocab` | `dict[int, bytes]` | **ganha uma entrada**: `id novo → bytes` |

A chave de tudo está na primeira linha: **`ids` é reescrita a cada rodada, e a
rodada seguinte conta os pares sobre a versão já fundida.**

É por isso que um merge pode ter como pai outro merge. Na rodada 3, o algoritmo
não está mais olhando bytes — está olhando a sequência que as rodadas 1 e 2
produziram.

```
rodada 1:  conta pares de  [bytes crus]           → cria id 256
rodada 2:  conta pares de  [... com 256 dentro]   → cria id 257, pode usar 256
rodada 3:  conta pares de  [... com 256 e 257]    → cria id 258, pode usar ambos
```

## Uma rodada, em cinco passos

```python
for i in range(num_merges):
    stats  = get_stats(ids)            # 1. quais pares existem e quantos
    melhor = max(stats, key=stats.get) # 2. o mais frequente  (escolha gulosa)
    novo   = 256 + i                   # 3. o próximo id livre

    ids = merge(ids, melhor, novo)     # 4. a sequência encolhe   ← note o `ids =`
    self.merges[melhor] = novo         # 5a. registra a receita
    self.vocab[novo] = self.vocab[melhor[0]] + self.vocab[melhor[1]]  # 5b. registra o significado
```

Três detalhes que quebram tudo se errados:

**`max(stats, key=stats.get)`** — iterar um dicionário percorre as *chaves*. Sem o
`key=`, `max(stats)` compararia as tuplas entre si e devolveria o maior par em
ordem numérica, não o mais frequente.

**`ids = merge(...)`** — `merge` é uma função **pura**: devolve uma lista nova e
não toca na original. Chamá-la sem atribuir descarta o resultado. A sequência
nunca encolheria, o mesmo par venceria toda rodada, e o treino rodaria sem erro
produzindo lixo.

**`novo = 256 + i`** — ids são alocados sequencialmente. A rodada `i` sempre
recebe o id `256 + i`, nunca um buraco.

---

## Traço completo

Corpus: `"o rato roeu a roupa do rei de roma"` — 34 bytes.

```
ids iniciais (34 tokens):
[111, 32, 114, 97, 116, 111, 32, 114, 111, 101, 117, 32, 97, 32, 114, 111, 117,
 112, 97, 32, 100, 111, 32, 114, 101, 105, 32, 100, 101, 32, 114, 111, 109, 97]
```

### Rodada 1

```
top pares:  ' '+'r': 5    'o'+' ': 3    'r'+'o': 3    'a'+' ': 2
vencedor:   (32, 114) = ' ' + 'r'   →   id 256 = ' r'
n: 34 → 29
```

O par `' r'` venceu com 5 ocorrências — o corpus tem cinco palavras começando com
`r` (*rato, roeu, roupa, rei, roma*). O algoritmo descobriu isso sozinho,
contando.

### Rodada 2

```
top pares:  'o'+' r': 3    ' r'+'o': 3    ' '+'d': 2    ' r'+'a': 1
vencedor:   (111, 256) = 'o' + ' r'   →   id 257 = 'o r'
n: 29 → 26
```

**Aqui está o ponto central.** Os dois pares mais frequentes contêm o símbolo
`256`, que não existia na rodada 1. Eles só puderam ser contados porque o
`get_stats` desta rodada rodou sobre a sequência **já fundida**.

Repare também no **empate**: `'o'+' r'` e `' r'+'o'` ocorrem 3× cada. Venceu o
primeiro porque `get_stats` o inseriu primeiro no dicionário, e `get_stats` insere
na ordem em que os pares aparecem no texto. Dicionários do Python 3.7+ preservam
ordem de inserção, então `max` encontra esse primeiro. **O desempate é
determinístico sem nenhum código de desempate.**

### Rodada 3

```
top pares:  ' r'+'o': 2    ' '+'d': 2    'o r'+'a': 1    'a'+'t': 1
vencedor:   (256, 111) = ' r' + 'o'   →   id 258 = ' ro'
n: 26 → 24
```

### Rodada 4

```
top pares:  ' '+'d': 2    'o r'+'a': 1    'a'+'t': 1    't'+'o r': 1
vencedor:   (32, 100) = ' ' + 'd'   →   id 259 = ' d'
n: 24 → 22
```

Último merge com contagem > 1. `' d'` captura *do* e *de*.

### Rodadas 5 e 6 — a degeneração

```
rodada 5:  todos os pares têm contagem 1
           vencedor: (257, 97) = 'o r' + 'a'  →  id 260 = 'o ra'
           n: 22 → 21

rodada 6:  vencedor: (260, 116) = 'o ra' + 't'  →  id 261 = 'o rat'
           n: 21 → 20
```

A partir da rodada 5 **nenhum par se repete**. O algoritmo não tem mais padrão
para achar, então ele começa a engolir o começo da string letra por letra:
`'o r'` → `'o ra'` → `'o rat'`. Isso não é aprendizado — é memorização.

---

## `encode` e `decode`

```python
def decode(self, ids):
    bs = b"".join(self.vocab[i] for i in ids)
    return bs.decode("utf-8", errors="replace")

def encode(self, text):
    ids = list(text.encode("utf-8"))
    for par, novo in self.merges.items():
        ids = merge(ids, par, novo)
    return ids
```

### A assimetria do roundtrip

```
decode(encode(x)) == x        ✓  sempre
encode(decode(y)) == y        ✗  não para y arbitrário
```

`decode` precisa aceitar **qualquer** sequência de ids, não só as que `encode`
produziu. Um modelo generativo amostra ids de uma distribuição e pode emitir o
byte `195` (primeiro byte de `'ç'`) sem o `167` que o completa — bytes que não
são UTF-8 válido.

Por isso `errors="replace"`: troca o inválido por U+FFFD em vez de levantar
`UnicodeDecodeError`. Sem isso, streaming de geração é impossível — metade de um
caractere multi-byte na tela quebraria a aplicação.

### `encode` é o `train` sem a descoberta

| | `train` | `encode` |
|---|---|---|
| De onde vem o par | `max(stats, key=stats.get)` | `self.merges.items()` |
| O que faz com ele | `merge(...)` + registra | `merge(...)` |

É o mesmo loop. `train` **descobre** quais fusões aplicar; `encode` **já sabe**.
Consequência: `encode(texto_de_treino)` reproduz exatamente os ids com que o
`train` terminou, sem nenhuma coordenação explícita entre os dois métodos.

E a ordem de inserção de `self.merges` **é** o algoritmo — não há controle de
sequência em lugar nenhum do código. Um `sorted()` bem-intencionado em cima disso
não levanta exceção: só devolve ids piores, em silêncio.

---

## Overfitting do tokenizer

A degeneração acima é o mesmo fenômeno de overfitting, só que no tokenizer em vez
do modelo.

```
corpus de 34 bytes,     6 merges  →  decora o prefixo da string, generaliza zero
corpus de 1 MB,      1000 merges  →  aprende 'the', 'ing', 'tion', generaliza
```

Com corpus grande, os merges vencedores são padrões que **se repetem** — sufixos,
prefixos, palavras comuns. Esses transferem para texto novo. Com corpus pequeno,
qualquer sequência é "frequente" em termos relativos e o algoritmo memoriza ruído.

Extremo: treinar em `"banana"` com 4 merges produz o token `b'banana'` e
`bytes/token = 6.00`. Número lindo e inútil — só serve se a palavra aparecer de
novo.

> **A métrica honesta nunca é bytes/token no corpus de treino.** É bytes/token em
> texto que o tokenizer nunca viu.

### O overfitting, visto pelos tokens aprendidos

A natureza do que o algoritmo aprende muda sozinha ao longo do treino, sem
ninguém mandar. Em **Dom Casmurro**:

```
merges    0-10:  'e '  'a '  'o '  ', '  's '  'qu'  'er'  'ar'  'm '  'es'
merges  250-260: 'dell'  'ocê'  'anh'  'õ'  '... '  'tamb'  'cl'  'pen'  '! '
merges  750-760: 'noit'  'dest'  'moç'  'cons'  'pequen'  'me\n'  'baix'  'raz'
merges 1950-1960: 'a. Quando '  'morre'  'palavras '  'o\nque '  'Europ'  'religios'
```

| faixa | o que aprende | generaliza? |
|---|---|---|
| 0-50 | **morfologia portuguesa**: `qu`, `er`, `ar`, `es`, `e ` | sim, para qualquer português |
| ~250 | **radicais e acentos**: `tamb`(ém), `ocê`, `anh`, `õ` | sim, em boa parte |
| ~750 | **palavras do corpus**: `noit`, `moç`, `pequen` | parcialmente |
| ~2000 | **frases inteiras**: `a. Quando `, `palavras `, `o\nque ` | não |

O mesmo fenômeno fica ainda mais nítido em **Tiny Shakespeare**, porque lá o
algoritmo memoriza nomes próprios e formatação:

```
merges    0-10:  'e '  'th'  't '  's '  'd '  ', '  'ou'  'er'  'in'  'y '
merges  250-260: 'ble '  'RIOLANUS:\n'  'vi'  'gra'  '--'  'First '  '.\n\nB'
merges  750-760: ':\nAnd '  'brother'  'Clar'  'than '  'MARCIUS:\n'
merges 1950-1960: 'Give '  'year'  'far '  'fear '  'hell'  'ould\n'  ',\nBy '
```

`'RIOLANUS:\n'` é um token inteiro — o algoritmo decorou **Coriolanus**, um
personagem desta peça específica. E `'.\n\nB'` decorou a *formatação*: quebra
dupla de linha antes do nome do próximo personagem.

Esses tokens valem zero em qualquer outro texto do mundo, e cada um ocupa um slot
de `V` que poderia ter ido para morfologia. É isso que o gap entre treino e teste
mede — quando o split é honesto.

---

## As duas estruturas, e por que duas

| | `merges` | `vocab` |
|---|---|---|
| Tipo | `dict[(int,int), int]` | `dict[int, bytes]` |
| Lê como | "par `(a,b)` vira o id `c`" | "o id `c` significa estes bytes" |
| Usada em | **`encode`** | **`decode`** |
| Ordem importa? | **Sim** | Não |

São duas visões do mesmo conhecimento, otimizadas para direções opostas.

### `vocab` é uma árvore

```
vocab[97]  = b'a'                         ← folha: byte cru
vocab[114] = b'r'                         ← folha: byte cru
vocab[256] = vocab[32]  + vocab[114]  = b' r'
vocab[257] = vocab[111] + vocab[256]  = b'o r'     ← pai é outro merge
vocab[261] = vocab[260] + vocab[116]  = b'o rat'
```

Isso é **indução**: a base são os 256 bytes, o passo é cada merge. Como todo nó
novo é construído a partir de nós que já existem, **toda folha é um byte**.

Consequência: `decode` nunca pode falhar. Não existe id indecifrável. É a garantia
que o `CharTokenizer` não tinha.

### Em `merges`, a ordem é parte do dado

`merges` não é só um mapa — é uma **sequência ordenada de instruções**. O `encode`
precisa reaplicar as fusões na mesma ordem em que foram aprendidas.

Por quê: na rodada 2 o par `('o', ' r')` só existia *porque* a rodada 1 já tinha
criado `' r'`. Aplicar o merge 257 antes do 256 é impossível — o símbolo `256` não
estaria lá para ser encontrado. A ordem codifica a dependência.

Por isso `self.merges` nunca deve ser ordenado, convertido para `set`, ou
reconstruído. Dicionários do Python 3.7+ preservam ordem de inserção, e o código
depende disso.

---

## Guloso não é ótimo

BPE escolhe o melhor par **da rodada atual**. Isso não garante a melhor sequência
de 1000 merges — fundir um par ligeiramente pior agora poderia abrir fusões muito
melhores depois.

Encontrar o conjunto ótimo de merges é caro demais para valer a pena, e o guloso
funciona bem o suficiente na prática. É uma escolha de engenharia consciente, não
um descuido.

## Retornos decrescentes

A frequência dos pares num texto natural segue aproximadamente uma lei de potência
(Zipf): o par nº 1 é muito mais comum que o nº 100, que é muito mais comum que o
nº 10.000. Como cada merge custa sempre `+1` no `V` mas rende cada vez menos:

### Medição real

`experiments/01_tokenization/curva_v_n.py` — treino em 300 KB de **Dom Casmurro**
(Machado de Assis), teste nos últimos 80 KB (nunca vistos):

```bash
uv run python experiments/01_tokenization/curva_v_n.py domcasmurro.txt
```

| merges | `V` | b/tok treino | b/tok teste | gap | ganho | custo atenção |
|---|---|---|---|---|---|---|
| 0 | 256 | 1.00 | 1.00 | +0.0% | — | 1.00× |
| 50 | 306 | 1.51 | 1.47 | +2.6% | +0.47 | 0.46× |
| 100 | 356 | 1.76 | 1.69 | +3.7% | +0.22 | 0.35× |
| 250 | 506 | 2.16 | 2.04 | +5.9% | +0.35 | 0.24× |
| 500 | 756 | 2.54 | 2.36 | +7.6% | +0.32 | 0.18× |
| 1.000 | 1.256 | 3.00 | 2.83 | +6.1% | +0.47 | 0.12× |
| 1.500 | 1.756 | 3.32 | 3.10 | +7.2% | +0.27 | 0.10× |
| 2.000 | 2.256 | 3.57 | 3.37 | +6.1% | +0.26 | 0.09× |

*(`custo atenção` = `(n/n₀)²` relativo ao tokenizer de bytes puro.)*

> ⚠️ A coluna `gap` deste corpus **não é confiável** — o split vaza. Veja
> [a seção sobre isso](#armadilha-split-de-treinoteste-que-vaza). As colunas de
> compressão e ganho continuam válidas.

**Eficiência por merge, nos dois extremos:**

```
merges    0 →   50:  +0.47 b/tok  ÷   50 merges  =  0.0094 por merge
merges 1500 → 2000:  +0.26 b/tok  ÷  500 merges  =  0.00052 por merge
                                                    └─ 18× menos eficiente
```

### O prêmio na atenção

O custo da atenção é proporcional a `n²`. Comparando dois tokenizers no **mesmo
texto**, o número de bytes é fixo e `n = bytes / b`, onde `b` é bytes por token:

```
  n      bytes / b        b₀              b  = bytes/token do BPE
 ──  =  ──────────  =  ─────              b₀ = bytes/token da referência (bytes crus)
  n₀    bytes / b₀        b

custo_relativo = (n/n₀)² = (b₀/b)²
```

Ou seja: **o ganho na atenção é o quadrado da taxa de compressão.**

A 2.000 merges, `b = 3.37`:

```
custo_relativo = (1.00 / 3.37)² = 0.088   →   atenção 11.4× mais barata
```

Em números concretos, num documento de 1.000 bytes:

```
bytes crus:  n = 1000 / 1.00 = 1000 tokens  →  1000² = 1.000.000 unidades de trabalho
BPE 2000:    n = 1000 / 3.37 =  297 tokens  →   297² =    88.000 unidades
```

Pagou `V` de 256 → 2.256 (linear, 8.8×) para ganhar 11.4× na atenção
(quadrático). É a assimetria da [Lição 1](01_tokenizacao.md) rendendo juros: cada
ganho em compressão é elevado ao quadrado no termo que mais pesa.

*(Para comparação, o GPT-2 comprime ~4×, logo atenção ~16× mais barata que bytes
crus.)*

É a curva achatando que explica por que o GPT-2 parou em `V = 50.257` e não em
1.000 nem em 1.000.000. Depois de certo ponto você paga `V` cheio por quase nada.

*(`50.257 = 50.000 merges + 256 bytes + 1 token especial `<|endoftext|>`.)*

É essa curva achatando que explica por que o GPT-2 parou em `V = 50.257` e não em
1.000 nem em 1.000.000. Depois de certo ponto você paga `V` cheio por quase nenhum
ganho em `n`.

*(`50.257 = 50.000 merges + 256 bytes + 1 token especial `<|endoftext|>`.)*

---

## Custo computacional

```
por rodada:  get_stats(ids)   O(n)
           + merge(ids, ...)  O(n)

total:  O(num_merges × n)
```

Com 1,1 MB e 1000 merges, são ~10⁹ operações em Python puro — minutos de espera.

O gargalo real é `get_stats` **recontar a lista inteira** a cada rodada, quando só
a vizinhança das posições fundidas mudou. A correção séria é manter as contagens
incrementalmente, com uma estrutura de dados bem mais complicada.

Micro-otimizar `merge` não resolve: ele é metade do trabalho, então o teto de ganho
é 50%, por melhor que seja a otimização. Saída prática para estudo: treinar numa
fatia menor do corpus.

> **Lição transversal:** em CPython, o custo dominante é o número de operações no
> nível do *bytecode*, não as alocações de memória. `zip` e fatiamento rodam em C;
> um `while` com indexação manual roda no interpretador e pode ser **mais lento**
> apesar de alocar menos. Medir > raciocinar. A mesma lógica vai justificar
> escrever atenção como produto de matrizes NumPy em vez de loops.

---

## Casos de borda

**Corpus sem pares suficientes.** Se `ids` encolhe até 1 elemento, `get_stats`
devolve `{}` e `max({})` levanta `ValueError: max() arg is an empty sequence` —
mensagem que não explica nada para quem chamou. Decisões possíveis: parar cedo em
silêncio (quebra a promessa `len(vocab) == vocab_size`), parar cedo com aviso (o
que a maioria das implementações faz), ou levantar erro explicativo. Na prática
quase nunca acontece com corpus real.

**`vocab_size = 256`.** Zero merges. BPE degenera num tokenizer de bytes puro —
cobertura universal, compressão nenhuma. Caso válido e útil como baseline.

**`vocab_size < 256`.** Sem sentido: você não pode representar menos que o
alfabeto completo de bytes sem perder cobertura.

**Texto não-ASCII.** Não é caso especial — veja a seção seguinte.

---

## O BPE redescobre o UTF-8 sozinho

Treinando em **Dom Casmurro** (Machado de Assis, 389 KB), 2.000 merges:

```
merge   10: [195, 163] -> 'ã'       merge  253: [195, 181] -> 'õ'
merge   37: [195, 167] -> 'ç'       merge  305: [194, 171] -> '«'
merge   41: [195, 169] -> 'é'       merge  600: [195, 137] -> 'É'
merge   54: [195, 161] -> 'á'       merge  631: [195, 180] -> 'ô'
merge  121: [195, 179] -> 'ó'       merge  724: [195, 173] -> 'í'
merge  138: [195, 186] -> 'ú'       merge  999: [195, 178] -> 'ò'
merge  198: [195, 170] -> 'ê'
```

14 caracteres multi-byte reconstituídos, **na ordem exata da frequência deles em
português**: `ã` primeiro (não, são, -ção), `ò` por último porque é raro.

Ninguém ensinou UTF-8 ao algoritmo. Ele só notou que o byte 195 e o byte 163
sempre aparecem juntos. A estrutura da codificação emerge da contagem de pares.

No `merge 10`, o `ã` já compete de igual para igual com `'e '` e `'qu'`.

---

## Armadilha: split de treino/teste que vaza

A mesma medição nos dois corpora dá resultados opostos:

| | 50 merges | 2.000 merges | comportamento |
|---|---|---|---|
| Tiny Shakespeare (en) | +1.0% | **+25.2%** | cresce monotonicamente |
| Dom Casmurro (pt) | +2.6% | **+6.1%** | parado, oscilando |

Parece que o BPE "overfitou menos" em português. Não foi isso. Quem aparece em
cada fatia (ocorrências por 10 mil caracteres):

| Shakespeare | treino | teste | | Dom Casmurro | treino | teste |
|---|---|---|---|---|---|---|
| `CORIOLANUS` | 5.00 | **0.00** | | `Escobar` | 2.40 | **5.00** |
| `GLOUCESTER` | 5.43 | **0.00** | | `Ezequiel` | 0.27 | **5.25** |
| `RICHARD` | 3.60 | **0.00** | | `Bentinho` | 1.70 | **0.75** |

Tiny Shakespeare é uma **coletânea de peças** — a fatia de teste é outra peça,
com outros personagens. Separação real: os tokens memorizados (`'RIOLANUS:\n'`)
valem zero no teste, e o gap de 24% mede generalização de verdade.

Dom Casmurro é **um romance só** — os mesmos personagens do começo ao fim.
Cortar o livro em dois não produz conjuntos independentes.

> **Lição:** um split por posição dentro de um documento homogêneo não é um split.
> O gap de 6.1% não mede generalização; mede o quanto a segunda metade parece com
> a primeira. A não-monotonicidade (7.6% → 6.1% → 7.2% → 6.1%) é o sinal de
> alerta: ruído onde deveria haver tendência.

Corolário: **não compare bytes/token entre corpora com splits de qualidades
diferentes.** O 3.37 do português foi medido num teste que vaza; o 2.83 do inglês,
não. Os números não são comparáveis.

Isso não é um detalhe de tokenização — é metodologia. O mesmo erro ao separar
dados para treinar o modelo produziria uma curva de loss de validação bonita e
mentirosa. Split honesto separa por **documento**, não por posição.

---

## Estado do código

| Função | Status |
|---|---|
| `get_stats(ids)` | ✅ |
| `merge(ids, pair, new_id)` | ✅ |
| `BPETokenizer.train(text, vocab_size)` | ✅ |
| `BPETokenizer.encode(text)` | ✅ |
| `BPETokenizer.decode(ids)` | ✅ |
| Pré-tokenização por regex | ⬜ |

---
