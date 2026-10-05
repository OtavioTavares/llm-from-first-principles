# 03 — Embeddings

> Código: `src/llm/embeddings/embedding.py`, `src/llm/embeddings/similarity.py`
> Testes: `tests/test_embedding.py`, `tests/test_similarity.py`
> Pré-requisito: [02 — BPE](02_bpe.md)

Aqui a matemática muda de natureza: sai combinatória, entra álgebra linear.

## O problema: ids são rótulos, não quantidades

O tokenizer entrega `[257, 97, 116]`. Esses números são **nomes**, não medidas. O
id `257` só reflete em que rodada aquele merge foi criado — não há nada de
"maior" ou "mais intenso" nele.

Mas toda operação que uma rede faz trata números como quantidades:

```
257 / 2 = 128.5      ← o token 128.5 não existe
(97 + 116) / 2       ← a "média" entre dois tokens não significa nada
257 > 97             ← verdadeiro numericamente, vazio semanticamente
```

Alimentar ids crus numa rede impõe uma **ordem e uma distância falsas** sobre
símbolos que não têm nenhuma das duas.

## Primeira tentativa: one-hot

Represente o id `i` como um vetor de tamanho `V`, com `1` na posição `i`:

```
V = 5

id 0  →  [1, 0, 0, 0, 0]
id 1  →  [0, 1, 0, 0, 0]
id 3  →  [0, 0, 0, 1, 0]
```

Isso corrige o problema: a distância entre **quaisquer** dois tokens distintos é
a mesma (`√2`), e nenhuma ordem é insinuada. Honesto — e inviável por dois
motivos.

**É enorme.** Com `V = 50.257`, cada token vira 50 mil números dos quais 50.256
são zero.

**Não cabe similaridade.** Todo par distinto está à mesma distância. "gato" e
"cachorro" são tão diferentes quanto "gato" e "parafuso". Não é que o modelo não
saiba — é que a representação **não tem onde guardar** essa informação.

## A solução: uma tabela aprendida

Uma matriz `E` de forma `(V, d)`. A linha `i` é o vetor do token `i`:

```
            d colunas
          ┌                            ┐
      0   │  -0.02   0.01   0.03  -0.01│
      1   │   0.00  -0.04   0.02   0.01│    E tem forma (V, d)
      2   │   0.03   0.02  -0.01   0.00│
     ...  │   ...                      │
    V-1   │   0.01  -0.03   0.00   0.02│
          └                            ┘
```

`encode("o rato")` → `[257, 97, 116]` → pega as linhas 257, 97 e 116 → matriz
`(3, d)`.

| | one-hot | embedding |
|---|---|---|
| tamanho por token | `V` (50.257) | `d` (768) |
| similaridade | impossível de expressar | distância e ângulo em `R^d` |
| origem dos valores | fixos por construção | **aprendidos** |

---

## De onde vem o `d`

**Você escolhe.** É um hiperparâmetro, exatamente como `num_merges` foi no BPE.

Vale relembrar a distinção da [Lição 1](01_tokenizacao.md): `V = 65` no
`CharTokenizer` era um *inventário* — você descobria. `V = 256 + num_merges` no
BPE era uma *decisão* — você girava o botão. O `d` é do segundo tipo. Ninguém
calcula, ninguém deriva.

**O que `d` significa:** quantos números você permite para descrever cada token.
É o orçamento de descrição.

```
d = 1     cada token é um ponto numa reta      → um único eixo de variação
d = 2     cada token é um ponto num plano      → dois eixos
d = 768   cada token é um ponto em R^768       → 768 eixos independentes
```

Pense em descrever uma pessoa. Com um número só você diz "altura" e nada mais —
duas pessoas de mesma altura ficam indistinguíveis. Com três, "altura, peso,
idade". Com 768, cabe muito mais nuance.

> **Onde a analogia quebra:** ninguém atribui significado aos eixos. Não existe
> "a dimensão 42 é formalidade". O modelo usa os 768 eixos como quiser, e eles
> são tipicamente ininterpretáveis individualmente. Você fornece o espaço; o
> treino decide como ocupá-lo.

Valores típicos:

| contexto | `d` |
|---|---|
| brinquedo / estudo | 32 – 128 |
| GPT-2 small | 768 |
| Llama 3 8B | 4.096 |
| GPT-3 | 12.288 |

O trade-off: `d` maior dá mais capacidade, mas custa `V · d` parâmetros e mais
computação em **toda** camada seguinte.

### O custo, em números

```
GPT-2 small:  V = 50.257   d = 768

parâmetros só na tabela:  50.257 × 768 = 38.597.376
modelo inteiro:                          124.000.000
                                          └─ embeddings são 31% do modelo
```

Quase um terço dos parâmetros do GPT-2 small são a tabela de embeddings — antes
de qualquer camada de atenção existir.

---

## De onde vêm os floats

**São números aleatórios. No início, não significam nada.**

Os `-0.02`, `0.01`, `0.03` do diagrama acima são ilustração de ruído — não têm
significado nem no exemplo nem num modelo real recém-inicializado.

```python
rng.normal(0, 0.02, size=(vocab_size, d_model))
```

Isso sorteia `V × d` números de uma gaussiana centrada em zero. No instante da
inicialização, o vetor do token `257` é tão arbitrário quanto jogar moedas.

> **A tabela começa como lixo e vira conhecimento.** Esse é o ponto inteiro.

### Como o lixo vira conhecimento

```
1. o modelo lê uns tokens e tenta prever o próximo
2. erra
3. o erro é propagado de volta até E
4. cada número de E é empurrado um tiquinho na direção que teria diminuído o erro
5. repete ~10⁶ vezes
```

Depois de milhões de ajustes minúsculos, o ruído foi esculpido em estrutura.

**Por que a similaridade emerge:** tokens que aparecem em contextos parecidos
recebem empurrões parecidos. "rei" e "rainha" ocorrem em posições similares nas
frases, então o gradiente os move em direções similares, e eles terminam perto no
espaço. Ninguém programou isso — é consequência de otimizar a predição.

*(Isso tem nome: **hipótese distribucional** — o significado de uma palavra é
dado pelas companhias que ela mantém.)*

É o mesmo tipo de emergência que o BPE já mostrou: lá, a estrutura do UTF-8
apareceu só de contar pares; aqui, a semântica aparece só de minimizar erro de
predição.

### O exemplo que cabe na cabeça

`V = 5`, `d = 2`. O vocabulário inteiro são **cinco pontos num plano** — dá para
desenhar numa folha.

```
no início (aleatório):            depois do treino (hipotético):

      gato                              gato cachorro      ← animais
  parafuso
           rei                      parafuso               ← sozinho
   cachorro
      rainha                                 rei rainha    ← realeza
```

Mesmos cinco pontos, mesmas duas coordenadas por ponto. O que mudou foram os
**valores** das coordenadas, empurrados pelo gradiente até a geometria refletir o
uso das palavras.

Com `d = 2` há pouquíssimo espaço e tudo colide. Com `d = 768` sobra espaço para
muitos agrupamentos simultâneos — por tema, por classe gramatical, por registro,
por idioma.

---

## A identidade que amarra tudo: buscar linha É multiplicar matriz

Isto não é analogia, é igualdade algébrica. Multiplique o one-hot do token `i`
por `E`:

```
                      ┌ e₀₀  e₀₁  e₀₂ ┐
[0, 0, 1, 0]    ×     │ e₁₀  e₁₁  e₁₂ │   =   [e₂₀, e₂₁, e₂₂]
 └── one-hot          │ e₂₀  e₂₁  e₂₂ │        └── exatamente a linha 2
     do id 2          └ e₃₀  e₃₁  e₃₂ ┘
   forma (1, V)            forma (V, d)           forma (1, d)
```

O `1` na posição 2 seleciona a linha 2; os zeros anulam o resto.

**Por que isso importa:** multiplicação de matrizes é diferenciável. A camada de
embedding não é um caso especial que precisa de regra de gradiente própria — é
uma camada linear comum. Na prática implementamos como indexação apenas porque
materializar uma matriz `(n, V)` de quase-zeros seria desperdício absurdo, não
porque seja uma operação diferente.

Está verificado numericamente em `test_indexar_equivale_a_multiplicar_pelo_one_hot`.

---

## Duas decisões de inicialização

### Por que aleatório, e não zeros

Se todas as linhas começassem iguais — zeros, uns, qualquer constante — todo
token teria o mesmo vetor. Pior: receberiam o **mesmo gradiente** em todo passo
de treino, e continuariam idênticos para sempre. O modelo jamais distinguiria um
token do outro.

Isso se chama **quebra de simetria**. O ruído aleatório é o que dá a cada token um
ponto de partida distinto para o gradiente trabalhar.

### Por que `0.02`, e não `1.0`

A escala do ruído é um dos ajustes mais sensíveis de uma rede: valores grandes
fazem as ativações explodirem camada após camada; pequenos demais fazem o sinal
sumir. `0.02` é o que o GPT-2 usa.

A matemática por trás dessa escolha vem na lição de inicialização. Por ora, é
convenção.

---

## Uma coisa que esta camada NÃO faz

```python
emb.forward([7, 3, 7])
# a linha 0 e a linha 2 são idênticas
```

O token `7` produz exatamente o mesmo vetor nas duas posições. **Esta camada não
sabe nada sobre ordem.** Para ela, "o rato roeu" e "roeu rato o" são o mesmo
conjunto de vetores.

Informação de posição é adicionada depois, por outro mecanismo (RoPE, no nosso
roadmap). Vale guardar: é uma lacuna deliberada, não um descuido.

---

## Similaridade: dando sentido aos vetores

> Código: `src/llm/embeddings/similarity.py` · Testes: `tests/test_similarity.py`

Sem uma medida de similaridade, a tabela é só um monte de números. Duas operações
geométricas resolvem isso.

### Norma: o comprimento de um vetor

Em 2D, `[3, 4]` é uma seta da origem até o ponto (3, 4), e seu comprimento sai de
Pitágoras:

```
        (3,4)
         ╱│
      5 ╱ │ 4          |a| = √(3² + 4²) = √25 = 5
       ╱  │
      ╱___│
        3
```

Em `d` dimensões a fórmula é **idêntica**, só com mais termos:

```
|a| = √(a₁² + a₂² + ... + a_d²)
```

> Você não consegue *visualizar* `R^768`, mas a aritmética não muda. É por isso
> que álgebra linear funciona: as fórmulas não sabem quantas dimensões existem.
> (`test_norm_em_muitas_dimensoes` usa cem `1`s e espera `√100 = 10`.)

### Produto escalar

Multiplique coordenada a coordenada e some:

```
a = [3, 4]
b = [4, 3]       a · b = 3×4 + 4×3 = 24
```

Dois vetores entram, **um número** sai — daí "escalar".

### A ponte entre álgebra e geometria

```
a · b  =  |a| · |b| · cos(θ)
└─┬──┘    └───────┬───────┘
 soma de         ângulo entre
 produtos        as duas setas
```

O lado esquerdo é aritmética pura — você calcula sem saber o que é ângulo. O
direito é geometria pura. **São iguais.**

Essa igualdade é o que permite medir *orientação* com uma soma de multiplicações,
e é a razão de redes neurais conseguirem fazer geometria em hardware que só sabe
multiplicar matrizes.

Verificando: `|a| = |b| = 5`, logo `cos θ = 24/25 = 0.96`, ou θ ≈ 16°. Confere com
a intuição — `[3,4]` e `[4,3]` são espelhadas em torno da diagonal.

### Por que cosseno, e não o produto escalar cru

O produto escalar mistura **orientação** com **tamanho**:

```
[1, 0] · [1, 0]  =  1
[1, 0] · [2, 0]  =  2      ← mesma direção, resultado dobrado
```

Usar o produto escalar cru como similaridade faria tokens com vetores grandes
parecerem "mais similares a tudo" — artefato, não semântica. Dividir pelas normas
remove o tamanho e isola a direção:

```
                a · b
cos(θ)  =  ─────────────
              |a| · |b|

cos = +1    mesma direção        ───→  ───→
cos =  0    perpendiculares      ───→   ↑
cos = -1    direções opostas     ───→  ←───
```

> Guarde `a · b = |a||b|cos(θ)`. Ela reaparece **literalmente igual** dentro da
> self-attention: `Q @ Kᵀ` é uma matriz de produtos escalares medindo quanto cada
> posição se importa com cada outra.

### O caso do vetor nulo

```python
if denominador == 0:
    return 0.0
```

`norm` do vetor nulo é `0`, e `0/0` é **indeterminado** — NumPy devolve `nan`, sem
levantar exceção.

`nan` é perigoso porque **contamina**: qualquer conta com `nan` dá `nan`, então um
único vetor nulo transforma a loss inteira em `nan` e nenhum gradiente faz sentido
mais. Pior, `nan == nan` é `False`, então comparações não pegam.

*(Comparar float com `==` aqui é a exceção à regra do `np.allclose`: zero é
exatamente representável, e `norm` de um vetor de zeros dá exatamente `0.0`, não
"quase zero". A regra do `allclose` vale para resultados de contas acumuladas.)*

---

## Em dimensão alta, tudo é perpendicular a tudo

Este é o resultado mais importante da lição, e é contraintuitivo.

Pares de vetores aleatórios, 3.000 amostras por dimensão:

| `d` | desvio do cosseno | `1/√d` | `\|cos\|` máximo |
|---|---|---|---|
| 2 | 0.7088 | 0.7071 | 1.000 |
| 4 | 0.4954 | 0.5000 | 0.993 |
| 16 | 0.2454 | 0.2500 | 0.805 |
| 64 | 0.1216 | 0.1250 | 0.423 |
| 128 | 0.0887 | 0.0884 | 0.324 |
| 768 | 0.0367 | 0.0361 | 0.129 |
| 4.096 | 0.0157 | 0.0156 | 0.058 |

O desvio medido acompanha **`1/√d`** em três ordens de grandeza.

Em `R²`, dois vetores aleatórios têm cosseno espalhado por todo o intervalo
`[-1, 1]` — o máximo observado é `1.000`. Em `R^768`, nenhum dos 3.000 pares
passou de `0.129`. Todos quase ortogonais.

Duas consequências que vão importar:

**É a linha de base — o zero da régua.** Este é o estado de "nenhum conhecimento".
Depois do treino, "rei" e "rainha" terão cosseno muito acima de `1/√d`, e é por
contraste com esse caos que a estrutura fica visível. Medir o ruído agora é o que
torna o sinal mensurável depois.

**É o que torna `d` grande útil.** Como vetores aleatórios já nascem quase
ortogonais, há espaço de sobra para o gradiente posicionar milhares de
agrupamentos distintos sem que se atrapalhem. Em `d = 2` tudo colide; em `d = 768`
cabe muita estrutura simultânea — por tema, por classe gramatical, por registro,
por idioma.

---

## Estado do código

| Item | Status |
|---|---|
| `Embedding.__init__` | ✅ |
| `Embedding.vocab_size` / `.d_model` | ✅ |
| `Embedding.forward(ids)` | ✅ |
| `similarity.norm` / `.dot` / `.cosine_similarity` | ✅ |

```python
class Embedding:
    def __init__(self, vocab_size, d_model, std=0.02, seed=0):
        rng = np.random.default_rng(seed)
        self.weight = rng.normal(0, std, size=(vocab_size, d_model))

    @property
    def vocab_size(self): return self.weight.shape[0]

    @property
    def d_model(self): return self.weight.shape[1]

    def forward(self, ids):
        return self.weight[ids]
```

## Três armadilhas de NumPy que já apareceram

**Eixo 0 é linha, eixo 1 é coluna.** Em `weight`, linha = token, coluna =
dimensão. Quase todo bug de NumPy daqui pra frente vai ser eixo trocado, e a
mensagem costuma ser só `shapes not aligned` sem dizer qual.

**Inteiro achata, lista preserva.**

```python
A[2]          → forma (d,)      ← perdeu uma dimensão
A[[2]]        → forma (1, d)    ← continua 2-D
A[[]]         → forma (0, d)    ← lista vazia funciona de graça
```

**Nunca compare floats com `==`.** A multiplicação `one_hot @ weight` soma `V`
produtos e pode diferir do valor original na 16ª casa decimal. Use
`np.allclose(a, b)`, que compara com tolerância.
