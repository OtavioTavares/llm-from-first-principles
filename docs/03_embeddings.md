# 03 — Embeddings

> Código: `src/llm/embeddings/embedding.py` · Testes: `tests/test_embedding.py`
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

## Estado do código

| Item | Status |
|---|---|
| `Embedding.__init__` | ⬜ |
| `Embedding.vocab_size` / `.d_model` | ⬜ |
| `Embedding.forward(ids)` | ⬜ |
| Similaridade (produto escalar / cosseno) | ⬜ |
