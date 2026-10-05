"""Similaridade entre vetores: as duas operacoes geometricas fundamentais.

A identidade central:

    a . b  =  |a| * |b| * cos(theta)

O lado esquerdo e uma soma de produtos (aritmetica pura). O lado direito
envolve o angulo entre as setas (geometria pura). Sao iguais -- e e essa
igualdade que permite medir orientacao com hardware de multiplicar matrizes.

A mesma formula reaparece dentro da self-attention, onde Q @ K.T e uma matriz
de produtos escalares.
"""

import numpy as np


def norm(v: np.ndarray) -> float:
    """Comprimento do vetor v.

        |v| = raiz(v_1^2 + v_2^2 + ... + v_d^2)

    E Pitagoras generalizado: em 2D, |[3, 4]| = raiz(9 + 16) = 5. Em d dimensoes
    a formula e identica, so com mais termos.

    Dicas de NumPy -- operacoes ELEMENTO A ELEMENTO:
        v ** 2        eleva cada elemento ao quadrado, devolve um array
        np.sum(x)     soma todos os elementos, devolve UM numero
        np.sqrt(x)    raiz quadrada

    (Existe np.linalg.norm que faz isso pronto. Escreva na mao primeiro para
    ver a conta; depois confira que bate com a versao da biblioteca.)
    """
    return np.sqrt(np.sum(v**2))


def dot(a: np.ndarray, b: np.ndarray) -> float:
    """Produto escalar: multiplica coordenada a coordenada e soma tudo.

        a . b = a_1*b_1 + a_2*b_2 + ... + a_d*b_d

    Exemplo:
        dot([3, 4], [4, 3]) = 3*4 + 4*3 = 24

    Dois vetores entram, UM NUMERO sai -- por isso "escalar".

    Dica de NumPy:
        a * b     multiplica elemento a elemento, devolve um ARRAY
                  (nao e multiplicacao de matrizes!)
        np.sum()  reduz o array a um numero

    (Existe np.dot(a, b), e tambem o operador a @ b. Escreva na mao primeiro.)
    """
    return np.sum(a*b)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """O cosseno do angulo entre a e b.

                  a . b
        cos =  -----------
               |a| * |b|

    Isolando a ORIENTACAO e descartando o tamanho. O produto escalar cru mistura
    as duas coisas:

        [1, 0] . [1, 0] = 1
        [1, 0] . [2, 0] = 2      <- mesma direcao, resultado diferente

    Dividir pelas normas corrige isso: os dois casos acima dao cosseno 1.0.

    Escala do resultado:
        +1   mesma direcao
         0   perpendiculares
        -1   direcoes opostas

    Use as duas funcoes que voce ja escreveu acima.

    CASO DE BORDA: o vetor nulo (todos zeros) tem norma 0, e a divisao estoura.
    Geometricamente faz sentido -- um vetor sem comprimento nao aponta para
    lugar nenhum, entao o angulo nao existe. Devolva 0.0 nesse caso, que e a
    convencao usual ("nenhuma similaridade detectavel").
    """

    numerador = dot(a, b)
    denominador = norm(a) * norm(b)

    # Vetor nulo: sem comprimento, sem direcao, sem angulo. Dividir daria
    # 0/0 = nan, que contaminaria silenciosamente tudo adiante.
    if denominador == 0:
        return 0.0

    return numerador / denominador
