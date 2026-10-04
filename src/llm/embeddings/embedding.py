"""Tabela de embeddings: de id inteiro para vetor em R^d.

O primeiro parametro APRENDIDO do modelo. Comeca aleatorio e e ajustado por
gradiente descendente junto com todo o resto.
"""

import numpy as np


class Embedding:
    """Matriz (vocab_size, d_model) onde a linha i e o vetor do token i.

    Atributos:
        weight: np.ndarray de forma (vocab_size, d_model)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        std: float = 0.02,
        seed: int = 0,
    ) -> None:
        """Inicializa a tabela com ruido gaussiano de media 0 e desvio `std`.

        Dicas de NumPy:
            rng = np.random.default_rng(seed)       gerador reproduzivel
            rng.normal(loc, scale, size=(a, b))     matriz a x b, N(loc, scale)

        Guarde o resultado em self.weight.

        POR QUE ALEATORIO, E NAO ZEROS:

        Se todas as linhas comecassem iguais (zeros, uns, qualquer constante),
        todo token teria exatamente o mesmo vetor. Pior: receberiam o mesmo
        gradiente em todo passo de treino, e continuariam identicos para sempre.
        O modelo nunca conseguiria distinguir um token do outro.

        Isso se chama quebra de simetria. O ruido aleatorio e o que da a cada
        token um ponto de partida distinto para o gradiente trabalhar.

        POR QUE 0.02, E NAO 1.0:

        A escala do ruido e um dos ajustes mais sensiveis de uma rede. Valores
        grandes fazem as ativacoes explodirem camada apos camada; pequenos demais
        fazem o sinal sumir. 0.02 e o valor que o GPT-2 usa -- vamos entender a
        matematica por tras dessa escolha quando chegarmos na inicializacao de
        verdade. Por ora, aceite como convencao.
        """
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        """V -- numero de tokens distintos. Dica: self.weight.shape da (V, d)."""
        raise NotImplementedError

    @property
    def d_model(self) -> int:
        """d -- dimensao de cada vetor de token."""
        raise NotImplementedError

    def forward(self, ids: list[int]) -> np.ndarray:
        """ids -> matriz (len(ids), d_model), uma linha por token.

        Exemplo:
            emb = Embedding(vocab_size=100, d_model=4)
            emb.forward([7, 3, 7]).shape   ->  (3, 4)

        Note que o id 7 aparece duas vezes e produz EXATAMENTE o mesmo vetor nas
        duas posicoes. Nesta camada o token ainda nao sabe onde esta na frase --
        posicao sera adicionada depois, por outro mecanismo.

        Dica de NumPy -- indexacao avancada (fancy indexing):

            A = np.array([[10, 11],
                          [20, 21],
                          [30, 31]])

            A[1]         -> [20, 21]           um inteiro: devolve UMA linha
            A[[2, 0, 2]] -> [[30, 31],         uma LISTA: devolve varias linhas,
                             [10, 11],          na ordem pedida, repetindo se
                             [30, 31]]          o indice se repetir

        Isso e exatamente o que forward precisa fazer. Uma linha de codigo.

        MAS LEMBRE DO QUE ISSO REALMENTE E:

        Indexar a linha i equivale a multiplicar o vetor one-hot de i por
        self.weight -- o 1 seleciona a linha, os zeros anulam o resto. Usamos
        indexacao porque materializar uma matriz (len(ids), V) de quase-zeros
        seria desperdicio, nao porque seja uma operacao diferente.

        Essa equivalencia e o que garante que o gradiente flui por aqui como por
        qualquer outra camada linear.
        """
        raise NotImplementedError
