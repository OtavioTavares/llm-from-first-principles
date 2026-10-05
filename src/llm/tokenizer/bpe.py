"""Byte Pair Encoding.

Comeca com os 256 bytes possiveis e funde pares frequentes para criar
simbolos novos. Diferente do CharTokenizer, aqui V e uma DECISAO
(V = 256 + num_merges), nao uma consequencia do corpus.

Este arquivo comeca pelos dois primitivos do algoritmo. A classe vem depois.
"""

from pathlib import Path

CABECALHO = "llm-bpe v1"


def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """Conta quantas vezes cada par ADJACENTE aparece em `ids`.

    Exemplo:
        get_stats([1, 2, 3, 1, 2])  ->  {(1, 2): 2, (2, 3): 1, (3, 1): 1}

    Numa lista de n elementos existem n-1 pares adjacentes: as posicoes
    (0,1), (1,2), ..., (n-2, n-1). Listas de 0 ou 1 elemento nao tem par
    nenhum, e o resultado e um dicionario vazio.

    Pares que se sobrepoem CONTAM os dois. Em [7, 7, 7] o par (7,7) aparece
    nas posicoes (0,1) e (1,2), entao a contagem e 2 -- mesmo que depois so
    seja possivel fundir um deles. Contar e uma coisa, fundir e outra.
    """

    all_pairs = zip(ids, ids[1:])

    count_pair = {}
    for pair in all_pairs:
        count_pair[pair] = count_pair.get(pair, 0) + 1
    
    return count_pair

    

def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Substitui toda ocorrencia de `pair` em `ids` pelo simbolo `new_id`.

    Exemplo:
        merge([1, 2, 3, 1, 2], (1, 2), 99)  ->  [99, 3, 99]

    Devolve uma lista NOVA; nao modifique `ids` no lugar.

    ATENCAO -- o detalhe onde todo mundo tropeca: apos casar um par voce
    consumiu DOIS elementos, entao o proximo candidato comeca duas posicoes
    adiante, nao uma. Em [7, 7, 7] fundindo (7,7) -> 99, o resultado correto
    e [99, 7] e nao [99, 99]: o 7 do meio foi engolido pelo primeiro par e
    nao pode ser reutilizado.

    Por isso um `for` sobre os indices nao resolve -- voce precisa controlar
    o avanco do indice voce mesmo (pense num `while`).
    """

    saida = []
    pular = False
    for a, b in zip(ids, ids[1:] + [None]):   # None = sentinela no fim
        if pular:
            pular = False
            continue
        if (a, b) == pair:
            saida.append(new_id)
            pular = True
        else:
            saida.append(a)
    return saida



class BPETokenizer:
    """Tokenizer BPE em nivel de byte.

    Estado apos o treino:
        self.merges: dict[tuple[int, int], int]
            par fundido -> id novo. A ORDEM DE INSERCAO importa: no encode as
            fusoes precisam ser reaplicadas na mesma ordem em que foram
            aprendidas. Dicionarios do Python 3.7+ preservam essa ordem.

        self.vocab: dict[int, bytes]
            id -> os bytes que ele representa. Comeca com os 256 bytes crus e
            ganha uma entrada por merge.
    """

    def train(self, text: str, vocab_size: int) -> None:
        """Aprende `vocab_size - 256` fusoes a partir de `text`.

        Passos:
          1. self.vocab comeca com {0: b'\\x00', 1: b'\\x01', ..., 255: b'\\xff'}
             Dica: bytes([i]) constroi o objeto bytes de um byte so.
          2. self.merges comeca vazio.
          3. ids = a lista de bytes de `text`.
             Dica: list(text.encode('utf-8')) ja devolve list[int] em 0..255.
          4. Repita `vocab_size - 256` vezes, com i = 0, 1, 2, ...:
               - stats  = get_stats(ids)
               - melhor = o par de MAIOR contagem
                   Dica: max(stats, key=stats.get) devolve a CHAVE de maior valor.
               - novo   = 256 + i
               - ids    = merge(ids, melhor, novo)
               - self.merges[melhor] = novo
               - self.vocab[novo] = self.vocab[melhor[0]] + self.vocab[melhor[1]]
                   (bytes + bytes concatena, igual a strings)

        vocab_size == 256 significa zero merges: BPE vira um tokenizer de bytes puro.
        Assuma vocab_size >= 256.

        Sobre empates: quando dois pares tem a mesma contagem, max() devolve o
        primeiro que encontrar ao percorrer o dicionario. get_stats insere os
        pares na ordem em que aparecem no texto, entao o desempate e deterministico
        -- vence o par que ocorre mais cedo. Nao e preciso tratar nada.
        """
        # 1. A base da inducao: os 256 bytes crus, cada um representando a si mesmo.
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: dict[tuple[int, int], int] = {}

        # 2. Texto -> bytes -> inteiros em 0..255.
        #    .encode() devolve um objeto `bytes`; iterar sobre ele ja da inteiros.
        ids = list(text.encode("utf-8"))

        # 3. Quantas fusoes cabem no orcamento de V que foi pedido.
        num_merges = vocab_size - 256


        for i in range(num_merges):
            stats = get_stats(ids)

            # A escolha gulosa: a chave de maior valor no dicionario de contagens.
            # Sem o `key=`, max() compararia as CHAVES (tuplas) em vez das contagens.
            melhor = max(stats, key=stats.get)

            novo = 256 + i  # o proximo id livre

            # ---------------- DAQUI PRA BAIXO E SEU ----------------
            # Faltam tres linhas, nesta ordem:
            #   a) aplicar a fusao:        ids = ...
            #   b) registrar o aprendizado: self.merges[...] = ...
            #   c) montar os bytes do novo simbolo a partir dos bytes dos pais:
            #      self.vocab[novo] = ...

            ids = merge(ids, melhor, novo)
            self.merges[melhor] = novo
            self.vocab[novo] = self.vocab[melhor[0]] + self.vocab[melhor[1]]



    def save(self, path: str | Path) -> None:
        """Grava o tokenizer treinado em disco.

        Formato (texto puro, uma fusao por linha):

            llm-bpe v1
            97 110
            98 256
            257 256

        O cabecalho identifica a versao do formato -- ele vai mudar quando
        entrarem tokens especiais e o padrao regex.

        SO O `merges` E SALVO. O `vocab` e inteiramente derivado dele (base =
        os 256 bytes, cada entrada nova = concatenacao dos pais), entao gravar
        os dois criaria duas fontes de verdade que podem divergir.

        O ID TAMBEM NAO E SALVO: a rodada i sempre recebe o id 256 + i, entao a
        ORDEM DAS LINHAS ja carrega essa informacao. A linha 0 define o 256, a
        linha 1 define o 257, e assim por diante.

        Dicas:
            Path(path).write_text(texto, encoding="utf-8")
            "\\n".join(lista_de_strings)    junta linhas
            f"{a} {b}"                      formata um par
        """
        linhas = [CABECALHO]
        for esq, dir_ in self.merges:
            linhas.append(f"{esq} {dir_}")

        Path(path).write_text("\n".join(linhas), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Reconstroi um tokenizer a partir do arquivo gravado por save().

        E um classmethod: nao opera sobre uma instancia existente, CRIA uma
        nova. Chamada como BPETokenizer.load("tok.txt"), sem instanciar antes.
        Dentro do metodo, `cls` e a propria classe -- cls() cria a instancia.

        Passos:
          1. Leia o arquivo e separe as linhas.
          2. Confira o cabecalho; se nao for o esperado, levante ValueError com
             mensagem util. Arquivo de formato errado deve falhar alto, nao
             produzir um tokenizer silenciosamente quebrado.
          3. Crie a instancia e inicialize vocab com os 256 bytes e merges vazio
             -- exatamente como o train faz no comeco.
          4. Para cada linha restante, na ordem:
               - leia os dois inteiros do par
               - o id novo e 256 + (indice da linha)
               - registre em self.merges
               - reconstrua self.vocab[novo] concatenando os bytes dos pais

        O passo 4 e o mesmo da inducao do train, menos a descoberta: la voce
        contava para achar o par, aqui voce ja sabe qual e.

        Dicas:
            Path(path).read_text(encoding="utf-8").splitlines()
            int("97")           converte texto para inteiro
            linha.split()       quebra "97 110" em ["97", "110"]
            enumerate(x)        da (indice, elemento)
        """
        linhas = Path(path).read_text(encoding="utf-8").splitlines()

        if not linhas or linhas[0] != CABECALHO:
            visto = linhas[0] if linhas else "<arquivo vazio>"
            raise ValueError(
                f"formato desconhecido: esperava {CABECALHO!r}, veio {visto!r}"
            )

        tok = cls()
        tok.vocab = {i: bytes([i]) for i in range(256)}
        tok.merges = {}

        # Mesma inducao do train, menos a descoberta: la o par era contado,
        # aqui ele ja vem escrito. A ordem das linhas da o id.
        for i, linha in enumerate(linhas[1:]):
            esq, dir_ = (int(x) for x in linha.split())
            novo = 256 + i
            tok.merges[(esq, dir_)] = novo
            tok.vocab[novo] = tok.vocab[esq] + tok.vocab[dir_]

        return tok

    def decode(self, ids: list[int]) -> str:
        """ids -> texto.

        Dois passos:
          1. Junte os bytes de cada id. Dica: self.vocab[i] da os bytes do id i,
             e b"".join(...) concatena uma sequencia de objetos bytes num so.
          2. Transforme esses bytes em str, decodificando como utf-8.
             Dica: objetos bytes tem o metodo .decode(encoding).

        ATENCAO -- por que NAO basta .decode('utf-8'):

        decode precisa aceitar QUALQUER sequencia de ids, nao so as que o seu
        encode produziu. Um modelo generativo amostra ids de uma distribuicao e
        pode emitir o byte 195 (primeiro byte de 'c-cedilha') sem o 167 que o
        completa. Os bytes resultantes nao sao utf-8 valido e .decode('utf-8')
        levanta UnicodeDecodeError.

        Passe errors='replace' para trocar os bytes invalidos pelo caractere de
        substituicao U+FFFD em vez de estourar. Um tokenizer que quebra com saida
        do modelo e inutil na pratica.
        """
        bs = b"".join(self.vocab[i] for i in ids)
        return bs.decode("utf-8", errors="replace")

    def encode(self, text: str) -> list[int]:
        """texto -> ids.

        Passos:
          1. Comece pelos bytes do texto, igual no train.
          2. Percorra self.merges NA ORDEM DE INSERCAO e aplique cada fusao a
             sequencia inteira, uma de cada vez.
             Dica: dict.items() percorre pares (chave, valor) nessa ordem --
             aqui, (par, id_novo), que sao exatamente os dois argumentos que
             faltam para chamar merge().

        POR QUE A ORDEM E OBRIGATORIA:

        O merge 257 do traco e ('o', 256) -- ele so encontra o padrao se o
        simbolo 256 ja tiver sido criado pelo merge anterior. Aplicar fora de
        ordem simplesmente nao acha nada, e falha em silencio: devolve ids
        validos, so que piores. Nao ha excecao para te avisar.

        Como voce repete aqui a mesma sequencia de operacoes do train,
        encode(texto_de_treino) reproduz exatamente os ids com que o train
        terminou.

        (Uma unica passada na ordem e suficiente: fundir dois simbolos nunca
        torna adjacentes dois simbolos que antes nao eram, porque o token novo
        ocupa o lugar dos dois. Entao nenhuma fusao posterior cria oportunidade
        para uma anterior.)
        """
        ids = list(text.encode("utf-8"))

        for par, novo in self.merges.items():
            ids = merge(ids, par, novo)

        return ids


#res = get_stats([1, 2, 3, 1, 2])
#print(res)


#print(merge([1, 2, 3, 1, 2], (1, 2), 99)) # ->  [99, 3, 99]
