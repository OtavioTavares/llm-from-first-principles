import numpy as np
import pytest

from llm.embeddings.embedding import Embedding


@pytest.fixture
def emb():
    return Embedding(vocab_size=50, d_model=8, seed=0)


# ------------------------------------------------------------------- formato

def test_a_tabela_tem_uma_linha_por_token(emb):
    assert emb.weight.shape == (50, 8)


def test_propriedades_derivam_do_formato(emb):
    assert emb.vocab_size == 50
    assert emb.d_model == 8


def test_forward_devolve_uma_linha_por_id(emb):
    assert emb.forward([7, 3, 42]).shape == (3, 8)


def test_forward_de_um_id_so(emb):
    assert emb.forward([7]).shape == (1, 8)


def test_forward_de_lista_vazia(emb):
    """Sem tokens, zero linhas -- mas ainda d_model colunas."""
    assert emb.forward([]).shape == (0, 8)


# ------------------------------------------------------------------ conteudo

def test_forward_devolve_as_linhas_certas(emb):
    saida = emb.forward([7, 3])
    assert np.array_equal(saida[0], emb.weight[7])
    assert np.array_equal(saida[1], emb.weight[3])


def test_forward_preserva_a_ordem_pedida(emb):
    assert np.array_equal(emb.forward([3, 7]), emb.forward([7, 3])[::-1])


def test_o_mesmo_token_da_sempre_o_mesmo_vetor(emb):
    """Esta camada nao sabe nada sobre posicao."""
    saida = emb.forward([7, 3, 7])
    assert np.array_equal(saida[0], saida[2])


# ------------------------------------------- a identidade com one-hot @ E

def test_indexar_equivale_a_multiplicar_pelo_one_hot(emb):
    """O ponto conceitual da licao, verificado numericamente."""
    ids = [7, 3, 42]

    one_hot = np.zeros((len(ids), emb.vocab_size))
    for linha, i in enumerate(ids):
        one_hot[linha, i] = 1.0

    assert np.allclose(one_hot @ emb.weight, emb.forward(ids))


# ------------------------------------------------------------ inicializacao

def test_a_simetria_esta_quebrada(emb):
    """Se duas linhas fossem iguais, os tokens seriam indistinguiveis p/ sempre."""
    assert not np.array_equal(emb.weight[0], emb.weight[1])


def test_todas_as_linhas_sao_distintas(emb):
    distintas = {tuple(linha) for linha in emb.weight}
    assert len(distintas) == emb.vocab_size


def test_a_escala_da_inicializacao(emb):
    grande = Embedding(vocab_size=20_000, d_model=64, std=0.02, seed=0)
    assert abs(grande.weight.mean()) < 0.001
    assert 0.019 < grande.weight.std() < 0.021


def test_std_e_configuravel():
    a = Embedding(vocab_size=20_000, d_model=64, std=0.5, seed=0)
    assert 0.49 < a.weight.std() < 0.51


# ---------------------------------------------------------- reprodutibilidade

def test_a_mesma_seed_da_a_mesma_tabela():
    a = Embedding(vocab_size=50, d_model=8, seed=42)
    b = Embedding(vocab_size=50, d_model=8, seed=42)
    assert np.array_equal(a.weight, b.weight)


def test_seeds_diferentes_dao_tabelas_diferentes():
    a = Embedding(vocab_size=50, d_model=8, seed=0)
    b = Embedding(vocab_size=50, d_model=8, seed=1)
    assert not np.array_equal(a.weight, b.weight)


# ------------------------------------------------------ integracao com o BPE

def test_recebe_a_saida_do_tokenizer():
    from llm.tokenizer.bpe import BPETokenizer

    # O numero de merges possiveis depende da DIVERSIDADE do texto, nao do
    # tamanho: repetir a mesma frase 20x esgota os pares em ~32 merges.
    corpus = (
        "Uma noite destas, vindo da cidade para o Engenho Novo, encontrei num "
        "trem da Central um rapaz do bairro, que eu conheco de vista e de "
        "chapeu. Cumprimentou-me, sentou-se ao pe de mim, falou da Lua e dos "
        "ministros, e acabou recitando-me versos. A viagem era curta, e os "
        "versos pode ser que nao fossem inteiramente maus. Sucedeu, porem, que "
        "como eu estava cansado, fechei os olhos tres ou quatro vezes."
    )
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=300)

    emb = Embedding(vocab_size=len(tok.vocab), d_model=16, seed=0)
    ids = tok.encode("o rato")

    assert emb.forward(ids).shape == (len(ids), 16)
