import numpy as np
import pytest

from llm.embeddings.similarity import cosine_similarity, dot, norm


# ---------------------------------------------------------------------- norm

def test_norm_pitagoras_em_2d():
    assert norm(np.array([3.0, 4.0])) == pytest.approx(5.0)


def test_norm_de_vetor_unitario():
    assert norm(np.array([1.0, 0.0])) == pytest.approx(1.0)


def test_norm_do_vetor_nulo():
    assert norm(np.array([0.0, 0.0, 0.0])) == pytest.approx(0.0)


def test_norm_nunca_e_negativa():
    assert norm(np.array([-3.0, -4.0])) == pytest.approx(5.0)


def test_norm_em_muitas_dimensoes():
    """A formula nao muda com d -- so tem mais termos."""
    v = np.ones(100)  # cem uns
    assert norm(v) == pytest.approx(10.0)  # raiz(100 * 1) = 10


def test_norm_bate_com_a_implementacao_do_numpy():
    rng = np.random.default_rng(0)
    v = rng.normal(0, 1, size=50)
    assert norm(v) == pytest.approx(np.linalg.norm(v))


# ----------------------------------------------------------------------- dot

def test_dot_exemplo_feito_a_mao():
    assert dot(np.array([3.0, 4.0]), np.array([4.0, 3.0])) == pytest.approx(24.0)


def test_dot_de_perpendiculares_e_zero():
    assert dot(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_dot_de_opostos_e_negativo():
    assert dot(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == pytest.approx(-1.0)


def test_dot_e_comutativo():
    a, b = np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])
    assert dot(a, b) == pytest.approx(dot(b, a))


def test_dot_de_um_vetor_consigo_mesmo_e_a_norma_ao_quadrado():
    """a . a = |a|^2 -- cai direto da definicao."""
    a = np.array([3.0, 4.0])
    assert dot(a, a) == pytest.approx(norm(a) ** 2)


def test_dot_bate_com_a_implementacao_do_numpy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=50), rng.normal(size=50)
    assert dot(a, b) == pytest.approx(np.dot(a, b))


# ------------------------------------------------------- cosine_similarity

def test_cosseno_de_vetores_identicos_e_um():
    a = np.array([3.0, 4.0])
    assert cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosseno_ignora_o_tamanho():
    """O ponto da funcao: so a direcao importa."""
    a = np.array([1.0, 0.0])
    assert cosine_similarity(a, np.array([2.0, 0.0])) == pytest.approx(1.0)
    assert cosine_similarity(a, np.array([99.0, 0.0])) == pytest.approx(1.0)


def test_cosseno_de_perpendiculares_e_zero():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosseno_de_opostos_e_menos_um():
    a, b = np.array([1.0, 0.0]), np.array([-5.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosseno_do_exemplo_feito_a_mao():
    """24 / (5 * 5) = 0.96, ou seja theta ~ 16 graus."""
    a, b = np.array([3.0, 4.0]), np.array([4.0, 3.0])
    assert cosine_similarity(a, b) == pytest.approx(0.96)


def test_cosseno_sempre_entre_menos_um_e_um():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a, b = rng.normal(size=10), rng.normal(size=10)
        assert -1.0 - 1e-9 <= cosine_similarity(a, b) <= 1.0 + 1e-9


def test_cosseno_do_vetor_nulo_nao_estoura():
    nulo = np.zeros(3)
    assert cosine_similarity(nulo, np.array([1.0, 2.0, 3.0])) == 0.0
    assert cosine_similarity(nulo, nulo) == 0.0


def test_cosseno_recupera_o_angulo():
    """cos(60 graus) = 0.5 -- confere a identidade a.b = |a||b|cos(theta)."""
    a = np.array([1.0, 0.0])
    b = np.array([0.5, np.sqrt(3) / 2])  # 60 graus da horizontal
    assert cosine_similarity(a, b) == pytest.approx(0.5)


# ------------------------------------------------ integracao com Embedding

def test_embeddings_aleatorios_sao_quase_perpendiculares():
    """A linha de base ANTES do treino: nenhuma estrutura, tudo perto de zero.

    Em dimensao alta, dois vetores aleatorios sao quase sempre quase
    perpendiculares. E contra esse caos que a estrutura aprendida vai aparecer.
    """
    from llm.embeddings.embedding import Embedding

    emb = Embedding(vocab_size=500, d_model=128, seed=0)
    sims = [
        cosine_similarity(emb.weight[i], emb.weight[j])
        for i in range(0, 100)
        for j in range(i + 1, 100)
    ]
    assert abs(np.mean(sims)) < 0.02
    assert max(abs(s) for s in sims) < 0.5
