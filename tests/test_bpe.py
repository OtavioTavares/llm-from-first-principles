from llm.tokenizer.bpe import get_stats, merge


# ---------------------------------------------------------------- get_stats

def test_stats_conta_pares_adjacentes():
    assert get_stats([1, 2, 3]) == {(1, 2): 1, (2, 3): 1}


def test_stats_soma_repeticoes():
    assert get_stats([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}


def test_stats_a_ordem_do_par_importa():
    """(1,2) e (2,1) sao pares DIFERENTES -- texto tem direcao."""
    assert get_stats([1, 2, 1, 2]) == {(1, 2): 2, (2, 1): 1}


def test_stats_conta_pares_sobrepostos():
    """Em [7,7,7] o par (7,7) ocorre em (0,1) e em (1,2): contagem 2."""
    assert get_stats([7, 7, 7]) == {(7, 7): 2}


def test_stats_n_menos_1_pares():
    ids = [4, 8, 15, 16, 23, 42]
    assert sum(get_stats(ids).values()) == len(ids) - 1


def test_stats_sem_pares_possiveis():
    assert get_stats([]) == {}
    assert get_stats([5]) == {}


# -------------------------------------------------------------------- merge

def test_merge_substitui_ocorrencia_unica():
    assert merge([1, 2, 3], (1, 2), 99) == [99, 3]


def test_merge_substitui_todas_as_ocorrencias():
    assert merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]


def test_merge_preserva_o_que_nao_casa():
    assert merge([5, 1, 2, 6], (1, 2), 99) == [5, 99, 6]


def test_merge_sem_ocorrencia_devolve_igual():
    assert merge([1, 2, 3], (7, 8), 99) == [1, 2, 3]


def test_merge_nao_reutiliza_elemento_ja_consumido():
    """O teste que importa: [7,7,7] -> [99,7], nunca [99,99]."""
    assert merge([7, 7, 7], (7, 7), 99) == [99, 7]


def test_merge_pares_consecutivos_exatos():
    assert merge([7, 7, 7, 7], (7, 7), 99) == [99, 99]


def test_merge_par_no_fim_da_lista():
    assert merge([5, 6, 1, 2], (1, 2), 99) == [5, 6, 99]


def test_merge_elemento_solto_no_fim_sobrevive():
    """O ultimo elemento nao inicia par nenhum, mas tem que aparecer na saida."""
    assert merge([1, 2, 7], (1, 2), 99) == [99, 7]


def test_merge_casos_degenerados():
    assert merge([], (1, 2), 99) == []
    assert merge([5], (1, 2), 99) == [5]


def test_merge_nao_modifica_a_entrada():
    ids = [1, 2, 3]
    merge(ids, (1, 2), 99)
    assert ids == [1, 2, 3]


# ------------------------------------------------------- os dois trabalhando

def test_uma_rodada_completa_de_bpe():
    """'abcabcabd' em bytes: funde o par mais frequente e encolhe 3."""
    ids = list("abcabcabd".encode("utf-8"))
    stats = get_stats(ids)

    melhor = max(stats, key=stats.get)
    assert melhor == (ord("a"), ord("b"))
    assert stats[melhor] == 3

    novos = merge(ids, melhor, 256)
    assert len(novos) == len(ids) - 3
    assert novos == [256, ord("c"), 256, ord("c"), 256, ord("d")]
