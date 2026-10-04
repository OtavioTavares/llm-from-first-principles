from llm.tokenizer.bpe import BPETokenizer

# 'abcabcabd' em bytes -- o exemplo que rodamos na mao.
#   a=97  b=98  c=99  d=100
TEXTO = "abcabcabd"
A, B, C, D = 97, 98, 99, 100


def test_sem_merges_e_so_um_tokenizer_de_bytes():
    tok = BPETokenizer()
    tok.train(TEXTO, vocab_size=256)
    assert tok.merges == {}
    assert len(tok.vocab) == 256
    assert tok.vocab[A] == b"a"
    assert tok.vocab[255] == b"\xff"


def test_primeiro_merge_e_o_par_mais_frequente():
    tok = BPETokenizer()
    tok.train(TEXTO, vocab_size=257)
    # (a,b) ocorre 3x; (b,c) e (c,a) ocorrem 2x
    assert tok.merges == {(A, B): 256}


def test_segundo_merge_usa_o_simbolo_criado_no_primeiro():
    """Prova que o BPE constroi em cima de si mesmo."""
    tok = BPETokenizer()
    tok.train(TEXTO, vocab_size=258)
    assert tok.merges == {(A, B): 256, (256, C): 257}


def test_vocab_e_a_concatenacao_dos_bytes_dos_pais():
    tok = BPETokenizer()
    tok.train(TEXTO, vocab_size=258)
    assert tok.vocab[256] == b"ab"
    assert tok.vocab[257] == b"abc"


def test_tamanho_do_vocab_e_exatamente_o_pedido():
    for v in (256, 257, 258, 260):
        tok = BPETokenizer()
        tok.train(TEXTO, vocab_size=v)
        assert len(tok.vocab) == v
        assert len(tok.merges) == v - 256


def test_ids_novos_sao_sequenciais_a_partir_de_256():
    tok = BPETokenizer()
    tok.train(TEXTO, vocab_size=260)
    assert sorted(tok.merges.values()) == [256, 257, 258, 259]


def test_merges_preserva_a_ordem_de_aprendizado():
    """A ordem e o que torna o encode reproduzivel -- vamos depender dela."""
    tok = BPETokenizer()
    tok.train(TEXTO, vocab_size=258)
    assert list(tok.merges.keys()) == [(A, B), (256, C)]


def test_treino_e_deterministico():
    a, b = BPETokenizer(), BPETokenizer()
    a.train(TEXTO, vocab_size=260)
    b.train(TEXTO, vocab_size=260)
    assert a.merges == b.merges
    assert a.vocab == b.vocab


def test_funciona_com_texto_nao_ascii():
    """Nivel de byte: acento nao e caso especial, e so uma sequencia de bytes."""
    tok = BPETokenizer()
    tok.train("coração coração coração", vocab_size=258)
    # todo token, por mais alto, se desmonta ate bytes validos
    for ids_, bs in tok.vocab.items():
        assert isinstance(bs, bytes)
    assert len(tok.vocab) == 258
