import pytest

from llm.tokenizer.bpe import BPETokenizer

TEXTO = "o rato roeu a roupa do rei de roma"


@pytest.fixture
def tok():
    t = BPETokenizer()
    t.train(TEXTO, vocab_size=260)
    return t


# ------------------------------------------------------------------- decode

def test_decode_de_bytes_crus(tok):
    assert tok.decode([111, 32, 114]) == "o r"


def test_decode_expande_simbolos_fundidos(tok):
    """O id 256 vale dois bytes; o decode tem que expandir."""
    assert tok.decode([256]) == " r"
    assert tok.decode([111, 256]) == "o r"


def test_decode_de_lista_vazia(tok):
    assert tok.decode([]) == ""


def test_decode_nao_quebra_com_utf8_invalido(tok):
    """195 e o primeiro byte de 'c-cedilha'; sozinho nao forma caractere."""
    resultado = tok.decode([195])
    assert resultado == "�"


def test_decode_nao_quebra_com_byte_solto_no_meio(tok):
    resultado = tok.decode([111, 195, 111])
    assert resultado.startswith("o")
    assert resultado.endswith("o")
    assert "�" in resultado


# ------------------------------------------------------------------- encode

def test_encode_sem_merges_e_so_os_bytes():
    t = BPETokenizer()
    t.train(TEXTO, vocab_size=256)
    assert t.encode("o r") == [111, 32, 114]


def test_encode_aplica_o_primeiro_merge(tok):
    """(32,114) -> 256 foi o merge 1."""
    assert tok.encode(" r") == [256]


def test_encode_aplica_merges_encadeados(tok):
    """256 = ' r', depois 257 = 'o' + ' r'. Exige a ordem correta."""
    assert tok.encode("o r") == [257]


def test_encode_comprime_de_verdade(tok):
    n_bytes = len(TEXTO.encode("utf-8"))
    assert len(tok.encode(TEXTO)) < n_bytes


def test_encode_de_string_vazia(tok):
    assert tok.encode("") == []


def test_encode_reproduz_o_resultado_do_train(tok):
    """A propriedade central: encode repete a sequencia de operacoes do train."""
    ids = tok.encode(TEXTO)
    # 34 bytes, 4 merges com contagens 5, 3, 2 e 2 -> 34-5-3-2-2 = 22
    assert len(ids) == 22


# ------------------------------------------------------------------ roundtrip

def test_roundtrip_no_corpus_de_treino(tok):
    assert tok.decode(tok.encode(TEXTO)) == TEXTO


def test_roundtrip_em_texto_nunca_visto(tok):
    for texto in ["o rei", "xyz", "", "    ", "9 + 9 = 18"]:
        assert tok.decode(tok.encode(texto)) == texto


def test_roundtrip_com_acento_e_emoji(tok):
    """Cobertura universal: nada esta fora do alcance de um tokenizer de bytes."""
    for texto in ["coração", "não é", "🙂", "日本語"]:
        assert tok.decode(tok.encode(texto)) == texto


def test_ids_sempre_dentro_do_vocabulario(tok):
    for texto in [TEXTO, "coração", "🙂"]:
        for i in tok.encode(texto):
            assert i in tok.vocab


def test_encode_e_deterministico(tok):
    assert tok.encode(TEXTO) == tok.encode(TEXTO)
