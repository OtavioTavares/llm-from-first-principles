import pytest

from llm.tokenizer.char import CharTokenizer


def test_vocab_e_construido_dos_caracteres_distintos():
    tok = CharTokenizer("banana")
    # 'banana' tem 3 caracteres distintos: a, b, n
    assert tok.vocab_size == 3


def test_ids_seguem_ordem_alfabetica():
    tok = CharTokenizer("banana")
    assert tok.stoi["a"] == 0
    assert tok.stoi["b"] == 1
    assert tok.stoi["n"] == 2


def test_itos_e_o_inverso_exato_de_stoi():
    tok = CharTokenizer("o rato roeu a roupa")
    for ch, i in tok.stoi.items():
        assert tok.itos[i] == ch
    assert len(tok.itos) == tok.vocab_size


def test_encode_produz_ids_validos():
    tok = CharTokenizer("banana")
    ids = tok.encode("banana")
    assert ids == [1, 0, 2, 0, 2, 0]
    assert all(0 <= i < tok.vocab_size for i in ids)


def test_roundtrip_preserva_o_texto():
    texto = "To be, or not to be: that is the question.\n"
    tok = CharTokenizer(texto)
    assert tok.decode(tok.encode(texto)) == texto


def test_um_token_por_caractere():
    """Propriedade que define este tokenizer -- e a razao de ele ser ruim."""
    texto = "primeiro principio"
    tok = CharTokenizer(texto)
    assert len(tok.encode(texto)) == len(texto)


def test_caractere_fora_do_vocabulario_falha_alto():
    """Melhor estourar do que silenciosamente corromper a sequencia."""
    tok = CharTokenizer("abc")
    with pytest.raises(KeyError):
        tok.encode("abcd")
