import pytest

from llm.tokenizer.bpe import CABECALHO, BPETokenizer

TEXTO = "o rato roeu a roupa do rei de roma"


@pytest.fixture
def tok():
    t = BPETokenizer()
    t.train(TEXTO, vocab_size=260)
    return t


# -------------------------------------------------------------------- save

def test_save_cria_o_arquivo(tok, tmp_path):
    destino = tmp_path / "tok.txt"
    tok.save(destino)
    assert destino.exists()


def test_o_cabecalho_identifica_o_formato(tok, tmp_path):
    destino = tmp_path / "tok.txt"
    tok.save(destino)
    assert destino.read_text(encoding="utf-8").splitlines()[0] == CABECALHO


def test_uma_linha_por_merge(tok, tmp_path):
    destino = tmp_path / "tok.txt"
    tok.save(destino)
    linhas = destino.read_text(encoding="utf-8").splitlines()
    assert len(linhas) == 1 + len(tok.merges)  # cabecalho + merges


def test_cada_linha_tem_o_par_que_foi_fundido(tok, tmp_path):
    destino = tmp_path / "tok.txt"
    tok.save(destino)
    linhas = destino.read_text(encoding="utf-8").splitlines()[1:]
    # o primeiro merge de TEXTO e (32, 114): espaco + 'r'
    assert linhas[0] == "32 114"


def test_a_ordem_das_linhas_codifica_o_id(tok, tmp_path):
    """Linha 0 -> id 256, linha 1 -> id 257, ... Nenhum id e escrito."""
    destino = tmp_path / "tok.txt"
    tok.save(destino)
    linhas = destino.read_text(encoding="utf-8").splitlines()[1:]

    for i, linha in enumerate(linhas):
        par = tuple(int(x) for x in linha.split())
        assert tok.merges[par] == 256 + i


def test_o_vocab_nao_e_salvo(tok, tmp_path):
    """Ele e derivado do merges -- gravar os dois criaria divergencia possivel."""
    destino = tmp_path / "tok.txt"
    tok.save(destino)
    conteudo = destino.read_text(encoding="utf-8")
    assert "rato" not in conteudo
    assert " r" not in conteudo.replace(CABECALHO, "")


def test_save_de_tokenizer_sem_merges(tmp_path):
    t = BPETokenizer()
    t.train(TEXTO, vocab_size=256)
    destino = tmp_path / "tok.txt"
    t.save(destino)
    assert destino.read_text(encoding="utf-8").splitlines() == [CABECALHO]


# -------------------------------------------------------------------- load

def test_load_reconstroi_os_merges(tok, tmp_path):
    destino = tmp_path / "tok.txt"
    tok.save(destino)
    assert BPETokenizer.load(destino).merges == tok.merges


def test_load_reconstroi_o_vocab(tok, tmp_path):
    """O vocab nao estava no arquivo -- tem que ser refeito pela inducao."""
    destino = tmp_path / "tok.txt"
    tok.save(destino)
    assert BPETokenizer.load(destino).vocab == tok.vocab


def test_load_traz_os_256_bytes_crus(tok, tmp_path):
    destino = tmp_path / "tok.txt"
    tok.save(destino)
    carregado = BPETokenizer.load(destino)
    assert carregado.vocab[97] == b"a"
    assert carregado.vocab[255] == b"\xff"


def test_load_de_tokenizer_sem_merges(tmp_path):
    t = BPETokenizer()
    t.train(TEXTO, vocab_size=256)
    destino = tmp_path / "tok.txt"
    t.save(destino)

    carregado = BPETokenizer.load(destino)
    assert carregado.merges == {}
    assert len(carregado.vocab) == 256


def test_cabecalho_errado_falha_alto(tmp_path):
    """Formato desconhecido deve estourar, nao produzir tokenizer quebrado."""
    destino = tmp_path / "ruim.txt"
    destino.write_text("formato-aleatorio v9\n97 110\n", encoding="utf-8")
    with pytest.raises(ValueError):
        BPETokenizer.load(destino)


# --------------------------------------------------------------- roundtrip

def test_o_tokenizer_carregado_encoda_igual(tok, tmp_path):
    """O que importa de verdade: mesmos ids, senao os pesos do modelo viram lixo."""
    destino = tmp_path / "tok.txt"
    tok.save(destino)
    carregado = BPETokenizer.load(destino)

    for texto in [TEXTO, "o rei", "coração", "🙂", ""]:
        assert carregado.encode(texto) == tok.encode(texto)


def test_o_tokenizer_carregado_decoda_igual(tok, tmp_path):
    destino = tmp_path / "tok.txt"
    tok.save(destino)
    carregado = BPETokenizer.load(destino)

    ids = tok.encode(TEXTO)
    assert carregado.decode(ids) == tok.decode(ids)


def test_salvar_e_carregar_e_idempotente(tok, tmp_path):
    """save -> load -> save produz byte a byte o mesmo arquivo."""
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    tok.save(a)
    BPETokenizer.load(a).save(b)
    assert a.read_bytes() == b.read_bytes()
