"""Tokenizer de caracteres: o caso mais simples possivel.

Serve de baseline. Tudo que vier depois (BPE) sera medido contra ele.
"""


class CharTokenizer:
    """Mapeia cada caractere distinto do corpus para um inteiro.

    O vocabulario e construido a partir dos caracteres presentes em `text`,
    ordenados alfabeticamente. A ordenacao importa: garante que o mesmo corpus
    sempre produza o mesmo vocabulario (reprodutibilidade).
    """

    def __init__(self, text: str) -> None:
        """Constroi o vocabulario a partir do corpus.

        Deve definir dois dicionarios:
            self.stoi: dict[str, int]  -- caractere  -> id
            self.itos: dict[int, str]  -- id         -> caractere

        Os ids vao de 0 a V-1, atribuidos na ordem alfabetica dos caracteres.
        """

        self.text:str = text
        
        chars:list[str] = list(sorted(set(text)))
        id_to_str = list(enumerate(chars))

        self.stoi: dict[str, int] = {v:k for k,v in id_to_str}
        self.itos: dict[int, str] = {k:v for k,v in id_to_str}

        
    @property
    def vocab_size(self) -> int:
        """Numero de simbolos distintos. Este e o V das contas da licao."""
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        """texto -> lista de ids."""

        array_encode = []
        for caracter in text:
            id_ = self.stoi[caracter]
            array_encode.append(id_)
        return array_encode
    

    def decode(self, ids: list[int]) -> str:
        """lista de ids -> texto."""
        return "".join(self.itos[i] for i in ids)

#tok = CharTokenizer("banana")
#res = tok.vocab_size
#print(res)

#tok = CharTokenizer("banana")
#print(tok.stoi)

#tok = CharTokenizer("banana")
#print(tok.itos)

#tok = CharTokenizer("banana")
#res = tok.encode('banana')
#print(res)


#tok = CharTokenizer('banana')
#print(tok.decode(tok.encode('banana')))