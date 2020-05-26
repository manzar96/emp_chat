from enum import Enum


class SPECIAL_TOKENS(Enum):
    PAD = '[PAD]'
    MASK = '[MASK]'
    UNK = '[UNK]'
    BOS = '[BOS]'
    EOS = '[EOS]'
    CLS = '[CLS]'



    @classmethod
    def has_token(cls, token):
        return any(token == t.name or token == t.value
                   for t in cls)

    @classmethod
    def to_list(cls):
        return list(map(lambda x: x.value, cls))

class DIALOG_SPECIAL_TOKENS(Enum):
    PAD = '<pad>'
    UNK = '<unk>'
    SOS = '<s>'
    EOS = '</s>'

    @classmethod
    def has_token(cls, token):
        return any(token == t.name or token == t.value
                   for t in cls)

    @classmethod
    def to_list(cls):
        return list(map(lambda x: x.value, cls))