import re

class WhiteSpaceTokenizer():
    def __init__(self):
        pass

    def tokenize(self, doc):
        tokens = doc.split()
        tokens[:] = [e.strip() for e in tokens]
        return tokens


class CharTokenizer():
    def __init__(self):
        pass

    def tokenize(self, doc):
         return list(re.sub(r"[\s]{2,}", " ", doc, flags=re.UNICODE))