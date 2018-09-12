import re

class EmptyPreprocessor():

    def __init__(self):
        pass


    def preprocess(self, text):
        return text


class regexPurgePreprocessor:

    def __init__(self, regex):
        self.regex=regex # chars to delete
        # self.ch_blacklist=ch_blacklist


    def preprocess(self, text):
        return re.sub(self.regex, ' ', text.lower())