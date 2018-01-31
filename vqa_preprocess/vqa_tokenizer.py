from nltk.tokenize import TweetTokenizer
import json
import re

class VQATokenizer:
    """ """
    def __init__(self, dictionary_file):

        self.tokenizer = TweetTokenizer(preserve_case=False)
        with open(dictionary_file, 'r') as f:
            data = json.load(f)
            self.word2i = data['word2i']
            self.answer2i = data['answer2i']
            self.preprocess_answers = data['preprocess_answers']

        self.dictionary_file = dictionary_file

        self.i2word = {}
        for (k, v) in self.word2i.items():
            self.i2word[v] = k

        self.i2answer = {}
        for (k, v) in self.answer2i.items():
            self.i2answer[v] = k

        # Retrieve key values
        self.no_words = len(self.word2i)
        self.no_answers = len(self.answer2i)

        self.unknown_question_token = self.word2i["<unk>"]
        self.padding_token = self.word2i["<unk>"]

        self.unknown_answer = self.answer2i["<unk>"]



    """
    Input: String
    Output: List of tokens
    """
    def encode_question(self, question):
        tokens = []
        words = []
        for token in self.tokenizer.tokenize(question):
            if token not in self.word2i:
                token = '<unk>'
            tokens.append(self.word2i[token])
            words.append(token)
        # return both the tokens and list of words
        return tokens, words

    def decode_question(self, tokens):
        return ' '.join([self.i2word[tok] for tok in tokens])

    def encode_answer(self, answer):
        if answer not in self.answer2i:
            return self.answer2i['<unk>']
        return self.answer2i[answer]

    def decode_answer(self, answer_id):
        return self.i2answer[answer_id]

    def tokenize_question(self, question):
        return self.tokenizer.tokenize(question)
