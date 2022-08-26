"""
Примеры использования пайплайна
"""
from bs4 import BeautifulSoup
import re
import string
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
from nltk.corpus.reader.tagged import SpaceTokenizer
import pandas as pd


class RTPPipeline:

    def __init__(self):
        print("Pipeline is ready!")
        self.text_tokens = None
        self.stemmed_tokens = None
        self.pos_tokens = None
        self.lemma_tokens = None
        self.vec_repr = None
        self.vectorizer = None

    def preprocess(self, text, remove_web_noise=True, remove_stop_words=True, remove_punctuation=True, tokenizer=None,
                   stemmer=None, add_pos_tags=False, lemmatizer=None, vectorizer=None):
        print("\nRaw Text:\n", text)
        self.text_tokens = None
        self.stemmed_tokens = None
        self.pos_tokens = None
        self.lemma_tokens = None
        self.vec_repr = None
        self.vectorizer = None

        if remove_web_noise:
            text = self.web_preprocess(text)
            print("Web Preprocessed Text:\n", text)
        if remove_stop_words:
            text = self.stop_words_preprocess(text)
            print("Stopwords Preprocessed Text:\n", text)
        if remove_punctuation:
            if vectorizer is not None:
                text2vec = text
            text = self.punctuation_preprocess(text)
            print("Punctuation Preprocessed Text:\n", text)
        if tokenizer is not None:
            self.text_tokens = self.token_preprocess(text, tokenizer)
            print("Tokenizer Preprocessed Text:\n", self.text_tokens)
        if stemmer is not None:
            if tokenizer is not None:
                self.stemmed_tokens = self.stemmer_preprocess(text, tokenizer, stemmer)
                print("Stemmer Preprocessed Text:\n", self.stemmed_tokens)
            else:
                raise AttributeError("Stemmer Preprocessing cannot be executed: no tokenizer has been assigned. \
                                      Please, specify the tokenizer with 'tokenizer=' attribute.")
        if add_pos_tags:
            if tokenizer is not None:
                self.pos_tokens = self.pos_preprocess(text, tokenizer)
                print("POS Preprocessed Text:\n", self.pos_tokens)
            else:
                raise AttributeError("POS Preprocessing cannot be executed: no tokenizer has been assigned. \
                                      Please, specify the tokenizer with 'tokenizer=' attribute.")
        if lemmatizer is not None:
            if add_pos_tags:
                self.lemma_tokens = self.lemma_preprocess(text, tokenizer, lemmatizer)
                print("Lemmatizer Preprocessed Text:\n", self.lemma_tokens)
            else:
                raise AttributeError("Lemmatizer Preprocessing cannot be executed: POS tagging has been declined. \
                                      You can allow the POS tagging with 'add_pos_tags=True' attribute.")
        if vectorizer is not None:
            if tokenizer is not None:
                if remove_punctuation:
                    self.vec_repr = self.vector_preprocess(text2vec, tokenizer, vectorizer,
                                                           punctuation=remove_punctuation)
                    df_vec_repr = pd.DataFrame(self.vec_repr)
                    print("Vectorizer Preprocessed Text:\n", df_vec_repr.head(10))
                else:
                    self.vec_repr = self.vector_preprocess(text, tokenizer, vectorizer)
                    df_vec_repr = pd.DataFrame(self.vec_repr)
                    print("Vectorizer Preprocessed Text:\n", df_vec_repr.head(10))
            else:
                raise AttributeError("Vectoriser Preprocessing cannot be executed: no tokenizer has been assigned. \
                                      Please, specify the tokenizer with 'tokenizer=' attribute.")

    def web_preprocess(self, text):
        text = BeautifulSoup(text, 'html.parser').get_text().lower()
        text = re.sub(r'https\S', '', text)
        return text

    def stop_words_preprocess(self, text):
        stop_words = set(stopwords.words('english'))
        tokens = []
        for sent in sent_tokenize(text):
            sent_tokens = []
            for word in SpaceTokenizer().tokenize(sent):
                if word.lower() not in stop_words:
                    sent_tokens.append(word)
            tokens.append(sent_tokens)
        text = ' '.join([' '.join(sent) for sent in tokens])
        return text

    def punctuation_preprocess(self, text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())
        return text

    def token_preprocess(self, text, tokenizer):
        text_tokens = tokenizer.tokenize(text)
        return text_tokens

    def stemmer_preprocess(self, text, tokenizer, stemmer):
        stemmed_tokens = [stemmer.stem(token) for token in self.token_preprocess(text, tokenizer)]
        return stemmed_tokens

    def pos_preprocess(self, text, tokenizer):
        pos_tokens = pos_tag(self.token_preprocess(text, tokenizer))
        return pos_tokens

    def lemma_preprocess(self, text, tokenizer, lemmatizer):
        lemma_tokens = [(token, lemmatizer.lemmatize(token, self.get_wordnet_pos(pos)), pos) for (token, pos) in
                        self.pos_preprocess(text, tokenizer)]
        return lemma_tokens

    def get_wordnet_pos(self, brown_tag):
        if brown_tag.startswith('N'):
            return wordnet.NOUN
        elif brown_tag.startswith('V'):
            return wordnet.VERB
        elif brown_tag.startswith('A'):
            return wordnet.ADJ
        elif brown_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def vector_preprocess(self, text, tokenizer, vectorizer, punctuation=False):
        if punctuation:
            sents = sent_tokenize(text)
            sents = [self.punctuation_preprocess(sent) for sent in sents]
            self.vectorizer = vectorizer(tokenizer=tokenizer.tokenize)
            vec_repr = self.vectorizer.fit_transform(sents).toarray()
        else:
            sents = sent_tokenize(text)
            self.vectorizer = vectorizer(tokenizer=tokenizer.tokenize)
            vec_repr = self.vectorizer.fit_transform(sents).toarray()
        return vec_repr
