import nltk # natural language tool kit

# importation des french stop words
from nltk.corpus import stopwords
french_stopwords = set(stopwords.words('french'))

import numpy as np

from lemmatiseur.lemmatizeur import Lemmatiseur
lemmatiseur = Lemmatiseur()

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

class Intent():
    '''
    Classe associée à un intent qui peut exister dans une phrase.
    Exemples = lumière, musique, blague
    '''

    def __init__(self, word_to_vec, nom):
        self.nom = nom
        self.key_words = []
        self.model = word_to_vec

    def train(self, sentence):
        '''
        Met à jour les key_words pour améliorer le classifier
        Très simple : les key_words est un dictionnaire avec tous les mots
        apparus en entrainement et leur fréquence d'apparition

        NB : pour un bon entrainement en raison de comment fonctionne notre
        classifier il faut utiliser des phrases longues avec des mots avec peu
        de chance d'apparition
        '''
        sentence = prepare_sentence(sentence)
        for word in sentence:
            # je ne l'ajoute que s'il est dans le word 2 vec
            try:
                self.model[word]

                if word not in self.key_words:
                    self.key_words.append(word)

            except KeyError:
                print(word, " n'est pas dans le W2V")

    def classifier_score(self, sentence):
        sentence = prepare_sentence(sentence)
        sentence_vectorized = sentence_to_vec_weights(sentence, self.model, self.key_words)
        dist_list = [np.linalg.norm(self.model[keyword]-sentence_vectorized) for keyword in self.key_words]
        min_dist = min(dist_list)
        return(1-min_dist/5)

    def __str__(self):
        return(str(self.key_words))

def prepare_sentence(sentence):
    '''
    Takes a sentence returns it tokenized w/o French stopwords & lemmatized
    Also deletes apostrophes
    '''

    sentence = tokenizer.tokenize(sentence)
    new_sentence = []
    for i in range(len(sentence)):
        if sentence[i] not in french_stopwords:
            word = lemmatiseur.lemmatize(sentence[i])
            new_sentence.append(word)
    return(new_sentence)

def sentence_to_vec(sentence, model):
    '''
    Prend une liste de mots et return la moy
    '''
    # éventuellement penser à dégager les outliers
    vec = np.zeros(model["lampe"].shape[0])
    length = 0
    for word in sentence:
        try:
            vec += model[word]
            length += 1
        except KeyError:
            continue
    if length == 0:
        raise ValueError('''Cette phrase ne contient que des mots
        absents dans le Word2Vec''')
    return(vec/length)

def sentence_to_vec_weights(sentence, model, key_words):
    '''
    Prend une liste de mots et return la moy
    '''
    # éventuellement penser à dégager les outliers
    vec = np.zeros(model["lampe"].shape[0])
    length = 0
    for word in sentence:
        try:
            if word in key_words:
                vec += 2*model[word]
                length += 2
            else:
                vec += model[word]
                length += 1
        except KeyError:
            continue
    if length == 0:
        raise ValueError('''Cette phrase ne contient que des mots
        absents dans le Word2Vec''')
    return(vec/length)
