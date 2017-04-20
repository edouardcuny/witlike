 # natural language tool kit
import nltk
from nltk.corpus import stopwords
french_stopwords = set(stopwords.words('french'))

# lemmatiseur créé à la mano à partir d'un gros csv
# s'appuie sur un back en postgres
from lemmatiseur.lemmatizeur import Lemmatiseur
lemmatiseur = Lemmatiseur()

# tokenizer qui supprime la ponction
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

import numpy as np

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
    Sentence2Vec naïf qui renvoie la moyenne des mots.
    '''
    # astuce pour créer un vect de la mê dimension que W2V
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
    Sentence2Vec naïf qui renvoie la moyenne pondérée des mots.
    Si mot not in key_words >> poids = 1
    Si mot in key_words >> poids = 2

    Si mot in keywords AND mot est premier mot phrase >> poids = 3
    Traduit le fait qu'on commence généralement par l'intention, "allume, joue"
    '''
    # astuce pour créer un vect de la mê dimension que W2V
    vec = np.zeros(model["lampe"].shape[0])

    length = 0
    for i in range(len(sentence)):
    # for word in sentence:
        try:
            if sentence[i] in key_words:
                print(sentence[i].upper(), sep=" ")
                if i == 0:
                    vec += 3*model[sentence[i]]
                    length += 3
                vec += 2*model[sentence[i]]
                length += 2
            else:
                print(sentence[i], sep=" ")
                vec += model[sentence[i]]
                length += 1
        except KeyError:
            # mot pas dans le word2vec
            continue
    if length == 0:
        raise ValueError('''Cette phrase ne contient que des mots
        absents dans le Word2Vec''')
    print()
    return(vec/length)
