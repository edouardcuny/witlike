import numpy as np

# toolkit pour le traitement du nlp
import toolkit as tk

class Intent():
    '''
    Classe associée à un intent qui peut exister dans une phrase.
    Exemples = lumière, musique, blague
    '''

    def __init__(self, nom):
        '''
        - nom est le nom de l'intent (humour, blague, etc)
        - keywords est un ensemble de mots qui caractérisent l'intent
          (lux = lumière, allumer... )
        - model est le word2vec utilisé
        '''
        self.nom = nom
        self.key_words = []

    def train(self, sentence):
        '''
        Ajoute sentence à la liste des key_words.
        Plus il y a de mots dans key_words, plus le classifier sera performant
        '''
        sentence = tk.prepare_sentence(sentence)
        for word in sentence:
            # je ne l'ajoute que s'il n'est pas déjà dans keywords
            if word not in self.key_words:
                    self.key_words.append(word)

    def __str__(self):
        '''
        Ce que va retourner la fonction print.
        '''
        return(str(self.key_words))
