import numpy as np

# toolkit pour le traitement du nlp
import toolkit as tk

class Intent():
    '''
    Classe associée à un intent qui peut exister dans une phrase.
    Exemples = lumière, musique, blague
    '''

    def __init__(self, word_to_vec, nom):
        '''
        - nom est le nom de l'intent (humour, blague, etc)
        - keywords est un ensemble de mots qui caractérisent l'intent
          (lux = lumière, allumer... )
        - model est le word2vec utilisé
        '''
        self.nom = nom
        self.key_words = []
        self.model = word_to_vec

    def train(self, sentence):
        '''
        Ajoute sentence à la liste des key_words.
        Plus il y a de mots dans key_words, plus le classifier sera performant
        '''
        sentence = tk.prepare_sentence(sentence)
        for word in sentence:
            # je ne l'ajoute que s'il est dans le word2vec
            try:
                self.model[word]

                if word not in self.key_words:
                    self.key_words.append(word)

            except KeyError:
                pass

    def classifier_score(self, sentence):
        '''
        Donne en fonction d'une phrase un score d'autant plus élevé que
        le mot est proche de l'intent.
        C'est la fonction charnière de notre classe.
        '''

        sentence = tk.prepare_sentence(sentence) # tokenize / stopwords / punctuation
        sentence_vectorized = tk.sentence_to_vec_weights(sentence, self.model, self.key_words) # poids
        dist_list = [np.linalg.norm(self.model[keyword]-sentence_vectorized) for keyword in self.key_words]
        min_dist = min(dist_list)
        return(1-min_dist/5) # formule à la con pour avoir quelque chose qui ressemble à peu près à un %

    def __str__(self):
        '''
        Ce que va retourner la fonction print.
        '''
        return(str(self.key_words))
