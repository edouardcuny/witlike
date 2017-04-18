# coding : utf-8

import word2vec # word2vec déjà entraîné
import os
import numpy as np
from intent import Intent
import pandas as pd

# importation du W2V
os.chdir("/Users/edouardcuny/Desktop/witlike/wit")
# model = word2vec.load('frWiki_no_phrase_no_postag_1000_skip_cut100.bin')
model = word2vec.load('frWac_no_postag_no_phrase_700_skip_cut50.bin')

# Lumière
lux = Intent(word_to_vec = model, nom = "lux")
lux.train("allumer")
lux.train("lumière")
lux.train("lampe")
lux.train("éteindre")

# Ratp
ratp = Intent(word_to_vec = model, nom = "ratp")
ratp.train("trajet")
ratp.train("itinéraire")
ratp.train("aller")

# Musique
music = Intent(word_to_vec = model, nom = "music")
music.train("jouer")
music.train("mettre")
music.train("couper")
music.train("chanson")

# Humour
humour = Intent(word_to_vec = model, nom = "humour")
humour.train("blague")
humour.train("vanne")
humour.train("rire")
humour.train("marrer")
humour.train("rigoler")
humour.train("drôle")
humour.train("marrant")

class Wit():
    def __init__(self, word_to_vec, *args):
        self.intents = args # liste de tous les intents
        self.threshold = 0.7 # si score en dessous, intent pas retenu
        self.word_to_vec = word_to_vec

    def train_wit(self):
        # à modifier de façon à ce que le seuil se fixe de façon automatique
        self.threshold = 0.7

    def new_intent(self, nom):
        '''
        Outil de création d'un nouvel intent.
        A développer.
        '''
        intent = Intent(word_to_vec = self.word_to_vec, nom = nom)
        self.intents.append(intent)

    def classify_intent(self, sentence):
        '''
        Renvoie pour 'sentence' l'intent avec le score maximal.
        NB = pour l'instant pas d'histoire de threshold cad qu'il renverra
        toujours un intent
        '''
        score_l = []
        for intent in self.intents:
            score_l.append(intent.classifier_score(sentence))
        idx = score_l.index(max(score_l))
        return(self.intents[idx].nom) # après la faire en mode dictionnaire

    def score(self, test):
        '''
        Renvoie un score et un pourcentage sur la performance de classification
        sur un set de test.

        score = moyenne(écart au carrés à 2 du score de l'intent)
        pourcentage = pourcentage de requêtes bien classifiées

        '''
        new = test.loc[test["intent"] != "nothing"].copy()
        new["score"] = np.vectorize(lambda x,y : self.string_to_intent(x).classifier_score(y))(new["intent"], new["phrase"])
        new["error"] = new["score"].apply(lambda x: (2.0-x)**2)
        score = np.mean(new["error"])

        new["pred"] = new["phrase"].apply(witlike.classify_intent)
        accuracy = len(new[new["pred"] == new["intent"]])/len(new)
        return(score, accuracy)



    def string_to_intent(self, strintent):
        for intent in self.intents:
            if intent.nom == strintent:
                return intent
        print(strintent.upper()," pas d'intent trouvé")

# je fais le test sur le csv écrit à la main
l = [lux, ratp, humour, music]
witlike = Wit (model, *l)
test = pd.read_excel("train_intent.xlsx", sep = ";")

# RENVOIE LA CLASSE RETENUE POUR CHAQUE SENTENCE
# test["pred"] = test["phrase"].apply(witlike.classify_intent)
# print(test)

# RENVOIE LE SCORE POUR CHAQUE INTENT
# test["pred"] = test["phrase"].apply(witlike.classify_intent)
# test["pred_lux"] = test["phrase"].apply(lux.classifier_score)
# test["pred_ratp"] = test["phrase"].apply(ratp.classifier_score)
# test["pred_music"] = test["phrase"].apply(music.classifier_score)
# test["pred_humour"] = test["phrase"].apply(humour.classifier_score)
# test.to_csv("test.csv")

# RENVOIE LE SCORE TOTAL DU TEST ET LE % DE PHRASES BIEN CLASSÉES
print(witlike.score(test))

from lemmatiseur.lemmatizeur import Lemmatiseur
lemmatiseur = Lemmatiseur()
print(lemmatiseur.lemmatize("marrante"))
