# coding : utf-8

import word2vec # word2vec déjà entraîné
import os
import pandas as pd

from intent import Intent

import random
random.seed(7)

from wit import Wit

# importation du W2V
os.chdir("/Users/edouardcuny/Desktop/witlike/wit")
# model = word2vec.load('frWiki_no_phrase_no_postag_1000_skip_cut100.bin')
MODEL = word2vec.load('frWac_no_postag_no_phrase_700_skip_cut50.bin')

# Lumière
LUX = Intent(word_to_vec=MODEL, nom="lux")
LUX.train("allumer")
LUX.train("lumière")
LUX.train("lampe")
LUX.train("éteindre")

# RATP
RATP = Intent(word_to_vec=MODEL, nom="ratp")
RATP.train("trajet")
RATP.train("itinéraire")
RATP.train("aller")
RATP.train("métro")

# Musique
MUSIC = Intent(word_to_vec=MODEL, nom="music")
MUSIC.train("jouer")
MUSIC.train("mettre")
MUSIC.train("couper")
MUSIC.train("chanson")

# HUMOUR
HUMOUR = Intent(word_to_vec=MODEL, nom="humour")
HUMOUR.train("blague")
HUMOUR.train("vanne")
HUMOUR.train("rire")
HUMOUR.train("marrer")
HUMOUR.train("rigoler")
HUMOUR.train("drôle")
HUMOUR.train("marrant")
HUMOUR.train("jouer")


# je crée mon objet wit (à mettre dans un main après)
l = [LUX, RATP, HUMOUR, MUSIC]
witlike = Wit(MODEL, *l)

# entraînement
data = pd.read_excel("train_intent_only_intents.xlsx", sep = ";")
data = data.sample(frac=1).reset_index(drop=True)

x_train = data.ix[:30, 0]
y_train = data.ix[:30, 1]
witlike.fit(x_train, y_train)
witlike.dump_model()

# classification
x_test = data.ix[30:, 0]
y_test = data.ix[30:, 1]

pred = witlike.classify_intent_2(x_test)
x_test = pd.DataFrame(x_test)
x_test["labels"] = y_test
x_test["pred"] = pred
x_test.to_csv("rf.csv")



# RENVOIE LA CLASSE RETENUE POUR CHAQUE SENTENCE
# test["pred"] = test["phrase"].apply(witlike.classify_intent)
# print(test)

# RENVOIE LE SCORE POUR CHAQUE INTENT
# test["pred"] = test["phrase"].apply(witlike.classify_intent)
# test["pred_LUX"] = test["phrase"].apply(LUX.classifier_score)
# test["pred_RATP"] = test["phrase"].apply(ratp.classifier_score)
# test["pred_MUSIC"] = test["phrase"].apply(music.classifier_score)
# test["pred_HUMOUR"] = test["phrase"].apply(HUMOUR.classifier_score)
# print(test)
# test.to_csv("test1.csv")

# RENVOIE LE SCORE TOTAL DU TEST ET LE % DE PHRASES BIEN CLASSÉES
# print(witlike.score(test))