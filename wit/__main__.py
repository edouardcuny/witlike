# coding : utf-8

import os
import pandas as pd
import pickle # pour dumper witlike object

from intent import Intent

import random
random.seed(7)

from wit import Wit

# importation du W2V
os.chdir("/Users/edouardcuny/Desktop/witlike/wit")

# Lumière
LUX = Intent(nom="lux")
LUX.train("allumer")
LUX.train("lumière")
LUX.train("lampe")
LUX.train("éteindre")

# RATP
RATP = Intent(nom="ratp")
RATP.train("trajet")
RATP.train("itinéraire")
RATP.train("aller")
RATP.train("métro")

# Musique
MUSIC = Intent(nom="music")
MUSIC.train("jouer")
MUSIC.train("mettre")
MUSIC.train("couper")
MUSIC.train("chanson")
MUSIC.train("son")
MUSIC.train("musique")
MUSIC.train("envoyer")

# HUMOUR
HUMOUR = Intent(nom="humour")
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
witlike = Wit(*l)

# entraînement
data = pd.read_excel("train_intent_only_intents.xlsx", sep = ";")
data = data.sample(frac=1).reset_index(drop=True)

x_train = data.ix[:, 0]
y_train = data.ix[:, 1]
witlike.fit(x_train, y_train)

pickle.dump( witlike, open( "wit.p", "wb" ) )
