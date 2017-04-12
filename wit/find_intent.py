# coding : utf-8

import word2vec # word2vec déjà entraîné
import os
import numpy as np
from intent import Intent

# importation du W2V
os.chdir("/Users/edouardcuny/Desktop/wit")
model = word2vec.load('frWiki_no_phrase_no_postag_1000_skip_cut100.bin')

# Lumière
lux = Intent(word_to_vec = model)

lux.train("allumer")
lux.train("lumière")
lux.train("lampe")
lux.train("éteindre")

# print(lux)

test1 = "elvis est un grand chanteur" # false facile
test2 = "elvis allumer la lumière dans la grande salle" # vrai avec un peu de bruit
test3 = "éteindre la lampe" # vrai simple
test4 = "jouer allumer le feu de Johnny"
test5 = "allumer la lumière petit appareil nommé après un chanteur connu"


print(lux.classifier(test1), "  False Facile")
print(lux.classifier(test2), "  True Noise")
print(lux.classifier(test3), "  True Easy")
print(lux.classifier(test4), "  False Hard")
print(lux.classifier(test5), "  True Lot of Noise")

# Trajet
trajet = Intent(word_to_vec = model)

trajet.train("trajet")
trajet.train("itinéraire")
trajet.train("aller")

test1 = "elvis est un grand chanteur" # false facile
test2 = "elvis dire moi comment aller au marché" # vrai avec un peu de bruit
test3 = "quel est l'itinéraire pour aller à république" # vrai simple

print("\n")
print(trajet.classifier(test1), "  False Facile")
print(trajet.classifier(test2), "  True Noise")
print(trajet.classifier(test3), "  True Easy")
print("\n")

# Compétition
threshold = 0.8
test1 = "elvis est un grand chanteur"
test2 = "elvis dire moi comment aller au marché"
test3 = "quel est l'itinéraire pour aller à république"
test4 = "jouer allumer le feu de Johnny"
test5 = "allumer la lumière petit appareil nommé après un chanteur connu"
test6 = "elvis allumer la lumière dans la grande salle"
test7 = "éteindre la lampe"
l = [test1, test2, test3, test4, test5, test6, test7]

for test in l:
    if lux.classifier(test)<threshold and trajet.classifier(test)<threshold:
        print(0, end = " ")
    elif lux.classifier(test) > trajet.classifier(test):
        print("L", end = " ")
    elif lux.classifier(test) < trajet.classifier(test):
        print("T", end = " ")

print("expected output : 0 T T 0 L L L ")
