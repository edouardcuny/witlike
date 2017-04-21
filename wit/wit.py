# coding : utf-8

import os
import numpy as np
from intent import Intent
import pandas as pd
import toolkit as tk # module de nlp perso
from sklearn.ensemble import RandomForestClassifier # classifier
from sklearn.externals import joblib # dumper model

class Wit():
    def __init__(self, *args):
        self.intents = args # liste de tous les intents
        self.threshold = 0.7 # si score en dessous, intent pas retenu
        self.key_words = [] # liste des key_words des intents
        for intent in self.intents:
            self.key_words += intent.key_words
        self.key_words = list(set(self.key_words)) # supprime duplicates
        self.model = RandomForestClassifier()

    def new_intent(self, nom):
        '''
        Outil de création d'un nouvel intent.
        A développer.
        '''
        intent = Intent(word_to_vec = self.word_to_vec, nom = nom)
        self.intents.append(intent)

    def string_to_intent(self, strintent):
        for intent in self.intents:
            if intent.nom == strintent:
                return intent
        print(strintent.upper()," pas d'intent trouvé")

    def s2v(self, sentence):
        '''
        Sentence2Vec 'maison'
        Prend une phrase en entrée
        Renvoie un vecteur de la même longueur que les key_words de wit
        avec des 1 si le key_word est dans la phrase 0 sinon
        '''
        sentence = tk.prepare_sentence(sentence)
        vec = np.zeros(len(self.key_words))
        for i in range(len(self.key_words)):
            if self.key_words[i] in sentence:
                vec[i] = 1
        return vec

    def i2v(self, intent):
        '''
        Intent2Vec
        Prend un intent, renvoie un entier correspondant à l'index de
        cet intent dans self.intents
        '''
        for i in range(len(self.intents)):
            if self.intents[i].nom == intent:
                return i

    def v2i(self, i):
        '''
        Vector2Intent
        Prend un entier
        et le transforme en intent
        '''
        return(self.intents[i].nom)

    def fit(self, x_train, y_train):
        '''
        x_train = df.series de phrases
        y_train = df.series d'intent
        '''

        x_train = x_train.apply(lambda x : self.s2v(x))
        x_train = x_train.values.flatten().tolist()
        x_train = np.array(x_train)

        # print("x_train", x_train)

        y_train = y_train.apply(lambda x : self.i2v(x))
        y_train = y_train.values.flatten().tolist()
        y_train = np.array(y_train)

        # print("y_train", y_train)

        self.model.fit(x_train, y_train)
        return None


    def classify_intent_2(self, x_test):
        '''
        x_test = df.series phrase
        '''

        x_test = x_test.apply(lambda x : self.s2v(x))
        x_test = x_test.values.flatten().tolist()
        x_test = np.array(x_test)

        vec = self.model.predict(x_test)

        # on convertit chaque prev du format 1-hot vector en intent
        vec = vec.tolist()
        for i in range(len(vec)):
            vec[i] = self.v2i(vec[i])

        return(vec)

    def dump_model(self, name):
        joblib.dump(self.model, name)
