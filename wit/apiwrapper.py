# coding : utf-8

from flask import Flask, redirect, url_for, request, jsonify

import numpy as np
import pandas as pd
import pickle

from wit import Wit

class Api():

    def __init__(self, wit_object):
        self.wit = wit_object
        self.app = Flask(__name__)

        @self.app.route('/classify_intent', methods=['POST'])
        def classify_intent():
            x_test = pd.Series(request.json['sentence'])
            return(self.wit.classify_intent_2(x_test)[0])

        @self.app.route('/get_intents')
        def get_intents():
            l = []
            for intent in self.wit.intents:
                l.append(intent.nom)
            return jsonify(intents = l)

        @self.app.route('/train', methods = ['POST'])
        def train():
            try:
                x_train = pd.Series(request.json['x_train'])
                y_train = pd.Series(request.json['y_train'])

            except:
                return("Input of Post not good format")

            self.wit.fit(x_train, y_train)
            pickle.dump( self.wit, open( "wit.p", "wb" ))
            return("The model was just trained")
