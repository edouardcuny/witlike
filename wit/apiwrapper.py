# coding : utf-8

from flask import Flask, redirect, url_for, request
import numpy as np
import pandas as pd

from wit import Wit

class Api():

    def __init__(self, wit_object):
        self.wit = wit_object
        self.app = Flask(__name__)

        @self.app.route('/classify_intent', methods=['POST'])
        def classify_intent():
            x_test = pd.Series(request.json['sentence'])
            return(self.wit.classify_intent_2(x_test)[0])
