import apiwrapper
from sklearn.externals import joblib
import pandas as pd
import pickle

# je crée un object wit en utilisant l'initializer
witlike = pickle.load( open('wit.p', 'rb'))

app = apiwrapper.Api(wit_object = witlike)
application = app.app

if __name__ == "__main__":
    application.run(debug = True)
