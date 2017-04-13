import psycopg2

class Lemmatiseur():

    def __init__(self):
        '''
        Connexion à la base de données postgres.
        Il faudra voir comment tout bien adapter.
        '''

        self.hostname = 'localhost'
        self.username = 'edouardcuny'
        self.database = 'lemmatiseur'
        self.conn = psycopg2.connect( host = self.hostname, user = self.username, dbname = self.database )
        self.cur = self.conn.cursor()

    def lemmatize(self, word):
        ''' Lemmatiseur :
        Si j'ai une entrée dans mon csv je renvoie le mot lemmatisé.
        Sinon je le laisse inchangé.
        '''

        try :
            self.cur.execute( "SELECT lemme FROM liste_lemme WHERE word = '{0}' ".format(word) )
            return(self.cur.fetchall()[0][0])

        except IndexError:
            # print("not in lemm")
            return(word)
