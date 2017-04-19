import psycopg2

class Lemmatiseur():

    def __init__(self):
        '''
        Connexion à la base de données postgres.
        Il faudra voir comment tout bien adapter.
        '''

        self.hostname = 'localhost'
        self.username = 'edouardcuny'
        self.database = 'Lem'
        self.conn = psycopg2.connect( host = self.hostname, user = self.username, dbname = self.database )
        self.cur = self.conn.cursor()

    def lemmatize(self, word):
        ''' Lemmatiseur :
        Si j'ai une entrée dans mon csv je renvoie le mot lemmatisé.
        Sinon je le laisse inchangé.
        '''

        # cgram = 'VER' est une règle arbitraire de priorité pour traiter le cas
        # 'mets de la musique' mets = NOM & VERBE, je mets la priorité
        # sur le verbe

        try :
            self.cur.execute( "SELECT lemme FROM lemme_table WHERE word = '{0}' AND cgram = 'VER'  LIMIT 1;".format(word) )
            return(self.cur.fetchall()[0][0])

        except IndexError:
            try :
                self.cur.execute( "SELECT lemme FROM lemme_table WHERE word = '{0}'  LIMIT 1;".format(word) )
                return(self.cur.fetchall()[0][0])

            except IndexError:
                return(word)
