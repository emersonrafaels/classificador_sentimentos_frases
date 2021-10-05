"""

    CLASSE DE TESTES DO
    CLASSIFICADOR DE EMOÇÕES UTILIZANDO FRASES EM PORTUGUÊS,

    SE UTILIZA DE UM MODELO PARA CLASSIFICAR UM TEXTO EM:

        1) POSITIVO
        2  NEGATIVO

    # Arguments
        model                          - Required : Diretório do modelo
                                                    para ser utilizado (String)
        data_test                      - Required : Dataframe de testes (Dataframe)
    # Returns
        model_result                   - Required : Resultado do modelo (Dict)

"""

__version__ = "1.0"
__author__ = """Emerson V. Rafael (EMERVIN)"""
__data_atualizacao__ = "04/10/2021"


from inspect import stack
import string

from dynaconf import settings
import numpy as np
import spacy
from spacy.lang.pt.stop_words import STOP_WORDS


class Emotion_Test():

    def __init__(self, model):

        # 1 - DEFININDO O MODELO A SER UTILIZADO
        self.model = model

        # 2 - DEFININDO A LINGUAGEM UTILIZADA NO MODELO SPACY (PARA PRÉ PROCESSAMENTO)
        self.language = settings.LANGUAGE

        # 3 - OBTENDO AS PONTUAÇÕES QUE SERÃO RETIRADAS DO MODELO
        self.punctuations = string.punctuation

        # 4 - NOME DAS COLUNAS DO DATAFRAME
        self.column_text = settings.NAME_COLUMN_TEXT
        self.column_emotion = settings.NAME_COLUMN_EMOTION

        # 5 - NOME DAS COLUNAS QUE RECEBERÃO OS RESULTADOS DE PRÉ PROCESSAMENTO
        self.variables = settings.VARIABLES
        self.prediction_class_column = settings.PREDICTION_CLASS_COLUMN
        self.predictions_column = settings.PREDICTION_COLUMN


    def get_spacy_model(self):

        """

            INICIA O MODELO SPACY COM BASE NA LINGUAGEM DEFINIDA.

            # Arguments

            # Returns
                validador                       - Required : Validação da função (Boolean)
                pln                             - Required : Modelo spacy (Spacy)

        """

        # INICIANDO O VALIDADOR
        validador = False

        # INICIANDO O MODELO DO SPACY
        pln = None

        # OBTENDO O MODELO
        if self.language == "portuguese":
            pln = spacy.load('pt_core_news_sm')
            validador = True
        elif self.language == "english":
            pln = spacy.load('en_core_news_sm')
            validador = True

        return validador, pln


    @staticmethod
    def get_model(model_to_read):

        """

            INICIA O MODELO SPACY COM BASE NO MODELO ENVIADO

            # Arguments
                model_to_read                  - Reqired : Modelo a ser utilizado (Spacy)

            # Returns
                validador                       - Required : Validação da função (Boolean)
                model                           - Required : Modelo spacy (Spacy)

        """

        # INICIANDO O VALIDADOR
        validador = False

        # INICIANDO O MODELO DO SPACY
        model = None

        # OBTENDO O MODELO
        try:
            model = spacy.load(model_to_read)

            validador = True
        except Exception as ex:
            print("ERRO NA FUNÇÃO {} - {}".format(stack()[0][3], ex))

        return validador, model


    def pre_processing_dataframe(text, pln, punctuations):

        """

            REALIZA O PRÉ PROCESSAMENTO DO DATAFRAME.

            PARA PRÉ-PROCESSAMENTO SERÃO UTILIZADAS AS TÉCNICAS:
                1) LEMATIZAÇÃO
                2) STOP WORDS
                3) RETIRAR PONTUAÇÕES
                4) TOKENIZAÇÃO.

            # Arguments
                text                            - Required : Texto para realizar o pré-processamento (String)
                pln                             - Required : Modelo spacy (Spacy)
                pontuactions                    - Required : Pontuações a serem retiradas (List)

            # Returns
                validador                       - Required : Validação da função (Boolean)
                pln                             - Required : Modelo spacy (Spacy)

        """

        # INICIANDO A LISTA QUE ARMAZENARÁ AS LEMMATIZAÇÕES
        lista = []

        try:
            # CONVERTENDO O TEXTO PARA LOWERCASE
            # É IMPORTANTE ESSE PADRÃO, PARA O SPACY RECONHECER OS TOKENS
            text_lower = text.lower()
            document = pln(text_lower)

            # PERCORRENDO TODOS OS TOKENS E OBTENDO AS LEMMARIZAÇÕES
            # OBTÉM A PALAVRA RAIZ
            for token in document:
                lista.append(token.lemma_)

            # RETIRANDO STOP WORDS, PONTUAÇÕES E NÚMEROS
            texto_formatted = ' '.join([palavra for palavra in lista if
                                        palavra not in STOP_WORDS and palavra not in punctuations and not palavra.isdigit()])

            return texto_formatted

        except Exception as ex:
            print("ERRO NA FUNÇÃO {} - {}".format(stack()[0][3], ex))

        return text


    def orchestra_pre_processing_model(self, dataframe):

        """

            ORQUESTRA O PRÉ PROCESSAMENTO DO DATAFRAME.

            PARA PRÉ-PROCESSAMENTO SERÃO UTILIZADAS AS TÉCNICAS:
                1) LEMATIZAÇÃO
                2) STOP WORDS
                3) RETIRAR PONTUAÇÕES
                4) TOKENIZAÇÃO.

            # Arguments
                text                            - Required : Texto para realizar o pré-processamento (String)
                pln                             - Required : Modelo spacy (Spacy)
                pontuactions                    - Required : Pontuações a serem retiradas (List)

            # Returns
                validador                       - Required : Validação da função (Boolean)
                pln                             - Required : Modelo spacy (Spacy)

        """

        # INICIANDO O MODELO DO SPACY
        # UTILIZADO PARA O PRÉ-PROCESSAMENTO
        validador, pln = Emotion_Test.get_spacy_model(self)

        if validador:

            # FORMATANDO OS TEXTOS
            # RETIRANDO PONTUAÇÕES E OBTENDO APENAS LEMMATIZAÇÃO
            dataframe[self.variables] = np.vectorize(Emotion_Test.pre_processing_dataframe)(dataframe[self.column_text],
                                                                                            pln,
                                                                                            self.punctuations)

            # FORMATANDO AS EMOÇÕES PARA UPPERCASE
            dataframe[self.column_emotion] = dataframe[self.column_emotion].apply(lambda x: str(x).upper())

        return validador, dataframe


    @staticmethod
    def get_prediction(model, text):

        """

            OBTÉM A PREDIÇÃO DO MODELO NO TEXTO (STRING) ENVIADA.

            # Arguments
                model                           - Required : Modelo spacy (Spacy)
                text                            - Required : Texto para realizar a predição (String)

            # Returns
                prediction                      - Required : Predição do modelo (Dict)

        """

        # INICIANDO A VARIÁVEL QUE ARMAZENARÁ O RESULTADO DA PREDIÇÃO
        # INICIA COM O VALOR DEFAULT
        predictions = settings.TEST_RESULT_DEFAULT
        prediction_class = settings.CLASS_RESULT_DEFAULT

        try:
            model_prediction = model(text).cats

            prediction_class, predictions = max(model_prediction, key=model_prediction.get), model_prediction
        except Exception as ex:
            print("ERRO NA FUNÇÃO {} - {}".format(stack()[0][3], ex))

        return prediction_class, predictions


    def orchestra_predictions(self, data_test):

        """

            ORQUESTRA A PREDIÇÃO DO MODELO EM TODA A BASE DE TESTES.

            # Arguments
                data_test                       - Required : Base contendo
                                                             os dados de teste (DataFrame)

            # Returns
                prediction                      - Required : Predições do modelo (Dict)

        """

        # INICIANDO O MODELO PARA SER TESTADO
        validador, model = Emotion_Test.get_model(self.model)

        if validador:

            # REALIZANDO AS PREDIÇÕES
            data_test[self.prediction_class_column], data_test[self.predictions_column] = np.vectorize(Emotion_Test.get_prediction)(model, data_test[self.variables])

        return validador, data_test




