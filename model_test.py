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
import spacy
from spacy.lang.pt.stop_words import STOP_WORDS


class Emotion_Test():

    def __init__(self, model):

        # 1 - DEFININDO O MODELO A SER UTILIZADO
        self.model = model

        # 2 - OBTENDO AS PONTUAÇÕES QUE SERÃO RETIRADAS DO MODELO
        self.punctuations = string.punctuation


    @staticmethod
    def get_spacy_model(model_to_read):

        """

            INICIA O MODELO SPACY COM BASE NO MODELO ENVIADO

            # Arguments
                model_to_read                  - Reqired : Modelo a ser utilizado (Spacy)

            # Returns
                validador                       - Required : Validação da função (Boolean)
                pln                             - Required : Modelo spacy (Spacy)

        """

        # INICIANDO O VALIDADOR
        validador = False

        # INICIANDO O MODELO DO SPACY
        pln = None

        # OBTENDO O MODELO
        try:
            pln = spacy.load(model_to_read)
        except Exception as ex:
            print("ERRO NA FUNÇÃO {} - {}".format(stack()[0][3], ex))

        return validador, pln


    def pre_processing_dataframe(self, text):

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

            # Returns
                validador                       - Required : Validação da função (Boolean)
                pln                             - Required : Modelo spacy (Spacy)

        """

        # INICIANDO A LISTA QUE ARMAZENARÁ AS LEMMATIZAÇÕES
        lista = []

        try:
            # CONVERTENDO O TEXTO PARA LOWERCASE
            # É IMPORTANTE ESSE PADRÃO, PARA O SPACY RECONHECER OS TOKENS
            texto = text.lower()
            documento = self.model(texto)

            # PERCORRENDO TODOS OS TOKENS E OBTENDO AS LEMMARIZAÇÕES
            # OBTÉM A PALAVRA RAIZ
            for token in documento:
                lista.append(token.lemma_)

            # RETIRANDO STOP WORDS, PONTUAÇÕES E NÚMEROS
            texto_formatted = ' '.join([palavra for palavra in lista if
                                        palavra not in STOP_WORDS and palavra not in self.punctuations and not palavra.isdigit()])

            return texto_formatted

        except Exception as ex:
            print("ERRO NA FUNÇÃO {} - {}".format(stack()[0][3], ex))

        return text


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
        prediction = settings.TEST_RESULT_DEFAULT

        try:
            prediction = model(text).cats
        except Exception as ex:
            print("ERRO NA FUNÇÃO {} - {}".format(stack()[0][3], ex))

        return prediction


    def orchestra_predictions(self, data_test):

        """

            ORQUESTRA A PREDIÇÃO DO MODELO EM TODA A BASE DE TESTES.

            # Arguments
                data_test                       - Required : Base contendo
                                                             os dados de teste (DataFrame)

            # Returns
                prediction                      - Required : Predições do modelo (Dict)

        """

        for value in data_test:
            print(value)




