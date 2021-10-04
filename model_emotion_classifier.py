"""

    CLASSIFICADOR DE EMOÇÕES UTILIZANDO FRASES EM PORTUGUÊS,
    UTILIZANDO 100% OS MODELOS DE MACHINE LEARNING
    DISPONIBILIZADOS PELO PRÓPRIO SPACY.

    PARA PRÉ-PROCESSAMENTO SERÃO UTILIZADAS AS TÉCNICAS:
        1) LEMATIZAÇÃO
        2) STOP WORDS
        3) RETIRAR PONTUAÇÕES
        4) TOKENIZAÇÃO.


    # Arguments
        path_data_train_dir            - Required : Base de dados de
                                                    treinamento a ser utilizada (String)
        path_data_test_dir             - Optional : Base de dados de
                                                    teste a ser utilizada (String)
    # Returns
        model_result                   - Required : Resultado do modelo (Model)

"""

__version__ = "1.0"
__author__ = """Emerson V. Rafael (EMERVIN)"""
__data_atualizacao__ = "03/07/2021"


from inspect import stack
import random
import string

from dynaconf import settings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from spacy import displacy
from spacy.lang.pt.stop_words import STOP_WORDS


class Emotion_Classifier():

    def __init__(self):

        # 1 - DEFININDO A LINGUAGEM UTILIZADA NO MODELO
        self.language = settings.LANGUAGE

        # 2 - OBTENDO AS PONTUAÇÕES QUE SERÃO RETIRADAS DO MODELO
        self.pontuacoes = string.punctuation

        # 3 - NOME DAS COLUNAS DO DATAFRAME
        self.column_text = settings.NAME_COLUMN_TEXT
        self.column_emotion = settings.NAME_COLUMN_EMOTION

        # 4 - NOME DAS COLUNAS QUE RECEBERÁO OS RESULTADOS DE PRÉ PROCESSAMENTO
        self.variables = settings.VARIABLES = "texto_formatado"
        self.target = settings.TARGET = "emocao_formatado"


    @staticmethod
    def read_csv(data_dir):

        """

            REALIZA LEITURA DA BASE (CSV)

            # Arguments
                path_data_train_dir            - Required : Base de dados de
                                                            treinamento a ser utilizada (String)
                path_data_test_dir             - Optional : Base de dados de
                                                            teste a ser utilizada (String)
            # Returns
                validador                      - Required : Validação da função (Boolean)
                model_result                   - Required : Resultado do modelo (Model)

        """

        # INICIANDO O VALIDADOR
        validador = False

        # INICIANDO O DATAFRAME DE RESULTADO DA LEITURA
        dataframe = pd.DataFrame()

        try:
            dataframe = pd.read_csv(data_dir, encoding='utf-8')

            validador = True
        except Exception as ex:
            print("ERRO NA FUNÇÃO {} - {}".format(stack()[0][3], ex))

        return validador, dataframe


    @staticmethod
    def get_infos_base(dataframe):

        """

            OBTÉM A QUANTIDADE DE LINHAS E COLUNAS DA BASE

            # Arguments
                dataframe                       - Required : Base de dados (DataFrame)

            # Returns
                validador                       - Required : Validação da função (Boolean)
                dataframe_shape                 - Required : Linhas e colunas do dataframe (Tuple)

        """

        # INICIANDO O VALIDADOR
        validador = False

        # INICIANDO O DATAFRAME DE RESULTADO DA LEITURA
        dataframe_shape = (None, None)

        try:
            if isinstance(dataframe, pd.DataFrame):
                dataframe_shape = dataframe.shape
        except Exception as ex:
            print("ERRO NA FUNÇÃO {} - {}".format(stack()[0][3], ex))

        return validador, dataframe_shape


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
    def pre_processing_dataframe(text, pln, pontuactions):

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

        # CONVERTENDO O TEXTO PARA LOWERCASE
        # É IMPORTANTE ESSE PADRÃO, PARA O SPACY RECONHECER OS TOKENS
        texto = text.lower()
        documento = pln(texto)

        # PERCORRENDO TODOS OS TOKENS E OBTENDO AS LEMMARIZAÇÕES
        # OBTÉM A PALAVRA RAIZ
        for token in documento:
            lista.append(token.lemma_)

        # RETIRANDO STOP WORDS, PONTUAÇÕES E NÚMEROS
        texto_formatted = ' '.join([palavra for palavra in lista if palavra not in STOP_WORDS and palavra not in pontuactions and not palavra.isdigit()])

        return texto_formatted


    @staticmethod
    def pre_processing_labels(emotion):

        """

            FORMATANDO OS LABELS DOS TARGETS DO MODELO.

            RETORNA UM DICIONÁRIO CONTENDO:
                SE ALEGRIA, O DICT DEVE SER TRUE PARA ALEGRIA E FALSE PARA MEDO
                SE ALEGRIA, O DICT DEVE SER TRUE PARA MEDO E FALSE PARA ALEGRIA.

            # Arguments
                emotion                             - Required : Emoção atual (String)

            # Returns
                result_label_dict                   - Required : Dict de target (Dict)

        """

        # RETORNANDO UM DICT COM A EMOÇÃO
        if emotion.lower() == "alegria":
            return ({'ALEGRIA': True, 'MEDO': False})
        elif emotion == 'medo':
            return ({'ALEGRIA': False, 'MEDO': True})
        else:
            return ({'ALEGRIA': False, 'MEDO': False})


    @staticmethod
    def get_dataframe_to_list_label_target(dataframe, variables, target):

        """

            CONVERTENDO O DATAFRAME EM UMA LIST BIDIMENSIONAL.

            1) RECEBE LABEL E TARGET
            2) FILTRA APENAS AS COLUNAS DESEJADAS NO DATAFRAME
            3) CONVERTE O DATAFRAME EM UMA LISTA BIDIMENSIONAL.

            # Arguments
                dataframe                       - Required : Dados para o modelo (DataFrame)
                variables                       - Required : Variáveis para o modelo (String)
                target                          - Required : Objetivo de cada conjunto de variáveis (String)

            # Returns
                result_label_dict                   - Required : Dict de target (Dict)

        """

        # INICIANDO O VALIDADOR
        validador = False

        # INICIANDO A LISTA BIDIMENSIONAL
        list_data = []

        # OBTENDO OS DADOS
        try:
            list_data = dataframe[[variables, target]].values.tolist()
        except Exception as ex:
            print("ERRO NA FUNÇÃO {} - {}".format(stack()[0][3], ex))

        return validador, list_data


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
        validador, pln = Emotion_Classifier.get_spacy_model(self)

        if validador:

            # FORMATANDO OS TEXTOS
            # RETIRANDO PONTUAÇÕES E OBTENDO APENAS LEMMATIZAÇÃO
            dataframe[self.variables] = np.vectorize(Emotion_Classifier.pre_processing_dataframe)(dataframe[self.column_text],
                                                                                                     pln,
                                                                                                     self.pontuacoes)
            dataframe[self.target] = dataframe[self.column_emotion].apply(lambda x: Emotion_Classifier.pre_processing_labels(x))

            # OBTENDO UMA LISTA COM OS LABELS/TARGETS
            list_data_model = Emotion_Classifier.get_dataframe_to_list_label_target(dataframe,
                                                                                    self.variables,
                                                                                    self.target)

        return validador, list_data_model


    @staticmethod
    def init_model_classifier():

        # INICIANDO O VALIDADOR DA FUNÇÃO
        validador = False

        # INICIANDO O MODELO
        modelo = spacy.blank('pt')

        try:
            # CREATE A PIPELINE
            categorias = modelo.create_pipe("textcat")

            # ADICIONANDO OS LABELS DESEJADOS
            categorias.add_label("ALEGRIA")
            categorias.add_label("MEDO")

            # ADICIONANDO ESSES LABELS AO PIPELINE
            modelo.add_pipe(categorias)

            validador = True

        except Exception as ex:
            print(ex)

        return validador, modelo


    @staticmethod
    def start_train_model(model, model_data, epochs, batch_size):

        # INICIANDO A LISTA QUE ARMAZENARÁ O RESULTADO DE CADA
        # INTERAÇÃO DE TREINAMENTO DO MODELO
        result_epochs_history = []
        result_epochs_history_losses = []

        # INICIANDO A LISTA QUE ARMAZENARÁ OS ERROS DE CADA ITERAÇÃO
        losses = {}

        model.begin_training()

        for epoch in range(epochs):

            # EMBARALHANDO OS DADOS
            # ISSO REORGANIZARÁ A ORDEM DOS ITENS DOS DADOS DE TREINAMENTO
            random.shuffle(model_data)

            # PERCORRENDO CADA UM DOS BATCHS
            # QUANTIDADE DE BATCHS = TAMANHO_DO_MODELO/BATCHS_SIZE
            for batch in spacy.util.minibatch(model_data, batch_size):
                textos = [model(texto) for texto, entities in batch]
                annotations = [{'cats': entities} for texto, entities in batch]
                model.update(textos, annotations, losses=losses)

            # A CADA
            if epoch % 10 == 0:
                print(losses)
                result_epochs_history.append(losses)
                result_epochs_history_losses.append(losses.get("textcat"))

            return result_epochs_history, result_epochs_history_losses


    @staticmethod
    def view_losses(history_losses):

        plt.plot(history_losses)
        plt.title('Progressão do erro')
        plt.xlabel('Épocas')
        plt.ylabel('Erro')
        plt.show()


    @staticmethod
    def orchestra_create_classifier(model_data, epochs, batch_size):

        # INICIANDO O MODELO
        validador, model = Emotion_Classifier.init_model_classifier()

        if validador:

            # REALIZANDO O TREINAMENTO
            history_epochs, history_losses = Emotion_Classifier.start_train_model(model, model_data, epochs, batch_size)

            print(history_epochs)
            print(history_losses)

            # VISUALIZANDO A PROGRESSÃO DOS ERROS
            Emotion_Classifier.view_losses(history_losses)


    @staticmethod
    def get_entities(engine, text):

        # PARSEANDO O TEXTO UTILIZANDO A ENGINE DE NLP
        doc = engine(text)

        for ent in doc.ents:
            print(ent.text, ent.start_char, ent.end_char, ent.label_, sep=" - ")


    @staticmethod
    def view_entities(engine, text):

        # PARSEANDO O TEXTO UTILIZANDO A ENGINE DE NLP
        doc = engine(text)

        displacy.serve(doc, style="ent")


