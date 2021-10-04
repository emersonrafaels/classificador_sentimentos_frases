import random
import string

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from spacy import displacy
from spacy.lang.pt.stop_words import STOP_WORDS


class Emotion_Classifier():

    def __init__(self):

        # 1 - DEFININDO A LINGUAGEM UTILIZADA NO MODELO
        self.language = "portuguese"

        # 2 - OBTENDO AS PONTUAÇÕES QUE SERÃO RETIRADAS DO MODELO
        self.pontuacoes = string.punctuation


    @staticmethod
    def read_csv(caminho_base):

        return pd.read_csv(caminho_base, encoding='utf-8')


    @staticmethod
    def get_infos_base(dataframe):

        print(dataframe.shape)


    def get_spacy_model(self):

        # INICIANDO O VALIDADOR
        validador = False

        # INICIANDO O MODELO DO SPACY
        pln = spacy.blank()

        # OBTENDO O MODELO
        if self.language == "portuguese":
            pln = spacy.load('pt_core_news_sm')
            validador = True
        elif self.language == "english":
            pln = spacy.load('en_core_news_sm')
            validador = True

        return validador, pln


    @staticmethod
    def pre_processing_dataframe(pln, texto, pontuacoes):

        # INICIANDO A LISTA QUE ARMAZENARÁ AS LEMMATIZAÇÕES
        lista = []

        # CONVERTENDO O TEXTO PARA LOWERCASE
        # É IMPORTANTE ESSE PADRÃO, PARA O SPACY RECONHECER OS TOKENS
        texto = texto.lower()
        documento = pln(texto)

        # PERCORRENDO TODOS OS TOKENS E OBTENDO AS LEMMARIZAÇÕES
        # OBTÉM A PALAVRA RAIZ
        for token in documento:
            lista.append(token.lemma_)

        # RETIRANDO STOP WORDS, PONTUAÇÕES E NÚMEROS
        texto_formatted = ' '.join([palavra for palavra in lista if palavra not in STOP_WORDS and palavra not in pontuacoes and not palavra.isdigit()])

        return texto_formatted


    @staticmethod
    def pre_processing_labels(emocao):

        # RETORNANDO UM DICT COM A EMOÇÃO
        if emocao.lower() == "alegria":
            return ({'ALEGRIA': True, 'MEDO': False})
        elif emocao == 'medo':
            return ({'ALEGRIA': False, 'MEDO': True})


    @staticmethod
    def get_list_label_target(dataframe, label, target):

        return dataframe[[label, target]].values.tolist()


    def orchestra_pre_processing_model(self, pln, dataframe):

        # INICIANDO O MODELO DO SPACY
        validador, pln = Emotion_Classifier.get_spacy_model(self)

        if validador:

            label = "texto_formatado"
            target = "emocao_formatado"

            # FORMATANDO OS TEXTOS
            # RETIRANDO PONTUAÇÕES E OBTENDO APENAS LEMMATIZAÇÃO
            dataframe['texto_formatado'] = dataframe['texto'].apply(lambda x: Emotion_Classifier.pre_processing_dataframe(pln, x,
                                                                                                                          self.pontuacoes))
            dataframe['emocao_formatado'] = dataframe['emocao'].apply(lambda x: Emotion_Classifier.pre_processing_labels(x))

            # OBTENDO UMA LISTA COM OS LABELS/TARGETS
            list_label_target = Emotion_Classifier.get_list_label_target(dataframe, label, target)

        return validador, list_label_target


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


