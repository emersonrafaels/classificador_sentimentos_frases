import random
import string

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from spacy import displacy
from spacy.lang.pt.stop_words import STOP_WORDS


def read_csv(caminho_base):

    return pd.read_csv(caminho_base, encoding = 'utf-8')


def get_infos_base(dataframe):

    print(dataframe.shape)


def pre_processing_dataframe(texto, pontuacoes):

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


def pre_processing_labels(emocao):

    # RETORNANDO UM DICT COM A EMOÇÃO
    if emocao.lower() == "alegria":
        return ({'ALEGRIA': True, 'MEDO': False})
    elif emocao == 'medo':
        return ({'ALEGRIA': False, 'MEDO': True})


def get_list_label_target(dataframe, label, target):

    return dataframe[[label, target]].values.tolist()


def orchestra_pre_processing_model(dataframe):

    label = "texto_formatado"
    target = "emocao_formatado"

    # OBTENDO AS PONTUAÇÕES QUE SERÃO RETIRADAS DO MODELO
    pontuacoes = string.punctuation

    dataframe['texto_formatado'] = dataframe['texto'].apply(lambda x: pre_processing_dataframe(x, pontuacoes))
    dataframe['emocao_formatado'] = dataframe['emocao'].apply(lambda x: pre_processing_labels(x))

    # OBTENDO UMA LISTA COM OS LABELS/TARGETS

    list_label_target = get_list_label_target(dataframe, label, target)

    return list_label_target


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


def view_losses(history_losses):

    plt.plot(history_losses)
    plt.title('Progressão do erro')
    plt.xlabel('Épocas')
    plt.ylabel('Erro')
    plt.show()


def orchestra_create_classifier(model_data, epochs, batch_size):

    # INICIANDO O MODELO
    validador, model = init_model_classifier()

    if validador:

        # REALIZANDO O TREINAMENTO
        history_epochs, history_losses = start_train_model(model, model_data, epochs, batch_size)

        print(history_epochs)
        print(history_losses)

        # VISUALIZANDO A PROGRESSÃO DOS ERROS
        view_losses(history_losses)


def get_entities(engine, text):

    # PARSEANDO O TEXTO UTILIZANDO A ENGINE DE NLP
    doc = engine(text)

    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_, sep=" - ")


def view_entities(engine, text):

    # PARSEANDO O TEXTO UTILIZANDO A ENGINE DE NLP
    doc = engine(text)

    displacy.serve(doc, style="ent")



# OBTENDO A BASE DE DADOS
caminho_base = "../base_treinamento.txt"
base_dados = read_csv(caminho_base)

# OBTENDO O MODELO
pln = spacy.load('pt_core_news_sm')

result_pre_processing = orchestra_pre_processing_model(base_dados)

result_model = orchestra_create_classifier(result_pre_processing, 1000, 30)


