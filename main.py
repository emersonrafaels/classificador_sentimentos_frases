from pandas import read_csv

from model_emotion_classifier import Emotion_Classifier


def orchestra_model():

    # OBTENDO A BASE DE DADOS
    caminho_base = "BASES/BASE_TREINAMENTO.txt"
    base_dados = read_csv(caminho_base)

    result_pre_processing = Emotion_Classifier.orchestra_pre_processing_model(base_dados)

    result_model = Emotion_Classifier.orchestra_create_classifier(result_pre_processing, 1000, 30)