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
__data_atualizacao__ = "03/10/2021"


from dynaconf import settings

from model_emotion_classifier import Emotion_Classifier
from UTILS import generic_functions


def orchestra_model():

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


    # OBTENDO A BASE DE DADOS DE TREINAMENTO
    path_data_train_dir = settings.PATH_DATA_TRAIN

    # INICIALIZADO A CLASSE DO MODELO
    orchest_model = Emotion_Classifier()

    # REALIZANDO A LEITURA DA BASE DE DADOS DE TREINAMENTO
    validador, train_database = generic_functions.read_csv(path_data_train_dir)

    if validador:

        # REALIZANDO O PRÉ-PROCESSAMENTO DA BASE DE TREINAMENTO
        validador, result_pre_processing = orchest_model.orchestra_pre_processing_model(train_database)

        if validador:

            # REALIZANDO O TREINAMENTO DO MODELO
            validador, model, result_model = orchest_model.orchestra_create_classifier(result_pre_processing,
                                                                                       settings.EPOCHS,
                                                                                       settings.BATCH_SIZE,
                                                                                       settings.VERBOSE)

            if validador:

                # SALVANDO O MODELO PARA USO FUTURO
                validador = orchest_model.save_model(model, settings.DIR_SAVE)

                if validador:
                    print("MODELO TREINADO COM SUCESSO - {}".format(generic_functions.obtem_date_time("%d-%m-%Y %H:%M:%S")))


if __name__ == '__main__':
    orchestra_model()