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


from dynaconf import settings

from model_test import Emotion_Test
from UTILS import generic_functions


def orchestra_model(model, path_data_test_dir):

    """

        CLASSE DE TESTES DO
        CLASSIFICADOR DE EMOÇÕES UTILIZANDO FRASES EM PORTUGUÊS,

        SE UTILIZA DE UM MODELO PARA CLASSIFICAR UM TEXTO EM:

            1) POSITIVO
            2  NEGATIVO

        # Arguments
            model                          - Required : Diretório do modelo
                                                        para ser utilizado (String)
            path_data_test_dir             - Required : Dataframe de testes (Dataframe)

        # Returns
            model_result                   - Required : Resultado do modelo (Dict)

    """

    # INICIALIZADO A CLASSE DO MODELO
    orchest_test_model = Emotion_Test(model)

    # REALIZANDO A LEITURA DA BASE DE DADOS DE TESTES
    validador, test_database = generic_functions.read_csv(path_data_test_dir)

    if validador:

        # REALIZANDO O PRÉ-PROCESSAMENTO DA BASE DE TESTES
        validador, result_test_pre_processing = orchest_test_model.orchestra_pre_processing_model(test_database)

        if validador:

            # REALIZANDO O TESTE DO MODELO
            prediction_class, result_model = orchest_test_model.orchestra_predictions(result_test_pre_processing)

            if validador:

                # SALVANDO O RESULTADO
                validador = generic_functions.save_excel(result_model, settings.DIR_RESULT_MODEL_SAVE)

                if validador:
                    print("MODELO TESTADO COM SUCESSO - {}".format(generic_functions.obtem_date_time("%d/%m/%Y %H:%M:%S")))


if __name__ == '__main__':

    model = "RESULTADOS/MODEL"
    path_data_test_dir = "BASES/BASE_TESTE.txt"

    orchestra_model(model, path_data_test_dir)