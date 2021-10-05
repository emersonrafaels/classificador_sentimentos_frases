import datetime
from inspect import stack
import time

import pandas as pd


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


def obtem_date_time(tipo_retorno):

    """

        OBTÉM TODOS OS POSSÍVEIS RETORNOS DE DATA E TEMPO.

        # Arguments
            tipo_retorno                    - Required : Formato de retorno. (String)

        # Returns

    """

    """%Y/%m/%d %H:%M:%S | %Y-%m-%d %H:%M:%S
    Dia: %d
    Mês: %
    Ano: %Y
    Data: %Y/%m/%d

    Hora: %H
    Minuto: %M
    Segundo: %S"""

    try:
        ts = time.time()
        stfim = datetime.datetime.fromtimestamp(ts).strftime(tipo_retorno)

        return stfim
    except Exception as ex:
        print("ERRO NA FUNÇÃO {} - {}".format(stack()[0][3], ex))