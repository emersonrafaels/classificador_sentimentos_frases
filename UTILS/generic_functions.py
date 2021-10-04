import datetime
from inspect import stack
import time

def obtem_date_time(self, tipo_retorno):

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