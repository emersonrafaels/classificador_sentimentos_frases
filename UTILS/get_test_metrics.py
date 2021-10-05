from inspect import stack

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


def get_test_report(targets, predictions):

    """

        CRIE UM RELATÓRIO DE TEXTO MOSTRANDO AS PRINCIPAIS MÉTRICAS DE CLASSIFICAÇÃO.


        # Arguments
            targets                     - Required : Classes alvo (List)
            predictions                 - Required : Classes preditas (List)

        # Returns
            classif_report              - Required : Relatório de classificação (String)

    """

    # INICIANDO A VARIÁVEL QUE ARMAZENARÁ O CLASIFICATION REPORT
    classif_report = None

    try:
        classif_report = classification_report(targets, predictions)

    except Exception as ex:
        print("ERRO NA FUNÇÃO {} - {}".format(stack()[0][3], ex))

    return classif_report




