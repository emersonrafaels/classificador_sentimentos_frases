from dynaconf import settings

from model_emotion_classifier import Emotion_Classifier
from UTILS import generic_functions


def orchestra_model():

    # OBTENDO A BASE DE DADOS DE TREINAMENTO
    path_data_train_dir = settings.PATH_DATA_TRAIN

    # INICIALIZADO A CLASSE DO MODELO
    orchest_model = Emotion_Classifier()

    # REALIZANDO A LEITURA DA BASE DE DADOS DE TREINAMENTO
    validador, train_database = orchest_model.read_csv(path_data_train_dir)

    if validador:

        # REALIZANDO O PRÉ-PROCESSAMENTO DA BASE DE TREINAMENTO
        validador, result_pre_processing = orchest_model.orchestra_pre_processing_model(train_database)

        if validador:

            # REALIZANDO O TREINAMENTO DO MODELO
            validador, model, result_model = orchest_model.orchestra_create_classifier(result_pre_processing,
                                                                                       settings.EPOCHS,
                                                                                       settings.BATCH_SIZE)

            if validador:

                # SALVANDO O MODELO PARA USO FUTURO
                validador = orchest_model.save_model(model, settings.DIR_SAVE)

                if validador:
                    print("MODELO TREINADO COM SUCESSO - {}".format(generic_functions.obtem_date_time))


if __name__ == '__main__':
    orchestra_model()