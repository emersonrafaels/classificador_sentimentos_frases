from dynaconf import settings

from model_emotion_classifier import Emotion_Classifier


def orchestra_model():

    # OBTENDO A BASE DE DADOS DE TREINAMENTO
    path_data_train_dir = settings.PATH_DATA_TRAIN

    # INICIALIZADO A CLASSE DO MODELO
    orchest_model = Emotion_Classifier()

    # REALIZANDO A LEITURA DA BASE DE DADOS DE TREINAMENTO
    validador, train_database = orchest_model.read_csv(path_data_train_dir)

    if validador:

        # REALIZANDO O PRÉ-PROCESSAMENTO DA BASE DE TREINAMENTO
        result_pre_processing = orchest_model.orchestra_pre_processing_model(train_database)

        # REALIZANDO O TREINAMENTO DO MODELO
        result_model = orchest_model.orchestra_create_classifier(result_pre_processing,
                                                                 settings.EPOCHS,
                                                                 settings.BATCH_SIZE)


if __name__ == '__main__':
    orchestra_model()