from dynaconf import settings

from model_emotion_classifier import Emotion_Classifier


def orchestra_model():

    # OBTENDO A BASE DE DADOS
    path_data_train_dir = settings.PATH_DATA_TRAIN

    train_database = Emotion_Classifier.read_csv(path_data_train_dir)

    result_pre_processing = Emotion_Classifier.orchestra_pre_processing_model(train_database)

    result_model = Emotion_Classifier.orchestra_create_classifier(result_pre_processing, 1000, 30)


if __name__ == '__main__':
    orchestra_model()