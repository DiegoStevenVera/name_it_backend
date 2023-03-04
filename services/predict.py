import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

from core.config import settings

from tensorflow.keras.applications import resnet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import RMSprop
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt

def _load_image_into_numpy_array(data):
    img = Image.open(BytesIO(data))
    img = img.resize((224, 224))
    return np.array(img)

def _load_dict_tokens():
    if not settings.TOKENS:
        path_tokens = f'{settings.FILES_PATH}/all_tokens.npy'
        with open(path_tokens, 'rb') as f:
            all_tokens = np.load(f, allow_pickle=True)
        print(f'Tokens uploaded: {all_tokens.shape}')
        settings.TOKENS = all_tokens

    return settings.TOKENS

def _load_models():
    if not settings.PET_NAME_MODEL:
        latent_dim = 64
        dropout_rate = 0.2
        len_tokens = settings.TOKENS.shape[0]

        settings.PET_NAME_MODEL = Sequential()
        settings.PET_NAME_MODEL.add(LSTM(latent_dim,
                    input_shape=(2, len_tokens),
                    recurrent_dropout=dropout_rate))
        settings.PET_NAME_MODEL.add(Dense(units=len_tokens, activation='softmax'))

        optimizer = RMSprop(learning_rate=0.01)
        settings.PET_NAME_MODEL.compile(loss='categorical_crossentropy', optimizer=optimizer)

        model_path = os.path.realpath(f'{settings.FILES_PATH}/pet_name_generator.h5')
        settings.PET_NAME_MODEL.load_weights(model_path)
        settings.RESNET_MODEL = resnet.ResNet101()

    return settings.PET_NAME_MODEL, settings.RESNET_MODEL

def _predict_name(type_pet_index, dict_tokens):
    x = np.zeros((1, 2, dict_tokens.shape[0]))
    x[0,0, np.where(dict_tokens == ' ')[0][0]] = 1
    x[0,1, type_pet_index] = 1
    len_tokens = dict_tokens.shape[0]

    adv_i = settings.PET_NAME_MODEL.predict(x, verbose=0)[0].argmax()

    x = np.zeros((1, 2, len_tokens))
    x[0,0, type_pet_index] = 1
    x[0,0, adv_i] = 1
    adj_i = settings.PET_NAME_MODEL.predict(x, verbose=0)[0].argmax()

    return dict_tokens[type_pet_index] \
            + ' ' + dict_tokens[adv_i] \
            + ' ' + dict_tokens[adj_i]

def predict_name_pet_from_img(data_b) -> str:
    _load_dict_tokens()
    _load_models()

    img_array = _load_image_into_numpy_array(data_b)
    img_prepro = resnet.preprocess_input(img_array)
    img_batch = np.expand_dims(img_prepro, axis=0)

    pred = settings.RESNET_MODEL.predict(img_batch)
    return _predict_name(pred.argmax(), settings.TOKENS)
