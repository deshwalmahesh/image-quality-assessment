
from src.handlers.model_builder import Nima
import numpy as np
from PIL import Image
from src.utils.utils import calc_mean_score


def load_image(path, basenet_preprocess, img_load_dims=(224, 224)):
    '''
    '''
    # img = np.array(utils.load_image(path, img_load_dims)) # expand dims
    img = np.array(np.array(Image.open(path).resize(img_load_dims, Image.NEAREST)))
    img = img[np.newaxis, ...] 
    return basenet_preprocess(img)


def predict(model, data_generator):
    return model.predict(data_generator, workers=2, use_multiprocessing=True, verbose = 0)


def build_nima(weights_file,base_model_name = "MobileNet"):
    '''
    '''
    nima = Nima(base_model_name, weights=None)
    nima.build()
    nima.nima_model.load_weights(weights_file)
    return nima


def get_score(image_path, nima):
    '''
    nima: Nima class from the build_nima class
    '''
    image = load_image(image_path, nima.preprocessing_function())
    preds = predict(nima.nima_model, image)
    return calc_mean_score(preds.squeeze())
