import tensorflowjs as tfjs
import tensorflow as tf
import keras
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default='decoder.hd5', type=str, help='path to the keras model you want to convert to tfjs')
parser.add_argument('--output_path', default='tfjs_decoder', type=str, help='path to the directory you want the tfjs model to be added into')
args = parser.parse_args()

model = keras.models.load_model(args.input_path)
tfjs.converters.save_keras_model(model, args.output_path)
