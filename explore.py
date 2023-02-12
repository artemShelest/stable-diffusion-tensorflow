import gc

import tensorflow as tf
import numpy as np

from stable_diffusion_tf.stable_diffusion import get_models, load_diffusion_model


def representative_dataset_gen():
    for _ in range(10):
        yield [np.random.uniform(low=0.0, high=1.0, size=(1, 28, 28)).astype(np.float32)]


def convert_model(model, fname):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    del model
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = representative_dataset_gen
    tflite_quant_model = converter.convert()
    del converter
    open(fname, "wb").write(tflite_quant_model)


def main():
    # text_encoder, decoder = get_models(512, 512)
    # print(diffusion_model.summary())
    # convert_model(text_encoder, "text_encoder.tflite")
    diffusion_model = load_diffusion_model(512, 512)
    # convert_model(diffusion_model, "diffusion_model.tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(diffusion_model)
    print("Created converter")
    del diffusion_model
    gc.collect()
    tf.keras.backend.clear_session()
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = representative_dataset_gen
    gc.collect()
    tf.keras.backend.clear_session()
    print("Started conversion")
    tflite_quant_model = converter.convert()
    print("Finished conversion")
    del converter
    gc.collect()
    tf.keras.backend.clear_session()
    print("Started file write")
    open("diffusion_model.tflite", "wb").write(tflite_quant_model)


if __name__ == "__main__":
    main()
