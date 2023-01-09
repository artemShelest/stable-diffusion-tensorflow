import numpy as np
from tqdm import tqdm
import math

import tensorflow as tf
from tensorflow import keras
import platform
import tflite_runtime.interpreter as tflite

from .autoencoder_kl import Decoder
from .diffusion_model import UNetModel
from .clip_encoder import CLIPTextTransformer
from .clip_tokenizer import SimpleTokenizer
from .constants import _UNCONDITIONAL_TOKENS, _ALPHAS_CUMPROD

_EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]

MAX_TEXT_LEN = 77


class Text2Image:
    def __init__(self, img_height=1000, img_width=1000, jit_compile=False):
        # UNet requires multiples of 2**7 = 128 to prevent dimension mismatch
        self.img_height = round(img_height / 128) * 128
        self.img_width = round(img_width / 128) * 128
        self.tokenizer = SimpleTokenizer()

        text_encoder, decoder = get_models(self.img_height, self.img_width)
        diffusion_model = load_diffusion_model(self.img_height, self.img_width)
        self.text_encoder = text_encoder
        self.diffusion_model = diffusion_model
        self.decoder = decoder
        self.te_input_details = self.text_encoder.get_input_details()
        self.te_output_details = self.text_encoder.get_output_details()
        # if jit_compile:
        #     self.text_encoder.compile(jit_compile=True)
        #     self.diffusion_model.compile(jit_compile=True)
        #     self.decoder.compile(jit_compile=True)

    def _text_encoder_predict(self, data):
        self.text_encoder.set_tensor(self.te_input_details[0]['index'], data[0])
        self.text_encoder.set_tensor(self.te_input_details[1]['index'], data[1])
        self.text_encoder.invoke()
        return self.text_encoder.get_tensor(self.te_output_details[0]['index'])

    def generate(
            self,
            prompt,
            batch_size=1,
            num_steps=25,
            unconditional_guidance_scale=7.5,
            temperature=1,
            seed=None,
    ):
        # Tokenize prompt (i.e. starting context)
        inputs = self.tokenizer.encode(prompt)
        assert len(inputs) < MAX_TEXT_LEN, f"Prompt is too long (should be < {MAX_TEXT_LEN} tokens)"
        phrase = inputs + [49407] * (MAX_TEXT_LEN - len(inputs))
        phrase = np.array(phrase)[None].astype("int32")
        phrase = np.repeat(phrase, batch_size, axis=0)

        # Encode prompt tokens (and their positions) into a "context vector"
        pos_ids = np.array(list(range(MAX_TEXT_LEN)))[None].astype("int32")
        pos_ids = np.repeat(pos_ids, batch_size, axis=0)
        # context = self.text_encoder.predict_on_batch([phrase, pos_ids])
        context = self._text_encoder_predict([phrase, pos_ids])

        # Encode unconditional tokens (and their positions into an
        # "unconditional context vector"
        unconditional_tokens = np.array(_UNCONDITIONAL_TOKENS)[None].astype("int32")
        unconditional_tokens = np.repeat(unconditional_tokens, batch_size, axis=0)
        self.unconditional_tokens = tf.convert_to_tensor(unconditional_tokens)
        # unconditional_context = self.text_encoder.predict_on_batch(
        #     [self.unconditional_tokens, pos_ids]
        # )
        unconditional_context = self._text_encoder_predict(
            [self.unconditional_tokens, pos_ids]
        )
        timesteps = np.arange(1, 1000, 1000 // num_steps)
        latent, alphas, alphas_prev = self.get_starting_parameters(
            timesteps, batch_size, seed
        )

        # Diffusion stage
        progbar = tqdm(list(enumerate(timesteps))[::-1])
        for index, timestep in progbar:
            progbar.set_description(f"{index:3d} {timestep:3d}")
            e_t = self.get_model_output(
                latent,
                timestep,
                context,
                unconditional_context,
                unconditional_guidance_scale,
                batch_size,
            )
            a_t, a_prev = alphas[index], alphas_prev[index]
            latent, pred_x0 = self.get_x_prev_and_pred_x0(
                latent, e_t, index, a_t, a_prev, temperature, seed
            )

        # Decoding stage
        decoded = self.decoder.predict_on_batch(latent)
        decoded = ((decoded + 1) / 2) * 255
        return np.clip(decoded, 0, 255).astype("uint8")

    def timestep_embedding(self, timesteps, dim=320, max_period=10000):
        half = dim // 2
        freqs = np.exp(
            -math.log(max_period) * np.arange(0, half, dtype="float32") / half
        )
        args = np.array(timesteps) * freqs
        embedding = np.concatenate([np.cos(args), np.sin(args)])
        return tf.convert_to_tensor(embedding.reshape(1, -1))

    def get_model_output(
            self,
            latent,
            t,
            context,
            unconditional_context,
            unconditional_guidance_scale,
            batch_size,
    ):
        timesteps = np.array([t])
        t_emb = self.timestep_embedding(timesteps)
        t_emb = np.repeat(t_emb, batch_size, axis=0)
        unconditional_latent = self.diffusion_model.predict_on_batch(
            [latent, t_emb, unconditional_context]
        )
        latent = self.diffusion_model.predict_on_batch([latent, t_emb, context])
        return unconditional_latent + unconditional_guidance_scale * (
                latent - unconditional_latent
        )

    def get_x_prev_and_pred_x0(self, x, e_t, index, a_t, a_prev, temperature, seed):
        sigma_t = 0
        sqrt_one_minus_at = math.sqrt(1 - a_t)
        pred_x0 = (x - sqrt_one_minus_at * e_t) / math.sqrt(a_t)

        # Direction pointing to x_t
        dir_xt = math.sqrt(1.0 - a_prev - sigma_t ** 2) * e_t
        # noise = sigma_t * tf.random.normal(x.shape, seed=seed) * temperature
        x_prev = math.sqrt(a_prev) * pred_x0 + dir_xt
        return x_prev, pred_x0

    def get_starting_parameters(self, timesteps, batch_size, seed):
        n_h = self.img_height // 8
        n_w = self.img_width // 8
        alphas = [_ALPHAS_CUMPROD[t] for t in timesteps]
        alphas_prev = [1.0] + alphas[:-1]
        latent = tf.random.normal((batch_size, n_h, n_w, 4), seed=seed)
        return latent, alphas, alphas_prev


def get_models(img_height, img_width, download_weights=True):
    n_h = img_height // 8
    n_w = img_width // 8
    # Create text encoder
    # input_word_ids = keras.layers.Input(shape=(MAX_TEXT_LEN,), dtype="int32")
    # input_pos_ids = keras.layers.Input(shape=(MAX_TEXT_LEN,), dtype="int32")
    # embeds = CLIPTextTransformer()([input_word_ids, input_pos_ids])
    # text_encoder = keras.models.Model([input_word_ids, input_pos_ids], embeds)
    try:
        delegate = [tflite.load_delegate(_EDGETPU_SHARED_LIB)]
    except ValueError:
        delegate = None
    text_encoder = tflite.Interpreter(model_path="text_encoder.tflite", experimental_delegates=delegate)
    text_encoder.allocate_tensors()

    # Create decoder
    latent = keras.layers.Input((n_h, n_w, 4))
    decoder = Decoder()
    decoder = keras.models.Model(latent, decoder(latent))

    # text_encoder_weights_fpath = keras.utils.get_file(
    #     origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/text_encoder.h5",
    #     file_hash="d7805118aeb156fc1d39e38a9a082b05501e2af8c8fbdc1753c9cb85212d6619",
    # )

    decoder_weights_fpath = keras.utils.get_file(
        origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/decoder.h5",
        file_hash="6d3c5ba91d5cc2b134da881aaa157b2d2adc648e5625560e3ed199561d0e39d5",
    )

    # text_encoder.load_weights(text_encoder_weights_fpath)

    decoder.load_weights(decoder_weights_fpath)
    return text_encoder, decoder


def load_diffusion_model(img_height, img_width):
    n_h = img_height // 8
    n_w = img_width // 8
    # Creation diffusion UNet
    context = keras.layers.Input((MAX_TEXT_LEN, 768))
    t_emb = keras.layers.Input((320,))
    latent = keras.layers.Input((n_h, n_w, 4))
    unet = UNetModel()
    diffusion_model = keras.models.Model(
        [latent, t_emb, context], unet([latent, t_emb, context])
    )
    diffusion_model_weights_fpath = keras.utils.get_file(
        origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/diffusion_model.h5",
        file_hash="a5b2eea58365b18b40caee689a2e5d00f4c31dbcb4e1d58a9cf1071f55bbbd3a",
    )
    diffusion_model.load_weights(diffusion_model_weights_fpath)
    return diffusion_model
