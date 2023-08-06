# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 08:28:08 2023

@author: Cephas
@project: Synaptiq
@app: front end user interface (FUI)
@adapted from https://github.com/krishnaik06/Dockers/blob/master/app1.py
"""
# libraries & packages
import numpy as np
import streamlit as st
import pickle

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

max_tokens = 80

# render custom objects
def casual_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    Preventing informations from future tokens to flow into current token.
    That means masking the upper half of the dot product in self attention.
    Ones in the lower triangle, counting from the lower right corner.
    """
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    return tf.tile(mask, mult)

np.transpose(casual_attention_mask(1, 10, 10, dtype=tf.int32)[0])

class TransformerBlock(layers.Layer):
    def __init__(self, num_heads, key_dim, embed_dim, ff_dim, 
                 dropout_rate=0.1, name=None, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.attn = layers.MultiHeadAttention(
            num_heads, key_dim, output_shape=embed_dim
            )
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn1 = layers.Dense(self.ff_dim, activation='relu')
        self.ffn2 = layers.Dense(self.embed_dim)
        

    def call(self, inputs):
        input_sahpe = tf.shape(inputs)
        batch_size = input_sahpe[0]
        seq_len = input_sahpe[1]
        casual_mask = casual_attention_mask(
            batch_size, seq_len, seq_len, tf.bool
        )
        attention_output, attention_scores = self.attn(
            inputs,
            inputs,
            attention_mask=casual_mask,
            return_attention_scores=True,
        )
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn1 = self.ffn1(out1)
        ffn2 = self.ffn2(ffn1)
        ffn_output = self.dropout2(ffn2)
        return  (self.layernorm2(out1 + ffn_output), attention_scores)
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "key_dim": self.key_dim,
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config
    
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, max_len, vocab_size, embed_dim, name=None, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__()
        self.max_len  = max_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_len": self.max_len,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config
    
# create a TextGenerator checpoint
class TextGenerator(callbacks.Callback):
    def __init__(self, index_to_word, model, top_k=10):
        self.index_to_word = index_to_word
        self.word_to_index = {
            word: index for index, word in enumerate(index_to_word)
        }
        self.model = model
    
    def sample_from(self, probs, temperature):
        probs = probs ** (1 / temperature)
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), p=probs), probs

    def generate(self, start_prompt, max_tokens, temperature):
        start_tokens = [
            self.word_to_index.get(x, 1) for x in start_prompt.split()
        ]
        sample_token = None
        info = []
        # set iterations limits to produce short texts
        iteration = 0
        iteration_limit = 1e4
        while len(start_tokens) < max_tokens and sample_token != 0 and iteration < iteration_limit:
            x = np.array([start_tokens])
            y, att = self.model.predict(x, verbose=0)
            sample_token, probs = self.sample_from(y[0][-1], temperature)
            info.append(
                {
                    "prompt": start_prompt,
                    "word_probs": probs,
                    "atts": att[0, :, -1, :],
                }
            )
            start_tokens.append(sample_token)
            start_prompt = start_prompt + " " + self.index_to_word[sample_token]
            iteration += 1
        start_prompt = start_prompt
        print(f"\ngenerated text:\n{start_prompt}\n")
        return start_prompt
    
custom_objects = {
    'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
    'TransformerBlock': TransformerBlock,
    'TextGenerator': TextGenerator,
}

# load model from disk
t_eps = models.load_model(r"C:\Users\hp\Desktop\Transformers\models\3epsilon_architecture.h5", custom_objects=custom_objects)
# load vocabulary
path = r"C:\Users\hp\Desktop\Transformers\learned_vocabulary.pkl"
vocabulary = pickle.load(open(path, 'rb'))

# text generator object
text_generator = TextGenerator(vocabulary, t_eps)

# ... (previous code remains unchanged)

def main():
    st.title("TEXT GENERATION: 3epsilon Alg.")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Transformer decoder-only for text generation</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    prompt = st.text_input("Start prompt", "Insert the beginning of a text here and let the Transformer continue!")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.8, step=0.01)
    result = None
    if st.button("Generate"):
        if prompt.strip() != "":
            result = text_generator.generate(prompt, max_tokens, temperature)
            # Display generated text
            if result:
                generated_text = result
                st.write("Generated Text:")
                st.write(generated_text)
        else:
            st.warning("Please provide a valid starting prompt.")

    if st.button("About"):
        st.text("Test our 3epsilon Transformer algorithm")
        st.text("Built with Streamlit")

if __name__ == '__main__':
    main()

