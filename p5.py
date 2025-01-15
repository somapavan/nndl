!pip install datasets
from datasets import load_dataset
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
text = "\n".join(dataset["train"]["text"][:1000])  # First 1000 lines

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = 20
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

predictors, labels = input_sequences[:, :-1], input_sequences[:, -1]
labels = tf.keras.utils.to_categorical(labels, num_classes=total_words)

model = Sequential([
    Embedding(total_words, 50, input_length=max_sequence_len - 1),
    LSTM(100),
    Dense(total_words, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

history = model.fit(predictors, labels, epochs=10, verbose=1)

import numpy as np

def generate_next_word(model, tokenizer, input_text, max_sequence_len=10):
    input_sequence = tokenizer.texts_to_sequences([input_text])[0]
    if len(input_sequence) > max_sequence_len - 1:
        input_sequence = input_sequence[-(max_sequence_len - 1):
    input_sequence = np.pad(input_sequence, (max_sequence_len - 1 - len(input_sequence), 0), mode='constant')
    input_sequence = np.array(input_sequence).reshape(1, max_sequence_len - 1)
    prediction = model.predict(input_sequence)
    predicted_index = np.argmax(prediction)
    predicted_word = tokenizer.index_word[predicted_index]
    return predicted_word

input_text = "The quick brown fox"
predicted_word = generate_next_word(model, tokenizer, input_text)
print(f"Input: {input_text}")
print(f"Predicted next word: {predicted_word}")
