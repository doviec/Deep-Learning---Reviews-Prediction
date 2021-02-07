import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Read data from excel file.
data = pd.read_csv("tripadvisor_hotel_reviews.csv")

# List of words that are not useful for prediction. (such as 'i','me','my' etc.)
stop_words = stopwords.words('english')


# process_text function removes all numbers and stop_words from the reviews.
def process_text(text):
    text = re.sub(r'\d+', ' ', text)  # replace all digits with blank
    text = text.split()
    text = " ".join([word for word in text if word.lower().strip() not in stop_words])  # adding all words that are not in stop_words
    return text


# Apply the 'process_text' function on our reviews.
reviews = data['Review'].apply(process_text)

# Number of words we want to use in our dictionary.
num_words = 10000

# Tokenize the most common 'num_words' words. (in our case 10K words)
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(reviews)

# Convert each unique word in our reviews to a unique integer.
sequences = tokenizer.texts_to_sequences(reviews)

# The length of the longest sequence.
max_seq_length = np.max(list(map(lambda x: len(x), sequences)))

# Padding all sequences with 0's to the length of the longest sequence.
inputs = pad_sequences(sequences, maxlen=max_seq_length, padding='post')


# Encoding Labels
# We classified each Review's Rating to 0, 1 or 2,
# if the rating is 4 or 5 it will encode into 2,
# if the rating is 3 it will encode into 1,
# it will encode to 0 otherwise.
def rating_to_class(x):
    if x == 5 or x == 4:
        return 2
    elif x == 3:
        return 1
    else:
        return 0


# Apply 'rating_to_class' function on our Ratings.
labels = np.array(data['Rating'].apply(rating_to_class))

# Splitting the data to train and test. (70% train, 30% test)
train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, labels, train_size=0.7, random_state=100)

# Convert each label to a representative vector:
# label 0 converted to [1,0,0], (Rating 4 or 5)
# label 1 converted to [0,1,0], (Rating 3)
# label 2 converted to [0,0,1]. (Rating 1 or 2)
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=3)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=3)

# Modeling
inputs = tf.keras.Input(shape=(max_seq_length,))

embedding_dim = 128

# Add embedding layer
embedding = tf.keras.layers.Embedding(
    input_dim=num_words,
    output_dim=embedding_dim,
    input_length=max_seq_length
)(inputs)

# Add GRU layer
gru = tf.keras.layers.Bidirectional(
    tf.keras.layers.GRU(embedding_dim, return_sequences=True)
)(embedding)

# Flatten the data coming out the GRU layer
flatten = tf.keras.layers.Flatten()(gru)

# Output layer using softmax activation function
outputs = tf.keras.layers.Dense(3, activation='softmax')(flatten)

# Initiate the model
model = tf.keras.Model(inputs, outputs)

# Compile the mode with Adam optimizer,
# categorical_crossentropy as the loss function,
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=[
        'accuracy'
    ]
)

# Train the model on our train inputs and labels,
# Split 20% of the train set to validation set,
# Define early stopping callback.
history = model.fit(
    train_inputs,
    train_labels,
    validation_split=0.2,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=2,
            restore_best_weights=True
        )
    ]
)


print("Test Evaluation")

# Evaluate the model
model.evaluate(test_inputs, test_labels)
