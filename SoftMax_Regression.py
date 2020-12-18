import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
stop_words = stopwords.words('english')

# functions:


def process_data(text):      # data clean from irrelevant words and numbers, spaces etc
    text = re.sub(r'\d+', ' ', text)
    text = nltk.word_tokenize(text)
    text = " ".join([word for word in text if word.lower().strip() not in stop_words])
    return text


def prepare_vocabulary(data):
    idx = 0
    for sentence in data:
        for word in nltk.word_tokenize(sentence):
            if word not in word2location:
                word2location[word] = idx
                idx += 1
    return idx


def convert2vec(sentence):
    res_vec = np.zeros(vocabulary_size)
    for word in nltk.word_tokenize(sentence): # nltk.word_tokenize(sentence): #also here...
        if word in word2location:
            res_vec[word2location[word]] += 1
    return res_vec


# preperation of data set
reviews_data_set = pd.read_csv("tripadvisor_hotel_reviews.csv")   # Read the csv file"
# reviews_data_set.info() # check no null
reviews_without_stopwords = reviews_data_set['Review'].apply(process_data)
rating_data_set = reviews_data_set['Rating']

# sorting data to Train & Test
number_train = 650
number_test = 5000
train_reviews = reviews_without_stopwords[0:number_train]
train_ratings = rating_data_set[0:number_train]

test_reviews = reviews_without_stopwords[number_train: number_train+number_test]
test_ratings = rating_data_set[number_train: number_train+number_test]

word2location = {}
vocabulary_size = 0
vocabulary_size = prepare_vocabulary(train_reviews)  # bag of words

features = vocabulary_size
categories = 3  # labels
x = tf.placeholder(tf.float32, [None, features])
y_ = tf.placeholder(tf.float32, [None, categories])
W = tf.Variable(tf.zeros([features, categories]))
b = tf.Variable(tf.zeros([categories]))


matmul_result = tf.matmul(x, W)

y = tf.nn.softmax(matmul_result + b)

loss = -tf.reduce_mean(y_*tf.log(y))
alpha = 0.005
update = tf.train.GradientDescentOptimizer(alpha).minimize(loss)

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# update = tf.train.GradientDescentOptimizer(alpha).minimize(cross_entropy)

# convert to bag of words the reviews
tmp_train_x = []
tmp_test_x = []

for review in train_reviews:
    tmp_train_x.append(convert2vec(review))

for review in test_reviews:
    tmp_test_x.append(convert2vec(review))

data_train_x = np.array(tmp_train_x)
data_test_x = np.array(tmp_test_x)

tmp_train_y = []
tmp_test_y = []

# convert to labels the ratings
for rating in train_ratings:
    rating_vec = [0, 0, 0]
    if rating > 3:
        rating_vec = [0, 0, 1]
    elif rating == 3:
        rating_vec = [0, 1, 0]
    else:
        rating_vec = [1, 0, 0]
    tmp_train_y.append(rating_vec)

for rating in test_ratings:
    rating_vec = [0, 0, 0]
    if rating > 3:
        rating_vec = [0, 0, 1]
    elif rating == 3:
        rating_vec = [0, 1, 0]
    else:
        rating_vec = [1, 0, 0]
    tmp_test_y.append(rating_vec)

data_train_y = np.array(tmp_train_y)
data_test_y = np.array(tmp_test_y)

# start session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
for i in range(0, 6000):
    sess.run(update, feed_dict={x: data_train_x, y_: data_train_y})
    if i % 1000 == 0:
        print("Loss: {}, Train Accuracy: {} ".format(loss.eval(session=sess, feed_dict={x: data_train_x, y_: data_train_y}),
                                               sess.run(accuracy, feed_dict={x: data_train_x, y_: data_train_y})))

# data to be printed
total_counter = 0
true_counter = 0
low_prediction_counter = 0
mid_prediction_counter = 0
high_prediction_counter = 0
low_actual_counter = 0
mid_actual_counter = 0
high_actual_counter = 0
true_low_prediction = 0
true_mid_prediction = 0
true_high_prediction = 0

# print(sess.run(accuracy, feed_dict={x: data_test_x, y_: data_test_y}))
print("*** Test Model Details - Neural Network ***")
for test_review, test_rating in zip(test_reviews, test_ratings):  # Accuracy Trues / All
    res = sess.run(y, feed_dict={x: [convert2vec(test_review)]})
    rating = "low"

    if res.item(0) > res.item(1) and res.item(0) > res.item(2):
        rating = "low"
        low_prediction_counter += 1
    elif res.item(1) > res.item(0) and res.item(1) > res.item(2):
        rating = "mid"
        mid_prediction_counter += 1
    else:
        rating = "high"
        high_prediction_counter += 1

    if 1 <= test_rating <= 2:
        low_actual_counter += 1
        if rating == "low":
            true_counter += 1
            true_low_prediction += 1

    if test_rating == 3:
        mid_actual_counter += 1
        if rating == "mid":
            true_mid_prediction += 1
            true_counter += 1

    if test_rating > 3:
        high_actual_counter += 1
        if rating == "high":
            true_high_prediction += 1
            true_counter += 1
    # print('review: {} | rating: {} | prediction {} | res {}'.format(test_review[:50], test_rating, rating, res))

    total_counter += 1

print("Total Train Reviews :{}".format(number_train))
print("Total Test Reviews: {}".format(total_counter))
print("Total True Predictions: {}".format(true_counter))
print("Low = 1*-2* | mid = 3* | High = 4*-5*")
print("Low - prediction: {}, Correct Prediction: {}, Actual: {}".format(low_prediction_counter, true_low_prediction, low_actual_counter))
print("Mid - prediction: {}, Correct Prediction: {}, Actual: {}".format(mid_prediction_counter, true_mid_prediction, mid_actual_counter))
print("High - prediction: {}, Correct Prediction: {}, Actual: {}".format(high_prediction_counter, true_high_prediction, high_actual_counter))

percentage = (true_counter / total_counter) * 100
print("alpha: {}".format(alpha))
print("Test Accuracy: {} % ".format(percentage))
print("Test Error: {} % ".format(100 - percentage))




