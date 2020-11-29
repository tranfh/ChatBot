import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = [] #list of all patterns
docs_y = [] #list of words for patterns

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        #Get all the words in pattern
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"]) #tracks what intent doc_x element is apart of

        if intent['tag'] not in labels:
            labels.append(intent["tag"])

# How many words it has seen already
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

# Create a bag of words 
# Represents all of given words in model
# Create one hot encoded
# [1, 0, 1] # if word exists put a 1 if not 0
# "a", "byte" "hi"
training = [] #List of training 
output = []

out_empty = [0 for _ in range(len(labels))]

# Turn tags onehot encoded for tags
for x, doc in enumerate(docs_x):
    bag = [] #one hot encoded bag of words
    
    wrds = [stemmer.stem(w.lower()) for w in doc]

    # Go through different words and put 1 or 0 if in our list
    for w in words:
        if w in wrds:
            bag.append(1) #word exists
        else:
            bag.append(0) #word doesnt exist

# Make a copy 
output_row = out_empty[:]

#look through label list see where tag is in list and set that value to 1
output_row[labels.index(docs_y[x])] = 1 
training.append(bag) #lists of 0s and 1s 
output.append(output_row) #lists of 0s and 1s 

# create arrays to feed into model
training = np.array(training) #np required for tflearn
output = np.array(output)

with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)

tf.compat.v1.reset_default_graph

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net,8) 
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8) #2 hidden layers 
# all of the neurons above connect to softmax layer below 
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") #neural network has 6 layers 
net = tflearn.regression(net)

# type of neural network that takes the above
model = tflearn.DNN(net)

# number of times it sees the data
model.fit(training, output, n_epoch=10000, batch_size=8, show_metric=True)
model.save("model.tflearn")

def bagOfWords(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    
    return np.array(bag)

def chat():
    print("Start talking with the bot!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        
        results = model.predict([bagOfWords(inp, words)])[0]
        results_index = np.argmax(results) # gives us index of greatest number to display
        tag = labels[results_index]

        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
    
            print(random.choice(responses))
        else:
            print("I didn't understand your question. Please ask any question.")

chat()