import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('punkt_tab')

#Setting Up the Lemmatizer

lemmatizer= WordNetLemmatizer()
lemmatizer= WordNetLemmatizer()
#Loading the Intents File
intents = json.loads(open('D:\Mybot_py\chatbot\Include\intents.json').read())

#Processing Words and Intents

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

#Breaking Down Patterns

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
#Cleaning and Sorting Words

words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))

classes = sorted(set(classes))
#Saving Processed Data
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

#Creating Training Data

training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)
    
#Shuffling and Splitting Data

random.shuffle(training)
training = np.array(training)

#Creating Training Data and Testing Data

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

#Building the Deep Learning Model

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

#Setting Up the Model

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#Training the Model

hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)

#Saving the Model

model.save('chatbot_model.h5', hist)
print('Done')


#Final Message

print("Model Created")
