import random
import json
import pickle #for serialization
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
# stem => work works worked working 

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

#sgd = stochastic graident descent
lemmatizer = WordNetLemmatizer()

#read the json file as a text and then load we finally get json object which is a dictionary.
intents = json.loads(open('intentsMauBot.json').read())

#create 3 empty lists
words = [] #all words that we have
classes =[]
documents = [] #combinations
ignore_letters = ['?','!','.',','] #do not take into account


for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        #with tokenize we will have a collection of words
        
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        #check if this class is already in the classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words)) #eliminated the duplicates

# print(words)

classes =sorted(set(classes))
pickle.dump(words,open('wordsMauBot.pkl','wb')) #save them to a file
pickle.dump(classes,open('classesMauBot.pkl','wb'))


training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        #if word occurs => 1 not occurs=>0
        bag.append(1) if word in word_patterns else bag.append(0)
        
    output_row = list(output_empty) #in order to type casting i am copying it
    output_row[classes.index(document[1])] = 1
    training.append([bag,output_row])
    
    
random.shuffle(training)
training = np.array(training) # turn into numpy array of training itself

train_x = list(training[:, 0]) #these are the x and y values to train chatbot 
train_y = list(training[:, 1])

model = Sequential() #create simple sequential model
model.add(Dense(128,input_shape=(len(train_x[0]),), activation='relu')) #we're going to add layers and first layer is input layer.
#rectified linear unit = relu
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))
#softmax function sumps up or scales the results in the output layer

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #lr =learning rate
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist= model.fit(np.array(train_x), np.array(train_y), epochs=200 , batch_size=5, verbose=1)
model.save('MauBot_model.h5',hist)



print("model created")



