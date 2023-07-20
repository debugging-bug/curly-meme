import nltk 
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random
nltk.download('wordnet')


words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read() #load json file
intents = json.loads(data_file) 

#iterate through the patters and tokenize the sentence using nltk.word_tokenize() function and appened each word in the words list. we also create a list of classes 
for intent in intents['intents']: 
    for pattern in intent['patterns']: 
        #tokenize each word 
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
#sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print (len(documents), "documents")
#classes = intents
print (len(classes), "classes", classes)
#words = all words, vocabulary
print (len(words), "unique lemmatized words", words)

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))


#convert text into numbers using the vocabulary
#create training and testing data sets 
training = []

#create an empty array for our output
output_empty = [0] * len(classes)
#training set, bag of words for each sentence
for doc in documents:
    #initialize our bag of words
    bag = [] #bag of words
    #list of tokenized words for the pattern
    pattern_words = doc[0]#tokenize the pattern
    #lemmatize each word
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    #create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    #output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    print(output_row)
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)
#create train and test data
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")

#create the model - 3 layers first layer 128 neurons, second later 64 neurons and 3rd output layer contains number of neurons equal to number of intents to predict output 
# intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#compile the model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=1000, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")

