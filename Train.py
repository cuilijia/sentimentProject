import pandas as pd
import jieba
from gensim.models import Word2Vec
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

def onehot(num):
    list=[0]*3
    list[num]=1
    return list

def getsentenceLen(data):
    max=0
    for d in data:
        if len(d)>max:
            max=len(d)
    return max

def rmStopwords(sent):
    stopword = [' ','@',';','&','\'','.','’','"','，','1','2','3','4','5','6','7','8','9','0']
    sent = sent.strip()
    seg_list = jieba.cut(sent, cut_all=False)
    newsent = []
    for s in seg_list:
        if s not in stopword:
            newsent.append(s)
    if(newsent.__len__()<sentenceLen):
        annewsent=[]
        for i in range(sentenceLen-newsent.__len__()):
            annewsent.append(" ")
        annewsent = annewsent + newsent
        newsent = annewsent
    return newsent

def W2V(sentence,wvmodel):
    newsent = []
    for s in sentence:
        newsent.append(wvmodel.wv[s])
    return newsent


def trainW2V(sentences, Sg, Size, Window, Min_count, Workers, Iter):
    model = Word2Vec(sentences, sg=Sg, size=Size, window=Window, min_count=Min_count, workers=Workers, iter=Iter)
    return model

def train(xtrainsent,ytrainsent):
    model = tf.keras.models.Sequential()
    BiLSTMlayer=tf.keras.layers.Bidirectional( tf.keras.layers.GRU(BatchSize, input_shape=(xtrainsent.shape[1], xtrainsent.shape[2])))
    model.add(BiLSTMlayer)
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(ytrainsent.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(xtrainsent, ytrainsent, epochs=15, batch_size=BatchSize, verbose=1)
    return model

def splitdata(X,Y):
    x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size=0.2, random_state = 42)
    Y_train = []
    Y_test = []
    for y in y_train:
        Y_train.append(onehot(y))
    for y in y_test:
        Y_test.append(onehot(y))

    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    x_train = np.array(list(x_train))
    x_test = np.array(list(x_test))

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2]))
    return x_train,Y_train,x_test,Y_test


sentenceLen = 70
Sg=0
Size=100
Window=5
Min_count=1
Workers=4
Iter=10
BatchSize=16

Data = pd.read_csv('data/twitter-airline-sentiment/Tweets.csv', sep=',')
print(Data.__len__())
# print(Data.head(10)["sentiment"])
sentenceLen = getsentenceLen(Data["text"])
print(sentenceLen)
Data["text"] = Data["text"].apply(lambda x: rmStopwords(x))

wvmodel = trainW2V(Data["text"], Sg, Size, Window, Min_count, Workers, Iter)

Data["text"] = Data["text"].apply(lambda x: W2V(x,wvmodel))

Data["airline_sentiment"]=Data["airline_sentiment"].replace("negative", 0)
Data["airline_sentiment"]=Data["airline_sentiment"].replace("neutral", 1)
Data["airline_sentiment"]=Data["airline_sentiment"].replace("positive", 2)

X = Data["text"]
Y = Data["airline_sentiment"]
x_train,Y_train,x_test,Y_test = splitdata(X,Y)

model=train(x_train,Y_train)
score = model.evaluate(x_test, Y_test, batch_size=BatchSize)
print(score)


# 0.7855  0.7903 0.7913 7964481