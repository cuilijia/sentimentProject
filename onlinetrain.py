import pandas as pd
import jieba
from gensim.models import Word2Vec
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

sentenceLen = 300

Sg=0
Size=250
Window=5
Min_count=1
Workers=4
Iter=10
BatchSize=16

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
    stopword = [' ', '@', ';', '&', '\'', '.', '’', '"', '，']
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
    model.add(tf.keras.layers.LSTM(BatchSize, input_shape=(xtrainsent.shape[1], xtrainsent.shape[2])))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(3, activation='softmax')) # Dense=>全连接层,输出维度=3
    # model.add(tf.keras.layers.Activation('softmax'))
    model.add(tf.keras.layers.Dense(ytrainsent.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(xtrainsent, ytrainsent, epochs=10, batch_size=BatchSize, verbose=1)
    # model.save('lstm.model')
    # print(model.summary())
    return model

def splitdata(Data):
    # trainData = Data.head(13000)
    # testData = Data[13000:]
    X = Data["text"]
    Y = Data["label"]

    # x_train = trainData["text"]
    # y_train = trainData["airline_sentiment"]
    # x_test = testData["text"]
    # y_test = testData["airline_sentiment"]

    x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size = 0.33, random_state = 42)

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

print("数据读入:")
readData = pd.read_csv('sentimentProject/sentiment140/train16.csv', sep=',')
# Data=Data.head(7000)+Data.tail(7000)
Data=pd.concat([readData.head(30000),readData.tail(30000)])
print(Data.__len__())

sentenceLen = getsentenceLen(Data["text"])
# print(sentenceLen)
print("去除停用词:")
Data["text"] = Data["text"].apply(lambda x: rmStopwords(x))

# print(Data["label"])
print("Word2vec:")
wvmodel = trainW2V(Data["text"], Sg, Size, Window, Min_count, Workers, Iter)

Data["text"] = Data["text"].head(20000).apply(lambda x: W2V(x,wvmodel))
#
Data["label"]=Data["label"].replace(0, 0)
Data["label"]=Data["label"].replace(4, 1)

# print(Data["label"])
print(Data["label"].__len__())
x_train,Y_train,x_test,Y_test = splitdata(Data)

print("LSTM:")
model=train(x_train,Y_train)

score = model.evaluate(x_test, Y_test, batch_size=BatchSize)
print(score)

