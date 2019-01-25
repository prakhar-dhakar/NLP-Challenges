import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
data = pd.read_csv('training.txt', sep="\t", header=None,names=["Result","query"])

test_x =pd.read_csv("sampleInput.txt",sep="\t", header=None,names=["Input"])
test_y=pd.read_csv("sampleOutput.txt",sep="\t", header=None,names=["Output"])

v = TfidfVectorizer()
x = v.fit_transform(data['Result'])

train_x = x
train_y = data['query']
svm = LinearSVC()
svm.fit(train_x,train_y)

test_x = test_x[1:]

texts = v.transform(test_x['Input'])
predicted = svm.predict(texts)
result=list(predicted==test_y['Output'])
print('%.3f'%(result.count(True)/len(result)))