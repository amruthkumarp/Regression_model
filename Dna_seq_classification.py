import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

###defining kmer function

def Kmers_funct(seq, size=6):
    return [seq[x:x+size].lower() for x in range(len(seq) - size + 1)]

###loading data into dataframe

chim_dna=pd.read_table('data.txt')
chim_dna.head()
chim_dna['class'].value_counts().sort_index().plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

###kmmer encoder
chim_dna['kmmer'] = chim_dna.apply(lambda x: Kmers_funct(x['sequence']), axis=1)
chim_dna = chim_dna.drop('sequence', axis=1)

chim_txt=list(chim_dna['kmmer'])
for i in range(0,len(chim_txt)):
    chim_txt[i]=' '.join(chim_txt[i])
chim_txt[0]

y_chim=chim_dna.iloc[:,0].values
print(y_chim)

vectorizer = CountVectorizer(min_df=1, ngram_range=(4,4))
X_chim = vectorizer.fit_transform(chim_txt)
print(X_chim)

###spilting the data
X_train, X_test, y_train, y_test = train_test_split(X_chim, y_chim, test_size = 0.20, random_state = 42)

###traing a model

classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Confusion matrix for predictions on human test DNA sequence\n")
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
print("\n")
print("Accuracy score: ", accuracy_score(y_test, y_pred))
print("F1 score: ", f1_score(y_test, y_pred, average='weighted'))
print("Precision score: ", precision_score(y_test, y_pred, average='weighted'))
print("Recall score: ", recall_score(y_test, y_pred, average='weighted'))

###testing a model
y_pred_chimp = classifier.predict(X_chim)
print("Confusion matrix for predictions on human test DNA sequence\n")
print(pd.crosstab(y_chim, y_pred_chimp, rownames=['True'], colnames=['Predicted'], margins=True))
print("\n")
print("Accuracy score: ", accuracy_score(y_chim, y_pred_chimp))
print("F1 score: ", f1_score(y_chim, y_pred_chimp, average='weighted'))
print("Precision score: ", precision_score(y_chim, y_pred_chimp, average='weighted'))
print("Recall score: ", recall_score(y_chim, y_pred_chimp, average='weighted'))

