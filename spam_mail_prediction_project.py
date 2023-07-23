
"""

Importing the Dependencies
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#tfidvectorizer is used to convert the text data into numerical value
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""Data collection and Preprocessing"""

#loading the data from csv file to a pandas Dataframe
raw_mail_data=pd.read_csv("/content/mail_data.csv")
raw_mail_data

#replace the null values with a null string
mail_data=raw_mail_data.where((pd.notnull(raw_mail_data)),'')
#it use to replace the all null values into null string in data set

#printing the first five rows of the dataset
mail_data.head()

#checking the number of rows and columns in dataset
mail_data.shape

"""Label Encoding"""

#label spam mail as 0;
#ham mail as 1;
mail_data.loc[mail_data["Category"]=='spam','Category',]=0
mail_data.loc[mail_data["Category"]=='ham','Category',]=1

"""spam = 0
ham = 1

"""

#separating the data as text and label
#separating the data as dependent and independent
x=mail_data["Message"]
y=mail_data["Category"]

x

y

"""splitting the data into train_test_split

"""

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)
#(random_state 3) is used for if u split the data in training and testing in every time it split in same way not in different way

print(x.shape)
print(x_train.shape)
print(x_test.shape)

"""feature Extraction"""

#transform the text data to feature vectors that can be input to the LOgistic Regresion model

feature_extraction=TfidfVectorizer(min_df=1,stop_words="english",lowercase="True")

vectorizer=CountVectorizer()
x_train_features=vectorizer.fit_transform(x_train)

x_test_features=vectorizer.transform(x_test)


#convert y_train and y_test values as intergers

y_train=y_train.astype('int')
y_test=y_test.astype("int")

print(x_train)

print(x_train_features)

"""Training the model

Logistic Regression

"""

model= LogisticRegression()

#training the logistic regression model with training data
model.fit(x_train_features,y_train)

"""evaluating the trained model

"""

#prediction on training data

prediction_on_training_data=model.predict(x_train_features)
accuracy_on_training_data=accuracy_score(y_train,prediction_on_training_data)

print("Accuracy_on_training_data: ",accuracy_on_training_data)

#prediction on test data

prediction_on_test_data=model.predict(x_test_features)
accuracy_on_test_data=accuracy_score(y_test,prediction_on_test_data)

print("Accuracy_on_test_data: ",accuracy_on_test_data)

"""Building a predictive system

"""

input_mail=["I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times."]

#convert text to feature vectors
input_data_feature=vectorizer.transform(input_mail)

#making prediction

prediction=model.predict(input_data_feature)

if (prediction[0]==1):
  print("Ham mail")
else:
  print("spam mail")



