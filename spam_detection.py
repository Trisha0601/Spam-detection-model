import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer   #<-- Converts text data into numerical data
from sklearn.naive_bayes import MultinomialNB   #<-- A machine learning algorithm for classification
import streamlit as st  #<-- A library for creating web applications

data = pd.read_csv("spam.csv")
# print(data.head())  <-- It prints the first 5 rows of the dataset
# print(data.shape)   <-- It prints the number of rows and columns in the dataset

data.drop_duplicates(inplace = True)  # <-- It removes duplicate rows from the dataset
# print(data.shape)
# print(data.isnull().sum())  #<-- It prints the number of missing values in each column of the dataset

data["Category"] = data["Category"].replace(["spam", "ham"], ["Spam", "Not spam"])  #<-- It replaces the values in the "Category" column with more descriptive labels

mess = data["Message"]  #<-- It selects the "Message" column from the dataset
cate = data["Category"]  #<-- It selects the "Category" column from the dataset

(mess_train, mess_test, cate_train, cate_test) = train_test_split(mess, cate, test_size = 0.2)

cv = CountVectorizer(stop_words = "english")
train_data = cv.fit_transform(mess_train)

# Creating a model
model = MultinomialNB()
model.fit(train_data, cate_train)  #<-- It trains the model using the training data

# Testing the model
test_data = cv.transform(mess_test)
# print(model.score(test_data, cate_test))

# Predicting the data
def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result
st.header("Spam Detection")

# output = predict("Congratulations! You've won a lottery")
input_message = st.text_input("Enter your message here")

if st.button("Validate"):
    output = predict(input_message)
    st.text(output)