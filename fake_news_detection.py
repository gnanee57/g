from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

path = "E:\Projects\/fake-news-detection-project"
true_df = pd.read_csv(path + '\True.csv')
fake_df = pd.read_csv(path + '\Fake.csv')
loaded_model = pickle.load(open(path + '\model1.pkl', 'rb'))

true_df['label'] = 0
fake_df['label'] = 1
true_df.head()
fake_df.head()
true_df = true_df[['text', 'label']]
fake_df = fake_df[['text', 'label']]
dataset = pd.concat([true_df, fake_df])

from sklearn.feature_extraction.text import TfidfVectorizer


vectorizer = TfidfVectorizer(max_features=10000, lowercase=False, ngram_range=(1, 2))

X = dataset.iloc[:35000, 0]
y = dataset.iloc[:35000, 1]

from sklearn.model_selection import train_test_split
train_X , test_X , train_y , test_y = train_test_split(X , y , test_size = 0.2 ,random_state = 0)

def fake_news_det(news):
    vec_train = vectorizer.fit_transform(train_X)
    vec_test = vectorizer.transform(test_X)
    input_data = [news]
    vectorized_input_data = vectorizer.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        print(pred)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")


if __name__ == '__main__':
    app.run(debug=True)
