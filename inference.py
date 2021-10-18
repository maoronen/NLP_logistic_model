from flask import Flask, request, render_template
import spacy
nlp = spacy.load('en_core_web_sm')
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import pickle
import os

import re
import string


app = Flask(__name__, template_folder='template')

infile = open('logistic_model.pkl', 'rb')
loaded_model = pickle.load(infile)
infile.close()

infile1 = open('tf.pkl', 'rb')
loaded_tf = pickle.load(infile1)
infile1.close()


@app.route('/')
def my_index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict_tweet():
    """
    This function takes arguments from the URL bar, creates an array to predict on,
    and lastly uses the pickled model to make a prediction which is returned to the client
    :return: string of prediction ('0' or '1')
    """

    def clean_text(text):
        '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
        and remove words containing numbers.'''
        stop_words = stopwords.words('english') + ['u', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = ''.join([c for c in text if c not in string.punctuation])
        tokens = re.split('\W+', text)
        text = ' '.join([word for word in tokens if word not in stop_words])
        text = nlp(text)
        text = ' '.join([word.lemma_ for word in text])
        return text

    input_text = request.form['input_text']
    vec_text = loaded_tf.transform([clean_text(input_text)])

    pred_proba = loaded_model.predict_proba(vec_text.toarray())
    pred = round(100 * pred_proba[0][1], 2)

    if pred >= 50:
        return render_template('index.html', prediction=f'Emergency; confidence ({pred}%)')
    else:
        return render_template('index.html', prediction=f'Non-emergency; confidence ({100-pred}%)')


if __name__ == '__main__':
    port = os.environ.get('PORT')
    if port:
        app.run(host='0.0.0.0', port=int(port))
    else:
        app.run()

