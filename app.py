from flask import Flask, render_template, request, redirect, session
from gensim.models.fasttext import FastText
import pandas as pd
import pickle
import os
from bs4 import BeautifulSoup
import numpy as np
from nltk.stem import WordNetLemmatizer
import nltk


app = Flask(__name__)


def docvecs(embeddings, docs):
    vecs = np.zeros((len(docs), embeddings.vector_size))
    for i, doc in enumerate(docs):
        valid_keys = [term for term in doc if term in embeddings.key_to_index]
        docvec = np.vstack([embeddings[term] for term in valid_keys])
        docvec = np.sum(docvec, axis=0)
        vecs[i,:] = docvec
    return vecs




# Initialize the lemmatizer
# nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()


# app.secret_key = os.urandom(16) 

def lemmatize_text(text):
    if not isinstance(text, str):
        return ''
    words = text.lower().split()
    lemmas = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(lemmas)



@app.route('/', methods=['GET'])
def index():
    i_df = pd.read_csv('assignment3_2.csv')
    query = request.args.get('q', '').strip().lower()

    lem_query = ' '.join([lemmatizer.lemmatize(w) for w in query.split()])

    if query:
        filtered_i = i_df[
            i_df['Clothes Title'].apply(lambda x: lem_query in lemmatize_text(x)) |
            i_df['Class Name'].apply(lambda x: lem_query in lemmatize_text(x)) |
            i_df['Clothes Description'].apply(lambda x: lem_query in lemmatize_text(x))
        ]
    else:
        filtered_i = i_df

    item_list = filtered_i.to_dict(orient='records')

    return render_template('home.html', items=item_list, query=query)




@app.route('/write-review')
def write_review():
    i_df = pd.read_csv('assignment3_2.csv')
    query = request.args.get('q', '').strip().lower()

    lem_query = ' '.join([lemmatizer.lemmatize(w) for w in query.split()])

    if query:
        filtered_i = i_df[
            i_df['Clothes Title'].apply(lambda x: lem_query in lemmatize_text(x)) |
            i_df['Class Name'].apply(lambda x: lem_query in lemmatize_text(x)) |
            i_df['Clothes Description'].apply(lambda x: lem_query in lemmatize_text(x))
        ]
    else:
        filtered_i = i_df

    item_list = filtered_i.to_dict(orient='records')

    return render_template('write-review.html', items=item_list, query=query)





@app.route('/reviews/<reviewname>')
def show_review(reviewname):
    return render_template(f"reviews/{reviewname}.html")


@app.route('/review', methods=['GET', 'POST'])
def review():
    title = request.args.get('title', '')

    if request.method == 'POST':
        review_title = request.form['review_title']
        description = request.form['description']
        button_pressed = request.form.get('button')

        # Tokenize and predict
        tk_rv = description.lower().split()
        reviewFT = FastText.load("fasttext_model.bin")
        review_wv = reviewFT.wv
        review_vec = docvecs(review_wv, [tk_rv])
        with open("logistic_regression_model.pkl", "rb") as f:
            model = pickle.load(f)
        label = model.predict(review_vec)[0] 




        if button_pressed == 'Save':
            # loading HTML template
            soup = BeautifulSoup(open('templates/review_template.html'), 'html.parser')

            # adding title
            div_title = soup.find('div', {'class': 'review-title'})
            h1 = soup.new_tag('h1')
            h1.string = review_title
            div_title.append(h1)

 
            div_desc = soup.find('div', {'class': 'review-description'})
            p_desc = soup.new_tag('p')
            p_desc.string = description
            div_desc.append(p_desc)


            div_label = soup.find('div', {'class': 'review-recommendation'})
            p_label = soup.new_tag('p')
            p_label.string = "Recommended" if label == 1 else "Not Recommended"
            div_label.append(p_label)

            # Save the review as an HTML file
            filename_base = '_'.join(review_title.lower().split())
            filepath = f"templates/reviews/{filename_base}.html"
            with open(filepath, "w", encoding='utf-8') as file:
                file.write(str(soup))

            return redirect(f"/")

        return render_template('review_result.html', title=title, review_title=review_title,
                               description=description, label=label)

    return render_template('review_form.html', title=title)



@app.route('/all-reviews')
def all_reviews():
    # listing all .html files in the 'templates/reviews' folder
    rv_dir = 'templates/reviews'
    review_files = [f for f in os.listdir(rv_dir) if f.endswith('.html')]

    # extractingg review names without ".html"
    review_names = [f.replace('.html', '') for f in review_files]

    return render_template('all_reviews.html', reviews=review_names)




@app.route('/items')
def items():
    # load clothing items 
    i_df = pd.read_csv('assignment3_2.csv')
    # Convert the DataFrame to a list of dictionaries
    item_list = i_df.to_dict(orient='records')
    return render_template('items.html', items=item_list)
