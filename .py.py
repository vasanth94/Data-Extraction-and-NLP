 
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat.textstat import textstat
import nltk
nltk.download('vader_lexicon')
from textblob import TextBlob
import matplotlib.pyplot as plt






# Load the input Excel file
df = pd.read_excel('input.xlsx')

# Create a directory to store the extracted articles
if not os.path.exists('articles'):
    os.makedirs('articles')

# Loop through each article URL
for index, row in df.iterrows():
    url = row['URL']
    url_id = row['URL_ID']

    # Send a request to the URL and get the HTML response
    response = requests.get(url)
    html = response.content

    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')

    # Extract the article title and text
    title = soup.find('title').text
    article_text = ''
    for p in soup.find_all('p'):
        article_text += p.text + '\n\n'

    # Remove unwanted characters and trim the article text
    article_text = article_text.replace('\n\n\n', '\n\n').strip()

    # Create a text file with the URL ID as its file name
    with open(f'articles/{url_id}.txt', 'w', encoding='utf-8') as f:
        f.write(title + '\n\n' + article_text)

    print(f'Extracted article {url_id} and saved to articles/{url_id}.txt')


    # Load the output data structure Excel file
output_structure = pd.read_excel('Output Data Structure.xlsx')

# Load the extracted article texts from the previous step
article_texts = []
for file in os.listdir('articles'):
    with open(os.path.join('articles', file), 'r', encoding='utf-8') as f:
        article_text = f.read()
        article_texts.append(article_text)

# Create a list to store the output data
output_data = []

# Loop through each article text
for article_text in article_texts:
    # Tokenize the text
    tokens = word_tokenize(article_text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Compute variables
    text_length = len(tokens)
    sentiment_score = SentimentIntensityAnalyzer().polarity_scores(article_text)['compound']
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vector = tfidf_vectorizer.fit_transform([article_text])
    tfidf_score = tfidf_vector.toarray()[0].sum()
    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = sentiment_analyzer.polarity_scores(article_text)
    pos_score = sentiment_scores['pos']
    neg_score = sentiment_scores['neg']
    polarity_score = sentiment_scores['compound']
    subjectivity_score = sentiment_scores.get('subjectivity')
    sentences = sent_tokenize(article_text)
    avg_sentence_length = np.mean([len(sentence.split()) for sentence in sentences])

    complex_words = [word for word in tokens if len(word) > 6]
    percentage_complex_words = len(complex_words) / len(tokens)

    fog_index = textstat.flesch_reading_ease(article_text)

    avg_words_per_sentence = np.mean([len(sentence.split()) for sentence in sentences])

    complex_word_count = len(complex_words)
    word_count = len(tokens)

    syllables_per_word = np.mean([textstat.syllable_count(word) for word in tokens])

    personal_pronouns = ['I', 'you', 'he', 'she', 'it', 'we', 'they']
    personal_pronoun_count = sum([1 for word in tokens if word.lower() in personal_pronouns])
    avg_word_length = np.mean([len(word) for word in tokens])

    blob = TextBlob(article_text)
    textblob_subjectivity_score = blob.sentiment.subjectivity

   # Create a dictionary to store the output data
    output_dict = {
        'Text Length': text_length,
        'Sentiment Score': sentiment_score,
        'TF-IDF Score': tfidf_score,
        'POSITIVE SCORE': pos_score,
        'NEGATIVE SCORE': neg_score,
        'POLARITY SCORE': polarity_score,
        'SUBJECTIVITY SCORE (TextBlob)': textblob_subjectivity_score,
        'AVG SENTENCE LENGTH': avg_sentence_length,
        'PERCENTAGE OF COMPLEX WORDS': percentage_complex_words,
        'FOG INDEX': fog_index,
        'AVG NUMBER OF WORDS PER SENTENCE': avg_words_per_sentence,
        'COMPLEX WORD COUNT': complex_word_count,
        'WORD COUNT': word_count,
        'SYLLABLE PER WORD': syllables_per_word,
        'PERSONAL PRONOUNS': personal_pronoun_count,
        'AVG WORD LENGTH': avg_word_length
       
    }

    # Append the output data to the list
    output_data.append(output_dict)

# Create a Pandas DataFrame from the output data
output_df = pd.DataFrame(output_data)

# Save the output DataFrame to an Excel file
output_df.to_excel('output.xlsx', index=False)

plt.figure(figsize=(10, 6))
df = output_df.rename(columns={'Sentiment\nScore': 'Sentiment Score'})
plt.bar(range(len(df)), df['Sentiment Score'])
plt.xlabel('Article Index')
plt.ylabel('Sentiment Score')
plt.title('Sentiment Scores by Article Index')
plt.show()