import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt


df = pd.read_csv('Corona_NLP_test.csv')

# Checking the structure of the dataset
print(df.head())

# Tokenize the text corpus
df['TokenizedText'] = df['OriginalTweet'].apply(word_tokenize)

# Checking the updated dataframe
print(df.head())

# Remove stop words from the tokenized text
stop_words = set(stopwords.words('english'))
df['FilteredText'] = df['TokenizedText'].apply(lambda tokens: [token for token in tokens if token.lower() not in stop_words])

# Checking the updated dataframe
print(df.head())

# Count word frequencies in the filtered text
word_counts = Counter([word for tokens in df['FilteredText'] for word in tokens])

# Displaying the word frequencies
print(word_counts.most_common(10))

# Generate a word cloud from the filtered text
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()