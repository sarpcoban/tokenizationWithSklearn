from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup
import requests

url = input("Enter a wikipedia URL that you want to scrap:")
ngram_start = input("Enter a start value of ngram:")
ngram_start = int(ngram_start)
ngram_end = input("Enter an end value of ngram:")
ngram_end = int(ngram_end)
corpus = []

response = requests.get(url)

soup = BeautifulSoup(response.content, "html.parser")

content = soup.find("div", {"id": "mw-content-text"})

paragraphs = content.find_all("p")

for paragraph in paragraphs:
    print(paragraph.get_text())
    corpus.append(paragraph.get_text())

vectorizer = CountVectorizer(ngram_range=(ngram_start, ngram_end))

X = vectorizer.fit_transform(corpus)
tokens = vectorizer.get_feature_names_out()
print(tokens)

print(X.toarray())
