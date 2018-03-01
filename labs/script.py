from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from pylab import barh, plot, yticks, show, grid, xlabel, figure

# newsgroups categories
categories = ['alt.atheism', 'talk.religion.misc',
              'comp.graphics', 'sci.space']

posts = fetch_20newsgroups(subset='train', categories=categories,
                           shuffle=True, random_state=42,
                           remove=('headers', 'footers', 'quotes'))

vectorizer = CountVectorizer(lowercase=True, stop_words='english')
X = vectorizer.fit_transform(posts.data)

# compute chi2 for each feature
chi2score = chi2(X, posts.target)[0]
figure(figsize=(6, 6))

wscores = zip(vectorizer.get_feature_names(), chi2score)
wchi2 = sorted(wscores, key=lambda x: x[1])
topchi2 = zip(*wchi2[-25:])
x = range(len(topchi2[1]))
labels = topchi2[0]
barh(x, topchi2[1], align='center', alpha=.2, color='g')
plot(topchi2[1], x, '-o', markersize=2, alpha=.8, color='g')
yticks(x, labels)
xlabel('$\chi^2$')
show()
