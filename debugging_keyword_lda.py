document_0 = "China has a strong economy that is growing at a rapid pace. However politically it differs greatly from the US Economy."
document_1 = "At last, China seems serious about confronting an endemic problem: domestic violence and corruption."
document_2 = "Japan's prime minister, Shinzo Abe, is working towards healing the economic turmoil in his own country for his view on the future of his people."
document_3 = "Vladimir Putin is working hard to fix the economy in Russia as the Ruble has tumbled."
document_4 = "What's the future of Abenomics? We asked Shinzo Abe for his views"
document_5 = "Obama has eased sanctions on Cuba while accelerating those against the Russian Economy, even as the Ruble's value falls almost daily."
document_6 = "Vladimir Putin is riding a horse while hunting deer. Vladimir Putin always seems so serious about things - even riding horses. Is he crazy?"

documents = [document_0, document_1, document_2, document_3, document_4, document_5, document_6]

from gensim import corpora, models
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from gensim.summarization import keywords
import summa.keywords
import rake
from collections import OrderedDict
import json
import heapq
import math
import operator


def keyword_gensim_lda(docs, k=5, num_topics=10, num_words=5):
    lines = [line.rstrip() for line in open('SmartStoplist.txt')]
    stop_list = set(lines)
    texts = [[word for word in gensim.utils.tokenize(document, lowercase=True, deacc=True,
                          errors='replace') if word not in stop_list] for document in docs]

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    lda = gensim.models.LdaModel(corpus, id2word=dictionary,
                                 num_topics=num_topics, alpha='auto', chunksize=1, eval_every=1)
    # print(lda.print_topics(num_topics=num_topics, num_words=num_words))
    gensim_topics = [t[1] for t in lda.show_topics(num_topics=num_topics,
                                                   num_words=num_words, formatted=False)]
    topics = [[(i[1], i[0]) for i in t] for t in gensim_topics]
    # print(topics)
    keywords = {}
    # Sum of probabilities for token in all topics
    for topic in topics:
        for t in topic:
            token = t[1]
            pr = t[0]
            if token in keywords:
                keywords[token] += pr
            else:
                keywords[token] = pr

    # Probability for each token multiplied by token frequency
    matrix = gensim.matutils.corpus2csc(corpus)
    for token, pr in keywords.items():
        for d in dictionary.items():
            if d[1] == token:
                token_index = d[0]
                break
        token_row = matrix.getrow(token_index)
        token_freq = token_row.sum(1).item()
        keywords[token] = pr * math.log(token_freq)

    # Sort keywords by highest score

    return sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:k]


top_k_keywords_lda_gensim = keyword_gensim_lda(documents, k=10)
print('top k words from gensim lda:')
print(OrderedDict(top_k_keywords_lda_gensim).keys())
print(top_k_keywords_lda_gensim)