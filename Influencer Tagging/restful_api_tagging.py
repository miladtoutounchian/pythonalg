import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim import corpora, models
import math
import heapq
from gensim.summarization import keywords
import rake
import urllib
import json
import re
import nltk
from pattern.text.en import singularize
import summa.keywords
import networkx as nx
import os
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
from slidingWindow import SlidingWindow
from collections import OrderedDict
from itp import itp
from wordsegment import segment
import math
import codecs
import pickle
import sys
# from instagramcrawler import InstagramCrawler
import os
from nltk.tag.stanford import StanfordPOSTagger

words = set(nltk.corpus.words.words())
all_words = [unicode(line.rstrip(), encoding='utf-8') for line in open('reduced_all_words.txt')]
stop_words = set(stopwords.words('english'))
np.random.seed(42)


def keyword_sklearn(docs, k):
    lines = [line.rstrip() for line in open('SmartStoplist.txt')]
    stop_list = set(lines)
    vectorizer = TfidfVectorizer(stop_words=stop_list)
    tfidf_matrix = vectorizer.fit_transform(docs)
    tfidf_scores = np.sum(tfidf_matrix, axis=0)
    tfidf_scores = np.ravel(tfidf_scores)
    return sorted(dict(zip(vectorizer.get_feature_names(), tfidf_scores)).items(), key=lambda x: x[1], reverse=True)[:k]


def keyword_gensim_tf_idf(docs, k):
    lines = [line.rstrip() for line in open('SmartStoplist.txt')]
    stop_list = set(lines)
    texts = [[word for word in gensim.utils.tokenize(document, lowercase=True, deacc=True,
                          errors='replace') if word not in stop_list] for document in docs]

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    word_tf_idf_dic = {}
    for doc in corpus_tfidf:
        for t in doc:
            key = dictionary.get(t[0])
            score = t[1]
            if key in word_tf_idf_dic:
                word_tf_idf_dic[key] += score
            else:
                word_tf_idf_dic[key] = score
    return sorted(word_tf_idf_dic.items(), key=lambda x: x[1], reverse=True)[:k]


def keyword_gensim_lda(docs, k=5, num_topics=10, num_words=5):
    lines = [line.rstrip() for line in open('SmartStoplist.txt')]
    stop_list = set(lines)
    texts = [[word for word in gensim.utils.tokenize(document, lowercase=True, deacc=True,
                          errors='replace') if word not in stop_list] for document in docs]

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda = gensim.models.LdaModel(corpus, id2word=dictionary,
                                 num_topics=num_topics)
    gensim_topics = [t[1] for t in lda.show_topics(num_topics=num_topics,
                                                   num_words=num_words, formatted=False)]
    topics = [[(i[1], i[0]) for i in t] for t in gensim_topics]
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


def rake_doc_pre(docs):
    s = ""
    for d in docs:
        s = s + " " + d
    return s


def keyword(docs, ratio):
    s = ""
    for d in docs:
        s = s + " " + d
    return keywords(s, ratio=ratio)


# def summa_keyword(docs):
#     s = ""
#     for d in docs:
#         s += d
#     return summa.keywords.keywords(s)


def summa_keyword(docs):
    C = []
    for i in docs:
        B = ['', '']
        A = i.split()
        for j in range(len(A) - 2):
            B[0] = A[j]
            B[1] = A[j + 1]
            if B[0] == B[1]:
                A[j] = ''
        C.append(' '.join(A))
    s = ""
    for d in C:
        s += d
    return summa.keywords.keywords(s)


def list_for_keyword(x):
    s = ""
    keywords = []
    for i in x:
        if i is not unicode('\n'):
            s = s+i
        else:
            keywords.append(s)
            s = ""
    if s:
        keywords.append(s)
    return keywords


def intersection(*arg):
    result = set(arg[0])
    for i in range(1, len(arg)):
        result = result & set(arg[i])
    return list(result)


def extractSummary(text, sent, dist, draw='F'):
    # Tokenize on the sentence
    sentenceTokens = sent_tokenize(text)

    # Clean the text and remove stop words
    s_f = cleanText(sentenceTokens)

    graph = generateGraph(s_f, dist)
    return graph

    calculated_page_rank = nx.pagerank(graph, max_iter=100, weight='weight')
    # return calculated_page_rank
    # most important sentences in ascending order of importance
    sentences = sorted(calculated_page_rank, key=calculated_page_rank.get, reverse=True)

    # return sentences
    number_sent = int(sent)

    summary = '. '.join([s for s in sentences[:number_sent]])

    return summary


def removeDuplicates(taggedTokens):
    L = []
    for i in taggedTokens:
        if i not in L:
            L.append(i)
        else:
            pass

    return L


def generateCoocurrenceMatrix(taggedTokens, windowLength):
    i = 0
    matrix_row = matrix_col = len(taggedTokens)
    w = []
    # First considering a fully connected graph
    # Filling diagonal entries to be `ZERO` {ignoring self loop}
    mat = np.zeros((matrix_row, matrix_col), int)

    # Matrix of `matrix_row` X `matrix_col` will be made
    # Logic to fill in the cell with co-occurence distance
    sliding_w = SlidingWindow(windowLength, taggedTokens)

    for i in range(len(taggedTokens)):
        prev, curr, nxt = sliding_w[i][0], sliding_w[i][1], sliding_w[i][2]
        for _i in prev:
            w.append(_i)
        w.append(curr)
        for i_ in nxt:
            w.append(i_)

        for j in range(len(taggedTokens)):
            if i == j:
                mat[i][j] = 0
            else:
                if taggedTokens[j] in w:
                    distance = abs(w.index(taggedTokens[j]) - w.index(taggedTokens[i]))
                    mat[i][j] = distance
        w = []
    return mat


def makeTokenGraph(nodePairsAndWeight):
    gr = nx.Graph()
    for i in nodePairsAndWeight:
        gr.add_node(i[0])
        gr.add_node(i[1])
        gr.add_edge(i[0], i[1])
        gr[i[0]][i[1]]['weight'] = i[2]

    return gr


def postProcessingKeywords(words, wordTokens):
    j = 1
    kPhrase = ''
    Phrases = []

    while j < len(wordTokens):
        fWord = wordTokens[j - 1]
        sWord = wordTokens[j]

        if fWord in words and sWord in words:
            if not fWord == sWord:
                kPhrase = fWord + ' ' + sWord
                Phrases.append(kPhrase)
                words.remove(fWord)
                words.remove(sWord)
        j += 1

    for i in words:
        Phrases.append(i)
    return Phrases


def extractKeywords(text, window=11):
    text1 = text.lower()

    # Clean the input
    # `[text]` because `cleanText` accepts that format
    # Also tokenize it
    text = cleanText([text1])
    wordTokens = nltk.word_tokenize(text[0])

    # Do POS Tagging and extract Nouns and Adjectives
    taggedTokens = nltk.pos_tag(wordTokens)
    taggedTokens = [i[0] for i in taggedTokens if i[1] in ["NN", "JJ", "NNP", "NNS", "JJS", "JJR", "NNPS"]]

    # return taggedTokens
    # Remove the duplicate ones | Not required here
    uniqueTokens = removeDuplicates(taggedTokens)

    # return uniqueTokens
    # Window size over which co-occurence matrix will be generated
    # Sliding Window of length of 5 means [w1,w2,w3,w4,w5,W,w6,w7,w8,w9,w10]
    windowSize = window
    mat = generateCoocurrenceMatrix(uniqueTokens, windowSize)
    # return mat
    # Node pairs that will make the graph
    n1, n2 = np.nonzero(mat)[0], np.nonzero(mat)[1]
    nodePairs = [(node1, node2) for node1, node2 in zip(n1, n2)]

    # Weight for all the connected node pairs
    weightList = []
    for n in nodePairs:
        w = mat[n[0]][n[1]]
        weightList.append(w)

    # Nodes in graph(pair) and their connected weight
    nodePairsAndWeight = [(node1, node2, weight) for node1, node2, weight in zip(n1, n2, weightList)]
    # return nodePairsAndWeight
    # print "Generating graph...\n"
    graph = makeTokenGraph(nodePairsAndWeight)

    # Apply PageRank
    calculated_page_rank = nx.pagerank(graph, weight='weight')
    # return calculated_page_rank
    # return uniqueTokens

    words = sorted(calculated_page_rank, key=calculated_page_rank.get, reverse=True)
    # return words
    # return words
    # Joining the keywords : if keywords selected occcur adjacent to eachother in the original text

    words_w = [uniqueTokens[i] for i in words[:30]]
    text1 = "".join([i for i in text1 if i not in punctuation])
    # print(words_w)
    # print(text1.split())
    words_w_f = postProcessingKeywords(words_w, text1.split())
    return words_w_f


stopModList = set(stopwords.words('english'))


def cleanText(text):
    s_f = []

    # For every sentence in sentenceTokens
    for s in text:
        stopRemoved = ' '.join([w for w in s.split() if w not in stopModList])
        puncRemoved = ''.join([w for w in stopRemoved if w not in punctuation])
        s_f.append(puncRemoved)
    return s_f


def word_count(string):
    my_string = string.lower().split()
    my_dict = {}
    lines = [line.rstrip() for line in open('SmartStoplist.txt')]
    stop_list = set(lines)
    for item in my_string:
        if item not in stop_list:
            if item in my_dict:
                my_dict[item] += 1
            else:
                my_dict[item] = 1
    return my_dict


def top_k_word(dictionary, k):
    return heapq.nlargest(k, dictionary, key=dictionary.get)


def top_keywords(*alg):
    all_keywords = []
    for i in alg:
        all_keywords.append(sum([item.split() for item in i], []))
    my_dict = {}
    for word in sum(all_keywords, []):
        if word in my_dict:
            my_dict[word] += 1
        else:
            my_dict[word] = 1

    return top_k_word(my_dict, k=3)


def phrasal_keyword_extraction(alg1, alg2):
    phrasal_keyword = intersection(alg1, alg2)
    ls = []
    for word in phrasal_keyword:
        if len(word.split()) >= 2:
            ls.append(word)
    return ls


def hashtag_most_frequent(hastag_list, k):
    lines = [line.rstrip() for line in open('SmartStoplist.txt')]
    stop_list = set(lines)
    hastag_list = [tag for tag in hastag_list if tag not in stop_list]
    word_counter = {}
    for word in hastag_list:
        if word in word_counter:
            word_counter[word] += 1
        else:
            word_counter[word] = 1
    popular_hashtags = sorted(word_counter, key=word_counter.get, reverse=True)
    return popular_hashtags[:k]


def keyword_to_category_GloVe(keyword_list):
    dic = {}
    with codecs.open('glove.840B.300d.txt', 'r') as f:
    # with codecs.open('glove.6B.300d.txt', 'r', 'utf-8') as f:
        for c, r in enumerate(f):
            sr = r.split()
            # if sr[0] in keyword_list + category_list:
            if sr[0] in [i.encode() for i in keyword_list]:
                # print(sr[0])
                dic[sr[0]] = [float(i) for i in sr[1:]]
                # print(c)
                if len(dic) == len(keyword_list):
                    break
    category_list = pickle.load(open("category2.p", "rb"))
    category = {}
    for i in keyword_list:
        distance = []
        for j in category_list:
            distance.append([j, np.linalg.norm(np.array(dic[i])-np.array(category_list[j]))])
        di = [s[0] for s in distance]
        mi = [s[1] for s in distance]
        idx = mi.index(min(mi))
        category[i] = di[idx]
    return category


def refine(dictionary, ls):
    for key, value in dictionary.items():
        if value in ls:
            del dictionary[key]
    return dictionary


def refine_list(lst, ls):
    l = lst
    A = []
    for e in l:
        if e in ls:
            ind = l.index(e)
            A.append(ind)
    return [l[i] for i in range(len(l)) if i not in A]


def keyword_pos_tagger_re(keyword_ls):
    for tpl in nltk.pos_tag(keyword_ls):
        if tpl[1] in ['JJ', 'DT', 'IN', 'VBZ']:
            i = keyword_ls.index(tpl[0])
            del keyword_ls[i]
    return keyword_ls


class Tag:

    def __init__(self, url):
        self.url = url
        self.hash = []
        self.caption = []
        self.biography = ''
        self.documents = None
        self.keyword_tf_idf_sklearn = []
        self.keyword_tf_idf_gensim = []
        self.keyword_lda_gensim = []
        self.keyword_rake = []
        self.keyword_gensim = []
        self.keyword_summa = []
        self.keyword_textrank = []
        self.keywords = []
        self.top_words = []
        self.phrasal_keywords = []
        self.keyword_hashtag = []
        self.top_hash = []
        self.category = {}

    def caption_biography(self, last_posts=40):
        try:
            response = urllib.urlopen(self.url)
            data = json.loads(response.read())
            date = []
            for t in range(len(data['user']['media']['nodes'])):
                date.append(data['user']['media']['nodes'][t]['date'])
            ind = [sorted(date, reverse=True).index(v) for v in date]
            self.caption = []
            for num_caption in ind[-last_posts:]:
                if 'caption' in data['user']['media']['nodes'][num_caption]:
                    self.caption.append(data['user']['media']['nodes'][num_caption]['caption'])
            self.biography = data['user']['biography']
            return self.caption, self.biography
        except StandardError:
            self.caption = []
            self.biography = []
            return self.caption, self.biography

    def clean_caption_biography(self):
        caption, biography = self.caption_biography()
        if biography:
            caption.append(biography)
            self.caption = caption
        else:
            self.caption = caption
        clean_cap_bio = []
        for unclean_text in self.caption:
            if isinstance(unclean_text, unicode):
                # remove emoji, hashtag, url, html
                clean_text = ' '.join(
                    re.sub("([@#][A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", unclean_text).split())

                clean_text = " ".join(
                    singularize(w).lower() for w in nltk.wordpunct_tokenize(clean_text) if
                    singularize(w).lower() in words and w.isalpha())
                clean_cap_bio.append(clean_text)
            else:
                clean_cap_bio = []
        self.documents = clean_cap_bio
        return self.documents

    def keyword(self):
        self.keyword_tf_idf_sklearn = OrderedDict(keyword_sklearn(self.documents, k=10)).keys()
        self.keyword_tf_idf_gensim = OrderedDict(keyword_gensim_tf_idf(self.documents, k=10)).keys()
        self.keyword_lda_gensim = OrderedDict(keyword_gensim_lda(self.documents, k=10)).keys()
        rake_object = rake.Rake("SmartStoplist.txt", 3, 2, 1)
        keywords_rake = rake_object.run(rake_doc_pre(self.documents))
        keyword_only_rake = []
        for keyword_score in keywords_rake:
            keyword_only_rake.append(keyword_score[0])
        self.keyword_rake = keyword_only_rake
        self.keyword_gensim = list_for_keyword(list(keyword(self.documents, ratio=0.7)))

        conservative_keywords = []
        for i in self.keyword_tf_idf_gensim:
            if all([any([i in j for j in self.keyword_rake]),
                    any([i in j for j in self.keyword_gensim]),
                    any([i in j for j in self.keyword_tf_idf_sklearn])]) is True:
                conservative_keywords.append(i)

        # self.keyword_summa = list_for_keyword(list(summa_keyword(self.documents)))
        # self.keyword_textrank = extractKeywords(rake_doc_pre(self.documents).decode('utf-8'))
        # counted_keywords = top_keywords(self.keyword_textrank, self.keyword_gensim, self.keyword_tf_idf_sklearn,
        #                              self.keyword_lda_gensim, self.keyword_rake, self.keyword_summa)
        self.keyword_textrank = extractKeywords(rake_doc_pre(self.documents).decode('utf-8'))
        counted_keywords = top_keywords(self.keyword_textrank, self.keyword_gensim, self.keyword_tf_idf_sklearn,
                                        self.keyword_lda_gensim, self.keyword_rake)
        self.keywords = list(set(counted_keywords + conservative_keywords))
        self.top_words = top_k_word(word_count(rake_doc_pre(self.documents)), k=3)
        self.phrasal_keywords = phrasal_keyword_extraction(self.keyword_rake, self.keyword_textrank)

    def hash_tag_extractor(self):
        C = self.caption
        # hash tag extraction
        hash_tag = []
        for i in C:
            p = itp.Parser()
            result = p.parse(i)
            for item in result.tags:
                if item:
                    # hash_tag.append(item)
                    for seg in segment(item):
                        hash_tag.append(seg)
        self.hash = hash_tag
        self.top_hash = hashtag_most_frequent(self.hash, k=3)
        return self.hash, self.top_hash

    def influencer_tag(self, pos=0):
        keyword_list_single = self.keywords + self.top_words
        # Eliminate words which is not in all_words
        hashtag = [w for w in self.top_hash if w in all_words]
        # print('Milad:')
        # print(hashtag)
        keyword_hastag = keyword_list_single + hashtag
        keyword_hastag = list(set(keyword_hastag))
        if pos == 1:
            keyword_hastag = keyword_pos_tagger_re(keyword_hastag)
        filtered_keyword_hashtag = [w for w in keyword_hastag if len(w) > 2]
        self.keyword_hashtag = refine_list(filtered_keyword_hashtag, ls=['check', 'day', 'good', 'today', 'tomorrow',
                                                                         'training', 'opening', 'price'])

        category = keyword_to_category_GloVe(self.keyword_hashtag)
        # Refine category dictionary for key and value pairs that values are in pre-defined list (time for example)
        category = refine(category, ls=['time', 'city', 'tree', 'place'])
        self.category = list(set(category.values()))
        return self.category


from flask_restplus import Api, Resource, fields
from flask import Flask, jsonify, request, make_response, abort, render_template, redirect, url_for

app = Flask(__name__)
api = Api(app, version='2.0', title='MuseFind Tagging API', description='Automated Tagging By NLP')
ns = api.namespace('MuseFind_api', description='Methods')
single_parser = api.parser()
single_parser.add_argument('influencer', help='influencer username', required=True)
single_parser.add_argument('pos', help='remove adj', type=int, required=True)


@ns.route('/tagging')
class Tagging(Resource):
    @api.doc(parser=single_parser, description='Enter Influencer Username and do you want pos remover')
    def get(self):
        """Insert the influencer username to get its keyword and its tags (Click to see more)"""
        args = single_parser.parse_args()
        URL = "https://www.instagram.com/{}/?__a=1".format(args.influencer)
        tag = Tag(URL)
        tag.clean_caption_biography()
        tag.keyword()
        # print(tag.keywords)
        # print(tag.top_words)
        # print(tag.phrasal_keywords)
        tag.hash_tag_extractor()
        # print(tag.top_hash)
        tag.influencer_tag(pos=args.pos)
        print(tag.keyword_hashtag)

        return {'Influencer': args.influencer, 'Keyword': tag.keyword_hashtag, 'Phrasal Keyword': tag.phrasal_keywords, 'Tags': tag.category}

if __name__ == '__main__':
    app.run()





