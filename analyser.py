# -*- coding: utf-8 -*-
__author__ = 'duyongan'
__date__ = '2018/6/28 10:01'
import jieba
jieba.initialize()
import re
import jieba.posseg as pseg
import networkx as nx
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
import numpy as np
import collections
import nltk
import nltk.data
from nltk.corpus import stopwords as en_stopwords


class analyse:
    def __init__(self,text,lang,stopwords,idf_map,my_dict):
        self.stopwords=stopwords
        self.idf_map=idf_map
        self.dict=my_dict
        self.en_stopwords=en_stopwords.words('english')
        # self.text = text
        self.lang = lang
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        # text = ''.join([t for t in text.split('\n') if len(t) > 4])
        text = text.replace('\u3000', '').replace('？”', '”').replace('！”', '”').replace('。”', '”')\
            .replace('\r','').replace('\t','').replace('\xa0','')
        if lang=='zh':
            text=re.sub('（.*?）', '', text)
            sentences = re.split(r"([。！？……])", text)
            sentences_1=[]
            for sen in sentences:
                if len(sen)>200:
                    for s in sen.split('\n'):
                        if len(s.replace('\n',''))>18:
                            if sen.find('来源：') == -1:
                                sentences_1.append(s.replace('\n', ''))
                                sentences_1.append('。')
                else:
                    if sen.find('来源：')==-1:
                        sentences_1.append(sen.replace('\n', ''))
            sentences=sentences_1
            if len(text)>500:
                sentences=sentences[:20]
            sentences.append('')
            self.sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2]) if len("".join(i))<200 and len("".join(i))>18]
            # 分词
            sentences2=[]
            for sentence in self.sentences:
                words =[tuple for tuple in list(pseg.cut(sentence))if list(tuple)[0].strip()]
                words2=[]
                temp=''
                enstart=False
                for i in range(len(words)):
                    if words[i].flag in ['n','nd','nh','ni','nl','ns','nt','nz','vn','nr','nrf','nsf','ng','nrj','nr1','nr2'] and len(temp)<=4 and not enstart:
                        if words[i].word not in self.stopwords:
                            temp=temp+words[i].word
                        if i==len(words)-1:
                            if temp.strip()!='':
                                words2.append(temp)
                    else:
                        if temp.strip()!='' and not enstart:
                            words2.append(temp)
                            temp=''
                        if words[i].flag=='eng':
                            en_word=nltk.pos_tag([words[i].word])[0][1]
                            if  en_word in ['NN','NNS','NNP','NNPS'] and words[i].word not in self.en_stopwords:
                                if enstart:
                                    if len(temp.strip().split()) > 2 or temp.strip().isupper() and temp.strip() not in self.en_stopwords:
                                        words2.append(temp.strip())
                                        temp=''
                                    if temp:
                                        temp+=' '+words[i].word.strip()
                                    else:
                                        temp = words[i].word.strip()
                                else:
                                    temp=words[i].word.strip()
                                    enstart=True
                                try:
                                    if words[i+1].flag!='eng':
                                        if len(temp.strip().split())>2 or temp.strip().isupper() and temp.strip() not in self.en_stopwords:
                                            words2.append(temp.strip())
                                        enstart=False
                                        temp=''
                                except:
                                    words2.append(temp.strip())
                                    enstart = False
                                    temp=''

                            if i+1<len(words):
                                if words[i + 1].flag != 'eng':
                                    if temp:
                                        if len(temp.strip().split()) > 2 or temp.strip().isupper() and temp.strip() not in self.en_stopwords :
                                            words2.append(temp.strip())
                                    enstart = False
                                    temp = ''
                            else:
                                if temp:
                                    if len(temp.strip().split()) > 2 or temp.strip().isupper() and temp.strip() not in self.en_stopwords:
                                        words2.append(temp.strip())
                                enstart = False
                                temp = ''
                sentences2.append(words2)
        elif lang=='en':
            text = re.sub('\(.*?\)', '', text)
            text = ''.join([t for t in text.split('\n') if len(t.split()) > 4])
            sentences = self.tokenizer.tokenize(text.replace('\n', ''))
            # sentences = re.split(r"([.?!…])", )
            sentences.append('')
            self.sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]
            self.sentences = [sentence for sentence in self.sentences if len(sentence.split())>18 and len(sentence.split())<200]
            if len(text.split())>500:
                self.sentences=self.sentences[:20]
            sentences2=[]
            for sentence in self.sentences:
                words =list(nltk.pos_tag(sentence.split()))
                words2=[]
                temp=''
                for i in range(len(words)):
                    if words[i][1] in ['NN','NNS','NNP','NNPS'] and len(temp.split())<4 and words[i][0] not in self.en_stopwords:
                        if temp:
                            temp=temp.strip()+' '+words[i][0].strip()
                        else:
                            temp=words[i][0].strip()
                        if i==len(words)-1:
                            if len(temp.strip().split()) > 0 or temp.strip().isupper() and temp not in self.en_stopwords :
                                words2.append(temp)
                    else:
                        if len(temp.strip().split()) > 0 or temp.strip().isupper() and temp not in self.en_stopwords :
                            words2.append(temp)
                        temp=''
                sentences2.append(words2)
        else:
            return []
        #去停用词和单字
        self.sentences3=[]
        for sentence in sentences2:
            sentence2=[]
            for word in sentence:
                if word in self.en_stopwords:
                    pass
                elif len(word)<=2:
                    pass
                else:
                    sentence2.append(word)
            if len(sentence2)>1:
                self.sentences3.append(sentence2)

        #单字词频统计
        def flatten(x):
            result = []
            for el in x:
                if isinstance(x, collections.Iterable) and not isinstance(el, str):
                    result.extend(flatten(el))
                else:
                    result.append(el)
            return result
        word_map={}
        word_list=flatten(self.sentences3)
        for word in list(set(word_list)):
            word_map[word]=word_list.count(word)
        #词对频数统计
        word2word_map={}
        word2word=[]
        for sentence in self.sentences3:
            for i in range(len(sentence)):
                for j in range(i+1,len(sentence)):
                    alist=[]
                    alist.append(sentence[i])
                    alist.append(sentence[j])
                    alist=sorted(alist)
                    word2word.append('_'.join(alist))
        for w2w in list(set(word2word)):
            word2word_map[w2w]=word2word.count(w2w)

        #计算共现网络权重
        word2word_weight={}
        for w2w in list(set(word2word)):
            word2word_weight[w2w]=(word2word_map[w2w]/word_map[w2w.split('_')[0]]+word2word_map[w2w]/word_map[w2w.split('_')[1]])*0.5

        #共现网络可视化
        G=nx.Graph()
        word_list2=[]
        for word in list(set(word_list)):
            word=unicode(word.encode("utf-8"),'utf-8')
            word_list2.append(word)
        G.add_nodes_from(word_list2)
        for w2w in word2word_weight.keys():
            G.add_edge(unicode(w2w.split('_')[0].encode("utf-8"),'utf-8'),unicode(w2w.split('_')[1].encode("utf-8"),'utf-8'),weight=word2word_weight[w2w])
        word_map2=[]
        for word in G.nodes:
            word_map2.append((word,word_map[word]))
        fre=[]
        idf_num=[]
        max_idf_num=max(self.idf_map.values())
        for flu in word_map2:
            fre.append(flu[1])
            try:
                idf_num.append(self.idf_map[flu[0]])
            except:
                idf_num.append(max_idf_num)
        fre=np.array(fre)
        idf_num=np.array(idf_num)
        try:
            pr1=np.array(list(nx.degree_centrality(G).values()))
        except:
            pr1=np.zeros(len(word_list2))
        try:
            pr2=np.array(list(nx.eigenvector_centrality(G).values()))
        except:
            pr2=np.zeros(len(word_list2))
        try:
            pr3=np.array(list(nx.betweenness_centrality(G).values()))
        except:
            pr3=np.zeros(len(word_list2))
        pr4=fre
        weight_=list((0.1*pr1/max(0.001,sum(pr1))+0.1*pr2/max(0.001,sum(pr2))+0.5*pr3/max(0.001,sum(pr3))+0.3*pr4*idf_num/max(0.001,sum(pr4))))
        for i,word in enumerate(G.nodes):
            if word.isupper():
                weight_[i] = weight_[i] * 0.25
            if word in self.dict:
                weight_[i] = weight_[i] * 10
        self.keywords=dict(zip(G.nodes,weight_))

    def getKeywords(self,num_of_keywords):
        keywords = sorted(self.keywords.items(), key=lambda k: k[1],reverse=True)
        keywords=keywords[:min(num_of_keywords,len(keywords)-1)]
        keywords=[term[0] for term in keywords]
        return keywords


    def getAbstract(self,num_of_abstract):
        if self.lang=='zh':
            sentences_score = {}
            for i,sentence in enumerate(self.sentences3):
                sentence_score = 0
                if len(sentence) > 0:
                    for word in sentence:
                        sentence_score += self.keywords[word]
                    sentences_score[i] = sentence_score / len(sentence)
                    if i==0:
                        sentences_score[i] =sentences_score[i] *1000
                    if len(self.sentences[i])>50:
                        sentences_score[i] = 0.01*sentences_score[i]
                else:
                    sentences_score[i] =0
            sentences_score = sorted(sentences_score.items(), reverse=True, key=lambda k: k[1])
            results = []
            for sentence_num in sorted(sentences_score[:num_of_abstract]):
                sentence = ''
                seq = re.split('([，；])', self.sentences[sentence_num[0]])
                seq.append('')
                seq = ["".join(i) for i in zip(seq[0::2], seq[1::2])]
                for sen in seq:
                    words = list(jieba.cut(sen))
                    words2 = []
                    for word in words:
                        if word in self.stopwords:
                            pass
                        elif len(word) <= 1:
                            pass
                        else:
                            words2.append(word)
                    if len(sen) <= 4:
                        __if_useful = False
                        word_flags = pseg.lcut(sen)
                        for word_flag in word_flags:
                            if word_flag.flag in ['n', 'nd', 'nh', 'ni', 'nl', 'ns', 'nt', 'nz', 'vn', 'nr', 'nrf', 'nsf',
                                                  'ng', 'nrj', 'nr1', 'nr2']:
                                __if_useful = True
                                break
                        if __if_useful:
                            if sentence != '':
                                sentence = sentence + sen
                    elif len(words2) == 0:
                        pass
                    else:
                        if sentence != '':
                            sentence = sentence + sen
                        else:
                            sentence = sentence + sen
                if len(sentence.split('：')[0])<4:
                    sentence=sentence.replace(sentence.split('：')[0], '')
                results.append(sentence)
        else:
            sentences_score = {}
            for i,sentence in enumerate(self.sentences3):
                sentence_score = 0
                if len(sentence) > 0:
                    for word in sentence:
                        sentence_score += self.keywords[word]
                    sentences_score[i] = sentence_score / len(sentence)
                    if i==0:
                        sentences_score[i] =sentences_score[i] *10
                    if len(self.sentences[i].split())>50:
                        sentences_score[i] = 0.01*sentences_score[i]
                else:
                    sentences_score[i] =0
            sentences_score = sorted(sentences_score.items(), reverse=True, key=lambda k: k[1])
            results = []
            for sentence_num in sorted(sentences_score[:num_of_abstract]):
                results.append(self.sentences[sentence_num[0]])
        return ''.join(results).replace('\t','')