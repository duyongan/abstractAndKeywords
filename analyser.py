# -*- coding: utf-8 -*-
# @Time    : 18-9-20 下午1:59
# @Author  : duyongan
# @FileName: analyser.py
# @Software: PyCharm
import re
from jieba import posseg
import hashlib
import numpy as np
import pickle

class analyse:
    def __init__(self,title,text,num_of_abstract=1,num_of_keywords=5):
        with open('stopwords', 'rb') as f:
            stopwords = pickle.load(f)
        with open('new_idf_map', 'rb') as f:
            new_idf_map = pickle.load(f)
        md5 = hashlib.md5()

        def _xor_sim(i, j):
            return list(np.logical_xor(i,j)).count(False)/ ((len(i)+len(j))/2)

        def _get_nlp_hash_code(text):
            text = text.replace('\u3000', '').replace('？”', '”').replace('！”', '”').replace('。”', '”') \
                .replace('\r', '').replace('\t', '').replace('\xa0', '')
            text = posseg.lcut(text)
            text_hash_code_list = [word_.word for word_ in text if
                                   len(word_.word) > 1 and word_.word not in stopwords and word_.flag.find(
                                       'n') != -1]
            word_hash_code_list = []
            for word in text_hash_code_list:
                md5.update(word.encode('utf-8'))
                temp = list(str(bin(int(md5.hexdigest(), 16)))[2:])
                if len(temp) != 128:
                    temp = [0] * (128 - len(temp)) + temp
                try:
                    new_idf_map[word]
                except:
                    new_idf_map[word] = 3
                temp = np.array([int(-1) if i == '0' else int(i) for i in temp]) * new_idf_map[word]
                word_hash_code_list.append(temp)
            text_hash_code = np.array(word_hash_code_list).sum(axis=0)
            text_hash_code = [int(1) if i > 0 else int(-1) for i in list(text_hash_code)]
            return text_hash_code_list, np.array(text_hash_code)

        _,title_hash_code=_get_nlp_hash_code(title)
        text = text.replace('\u3000', '').replace('？”', '”').replace('！”', '”').replace('。”', '”')\
            .replace('\r','').replace('\t','').replace('\xa0','')
        text = re.sub('（.*?）', '', text)
        sentences = re.split(r"([。！？……])", text)
        sentences_1 = []
        for sen in sentences:
            if len(sen) > 200:
                for s in sen.split('\n'):
                    if len(s.replace('\n', '')) > 18:
                        sentences_1.append(s.replace('\n', ''))
                        sentences_1.append('。')
            else:
                sentences_1.append(sen.replace('\n', ''))
        sentences = sentences_1
        if len(sentences) > 20:
            sentences = sentences[:20]
        sentences.append('')
        sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2]) if
                          len("".join(i)) < 200 and len("".join(i)) > 18]
        words=[]
        vecs=[]
        for sentence in sentences:
            word,vec=_get_nlp_hash_code(sentence)
            vecs.append(_xor_sim(vec,title_hash_code))
            words+=word
        abstract=[]
        for sim in sorted(vecs,reverse=True):
            key=vecs.index(sim)
            abstract.append(sentences[key])
        self.abstract=''.join(abstract[:num_of_abstract])
        keywords_set=list(set([word for word in words if len(word)>1]))
        keywords_count = [words.count(keyword) for keyword in keywords_set]
        keywords_count = np.array(keywords_count)/sum(keywords_count)
        keywords_count=[len(keyword)**3*keywords_count[i]*new_idf_map[keyword] for i,keyword in enumerate(keywords_set)]
        keywords_map=dict(zip(keywords_set,keywords_count))
        keywords = sorted(keywords_map.items(), key=lambda k: k[1], reverse=True)
        keywords=keywords[:min(num_of_keywords,len(keywords)-1)]
        keywords=[term[0] for term in keywords]
        self.keywords=','.join(keywords)





title='人工智能发展未来可期北京股商安徽分公司谈多方面需正规协同合作'
text="""
人工智能(AI)给现代社会带来积极的变革也许可以分为几个阶段：从可以拯救生命的自动驾驶汽车，到发现癌症治疗方法的数据分析程序，再到专门为无法说话的人设计的发声机器等等，人工智能将成为人类历史上最具革命性的创新之一。但是，要实现这一美好的愿景，仍有很长的一段路要走，而且这个过程需要持续的投入。人类社会遇到了史无前例的重大转型阶段，我们并没有一张现成的蓝图来为我们指引方向，但有一点是十分明确的：人工智能的挑战并不是一家公司、一个行业或是一个国家仅凭一己之力就能解决的，要想完全实现人工智能的美好愿景，需要整个技术生态系统和世界各国政府的通力合作。
　　为了实现这一愿景，产学界多年来一直在积极探索，而且一些早期的解决方案已经初见成效。各国政府和组织目前也正在积极制定战略推动人工智能的发展，来解决我们面临的一些挑战。中国、印度、英国、法国和欧盟等已经制定了正式的人工智能规划，我们需要更多国家层面的人工智能战略，最终让政府、产业届和学术界合作推动人工智能的长远发展。那么政府和行业组织要如何帮助推动人工智能发展?针对这个问题，建议优先考虑以下三点：
　　教育
　　从小学开始，学校系统在设计课程时就应该考虑到人工智能，并开发相关的教育课程。在这方面起步较早的是澳大利亚国立大学正在开发的人工智能学位课程，英特尔的资深研究员、澳大利亚计算机科学教授Genevieve Bell开创先河的设计了这门课程，我们需要看到更多这样的课程出现在学校中。学校也可以采取一些过渡措施，更好地从早期教育就鼓励实施STEM(科学、技术、工程、数学)教育。此外，例如面向数据科学家减免学费或者为他们增加更多的学位课程，将是培养更多人才的一条途径，我们急需这些人才来全面的实现人工智能效用。
　　另一方面，我们还要从人类本身为出发点去思考问题。比如大部分学校都教授学生基础的打字技能或者计算机技能，那么在未来的人工智能社会，学校就需要教学生学会“引导计算”技能，以便将来能够更好地利用机器去工作。因为在人工智能广泛应用的未来，很多工作肯定会实现自动化，因此不断的强化只有人类才能具备的技能也是非常至关重要的。
　　研发
　　为了制定有效的政策，要从人工智能的角度出发来采取行动。想做到这一点，最佳途径之一就是大力开展和加大研发投入。美国和欧洲等国家就正在推进关于算法可解释性的项目计划;而在英国，在政府的资助下，研发人员正在研究利用人工智能进行疾病的早期诊断、减少农作物的病害，并在公共部门提供数字化服务等等。这些做法都是值得肯定的，对于人类的发展也是多多益善的。
　　不同国家和行业组织间应该主动制定有效的方法来促进人类与人工智能之间的协作，以确保人工智能系统的安全性，并且应该开发可以用于人工智能训练和测试的共享公共数据集和共享环境。通过政府、产业界与学术界的互相协作，我们面临的很多的人工智能挑战都会迎刃而解。
　　监管环境
　　人工智能对法律法规体系也是有影响的。有关责任、隐私、安全和道德的法律政策数不胜数，而人工智都都可能在这些领域发挥作用，在制定法律法规之前，都需要进行周详缜密的讨论。如果单纯因为法律法规的界定而急于取缔各种形式的人工智能，这将阻碍人工智能行业的整体发展。对此，我们可以尽早采取积极措施推动数据以责任制和安全的方式被公开化，大力推动深度学习和人工智能的发展进度。
　　在医疗保健领域，数据的公开化将会带来很显著的影响。隐蔽掉具体身份信息后的医疗记录、基因组数据集、医疗研究和治疗计划等等都可以提供大量的数据，为人工智能提供其所需要的洞察力，帮助人类在精神健康、心血管疾病、药物治疗等方面取得突破性发现。在保护隐私和安全的前提下，如果允许研究员可以联合访问位于不同工作站的分布式存储库中的数据，这将让人工智能在人类健康建设的工程中发挥非常大的作用。
　　尽管我们对于人工智能的未来充满了期待，但仍然是前路漫漫。这需要政、产、学三界共同的努力，我们期待终有一天，人工智能为人类生活带来更积极的作用。
"""
analyser=analyse(title,text)
print(analyser.abstract)
print(analyser.keywords)