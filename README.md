#  abstractAndKeywords
# 基于图论与tf-idf的关键字与摘要算法优化。实验结果比textRank与tf-idf略好。
# 使用方法如下：
text="""
"【中国智能制造网 企业动态微软】联合通用电气宣布将拓展两家公司之间的合作，将运营技术和信息技术结合起来以消除工业企业在推进数字化转型项目方面面临的障碍。
据报道，微软联合通用电气宣布将拓展两家公司之间的合作，将运营技术和信息技术结合起来以消除工业企业在推进数字化转型项目方面面临的障碍。具体的合作是，通用电气旗下的软件业务部门“GF Digtal”计划在微软Azure云平台上标准化其Predix解决方案，并将Predix产品组合与Azure本地云功能包括物联网和数据分析进行深度融合。
"""
import pandas as pd\n
stopwords=pd.read_csv("stopwords.txt",index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')\n
stopwords=stopwords['stopword'].values\n
f=open('idf.txt',encoding='utf-8')\n
lines=f.readlines()\n
f.close()\n
idf_map={}\n
for line in lines:\n
    line=line.replace('\n','').split()\n
    idf_map[line[0]]=float(line[1])\n
analyser=analyse(text,10,2,stopwords,'zh',idf_map)\n
print(analyser.getAbstract())\n
print(analyser.getKeywords())\n
