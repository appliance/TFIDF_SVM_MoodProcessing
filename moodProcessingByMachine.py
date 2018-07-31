import jieba
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from numpy import zeros
import random
from sklearn.svm import SVC


# 获取去停用词
stopwords = []
with open('C:/Users/Administrator/Desktop/NLP/stopwords.txt', 'r') as f:
    for stopword in f.readlines():
        stopwords.append(stopword.strip())          # 读取每行的去停用词的时候需要把后面的换行去除，否则下面循环匹配去停用词的时候，根本都匹配不上

# 读取测试的所有文件内容，并进行预处理
with open('C:/Users/Administrator/Desktop/NLP/review_sentiment.v1/train2.rlabelclass', 'r') as f:
    # 创建一个测试集保存预处理结果
    train_list = []
    for line in f.readlines():
        cur_txt_name = (line.strip()).split(' ')[0]
        cur_txt_tag = (line.strip()).split(' ')[1]
        cur_txt_url = 'C:/Users/Administrator/Desktop/NLP/review_sentiment.v1/train2/' + ''.join(cur_txt_name)
        print('当前text文本地址为：', cur_txt_url)
        # 开始读取当前文件内容
        cur_txt_content = []
        with open(cur_txt_url, 'r') as f:
            for line in f.readlines():
                cur_txt_content.append(line.strip())
        # 开始进行jieba分词
        cur_txt_tokens = list(jieba.cut(''.join(cur_txt_content)))
        cur_removesw_txt_tokens = []
        for token in cur_txt_tokens:
            if token not in stopwords:
                cur_removesw_txt_tokens.append(token)
        cur_content_tag = (cur_removesw_txt_tokens, cur_txt_tag)
        # 已经获取了用于测试的测试集
        train_list.append(cur_content_tag)
                   # [content, tag]形式将训练集合中的文件内容保存在test_list中
    random.shuffle(train_list)


# 利用tfidf获取特征集
# 将文本内容矩阵转化为空格分隔的字符
txt_contents = []
for index in range(len(train_list)):
    txt_content = train_list[index][0]
    txt_contents.append(" ".join(txt_content))


# 利用sklearn进行训练，获取权重值大于0的单词集
vectorizer = CountVectorizer(min_df=1, token_pattern='(?u)\\b\\w+\\b')
transfromer = TfidfTransformer()
tf = vectorizer.fit_transform(txt_contents)
tfidf = transfromer.fit_transform(tf)
word = vectorizer.get_feature_names()
weight = tfidf.toarray()
# 将词和权重值保存到一个集合中
word_weight_group = []
for i in range(len(weight)):
    for j in range(len(word)):
        if weight[i][j] > 0:
            cur_word_weight = (word[j], weight[i][j])
            word_weight_group.append(cur_word_weight)
# 利用权重值对关键词进行排序
word_weight_group = sorted(word_weight_group, key=lambda word_weight_group:(word_weight_group[1]), reverse=True)

# 去除重复
word_weight_group_rm = []
for i in range(len(word_weight_group)):
    if word_weight_group[i][0] in word_weight_group_rm:
        pass
    else:
        word_weight_group_rm.append(word_weight_group[i][0])


token_features = word_weight_group_rm[:2000]
# 获取权重较高的特征词 保存在token_features中

# 利用svn向量机进行分类器的训练
features = zeros([len(train_list), len(token_features)], dtype=float)           # features 表示一个矩阵，此处对矩阵进行初始化
for n in range(len(train_list)):
    cur_document_words = set(train_list[n][0])
    for m in range(len(token_features)):
        if token_features[m] in cur_document_words:
            features[n, m] = 1


target = [c for (d, c) in train_list]
train_set = features[:5000, :]          # 训练集合  取前5000篇文章
target_train = target[:5000]            # 训练集合的结果

test_set = features[5000:10000, :]           # 测试集  取5000篇文章
target_test = target[5000:10000]             # 测试集合的结果

svclf = SVC(kernel='linear')
svclf.fit(train_set, target_train)
pred = svclf.predict(test_set)

print(sum([1 for n in range(len(target_test)) if pred[n]==target_test[n]])/len(target_test))



