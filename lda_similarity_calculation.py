import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import gensim
from gensim import corpora, models
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 下载必要的NLTK数据
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


def preprocess_text(text):
    """
    文本预处理：分词、去停用词、词形还原
    """
    # 分词
    tokens = word_tokenize(text.lower())

    # 移除标点和停用词
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

    # 词形还原
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens


def calculate_lda_similarity(text1, text2, num_topics=2):
    """
    计算两段文本的LDA主题相似性
    """
    # 预处理文本
    texts = [preprocess_text(text1), preprocess_text(text2)]

    # 创建词典和词袋
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # 训练LDA模型
    lda_model = models.LdaModel(corpus=corpus,
                                id2word=dictionary,
                                num_topics=num_topics,
                                random_state=42,
                                passes=15)

    # 获取文档主题分布
    doc1_topics = lda_model.get_document_topics(corpus[0], minimum_probability=0)
    doc2_topics = lda_model.get_document_topics(corpus[1], minimum_probability=0)

    # 转换为概率向量
    vec1 = np.array([prob for _, prob in doc1_topics])
    vec2 = np.array([prob for _, prob in doc2_topics])

    # 计算余弦相似度
    similarity = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]

    return similarity, lda_model, texts


# 示例文本
text1 = """
Artificial intelligence is transforming various industries. 
Machine learning algorithms can now recognize patterns in large datasets 
and make predictions with remarkable accuracy.
"""

text2 = """
Deep learning models have revolutionized computer vision tasks. 
These neural networks can identify objects in images almost as well as humans, 
enabling advancements in autonomous vehicles and medical imaging.
"""

# 计算相似度
similarity, model, processed_texts = calculate_lda_similarity(text1, text2)

print(f"文本语义相似度: {similarity:.4f}")
print("\nLDA模型主题词:")
for topic_id in range(model.num_topics):
    words = model.show_topic(topic_id, topn=5)
    topic_words = ", ".join([word for word, _ in words])
    print(f"主题 #{topic_id}: {topic_words}")