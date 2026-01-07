import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

# 下载必要的NLTK数据（如果尚未下载）
# nltk.download('stopwords', quiet=True)

nltk.data.find('/newdisk/public/JZY/punkt_tab/english')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TopicConsistencyCalculator:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)  # 考虑单字和双字词组
        )

    def preprocess_text(self, text):
        """文本预处理：清洗、分词、词形还原"""
        # 清洗文本
        text = re.sub(r'[^\w\s]', '', text.lower())

        # 分词
        tokens = word_tokenize(text)

        # 移除停用词并词形还原
        processed_tokens = [
            self.lemmatizer.lemmatize(token) for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]

        return ' '.join(processed_tokens)

    def calculate_topic_consistency(self, text1, text2):
        """
        计算两个文本之间的主题一致性

        参数:
        text1, text2: 要比较的文本

        返回:
        主题一致性得分 (0-1之间)
        """
        # 预处理文本
        processed_text1 = self.preprocess_text(text1)
        processed_text2 = self.preprocess_text(text2)

        # 创建TF-IDF向量
        tfidf_matrix = self.vectorizer.fit_transform([processed_text1, processed_text2])

        # 计算余弦相似度
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        return similarity

    def calculate_multiple_documents(self, documents):
        """
        计算多个文档之间的主题一致性矩阵

        参数:
        documents: 文档列表

        返回:
        相似度矩阵
        """
        # 预处理所有文档
        processed_docs = [self.preprocess_text(doc) for doc in documents]

        # 创建TF-IDF向量
        tfidf_matrix = self.vectorizer.fit_transform(processed_docs)

        # 计算所有文档之间的余弦相似度
        similarity_matrix = cosine_similarity(tfidf_matrix)

        return similarity_matrix


# 示例使用
if __name__ == "__main__":
    # 创建计算器实例
    calculator = TopicConsistencyCalculator()

    # 示例文本
    text1 = """
    Artificial intelligence is transforming the world of technology. 
    Machine learning algorithms can now recognize patterns in data that 
    were previously invisible to humans. Deep learning models have achieved 
    remarkable success in image recognition and natural language processing.
    """

    text2 = """
    The field of machine intelligence has made significant advances in recent years. 
    Neural networks and deep learning techniques are enabling computers to perform 
    tasks that once required human cognition. AI systems can now understand and 
    generate human language with impressive accuracy.
    """

    text3 = """
    Climate change is one of the most pressing issues facing our planet. 
    Rising global temperatures are causing extreme weather events and 
    threatening ecosystems worldwide. Reducing carbon emissions is essential 
    to mitigate the effects of global warming.
    """

    # 计算两个文本之间的主题一致性
    similarity_ai = calculator.calculate_topic_consistency(text1, text2)
    print(f"AI相关文本之间的主题一致性: {similarity_ai:.4f}")

    similarity_climate = calculator.calculate_topic_consistency(text1, text3)
    print(f"AI与气候变化文本之间的主题一致性: {similarity_climate:.4f}")

    # 计算多个文档的相似度矩阵
    documents = [text1, text2, text3]
    similarity_matrix = calculator.calculate_multiple_documents(documents)

    print("\n文档相似度矩阵:")
    for i, row in enumerate(similarity_matrix):
        print(f"文档{i + 1}: {row}")