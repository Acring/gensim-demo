import logging
from gensim import corpora, models, similarities

"""
gensim的其他模块

tf-idf 词频-逆文档频率 计算一个文档和其他文档的相似度

"""


class GensimTest(object):

    corpus = None
    dictionary = None

    def __init__(self, documents=None, filename=None):
        if documents:
            texts = [[word for word in document.lower().split()] for document in documents]

            # 词典
            dictionary = corpora.Dictionary(texts)
            # 词库，以(词，词频)方式存贮
            corpus = [dictionary.doc2bow(text) for text in texts]
            print("词典:", dictionary)
            print("词库:", corpus)
            self.dictionary = dictionary
            self.corpus = corpus

    def get_vec(self, document):
        """
        根据当前的语料库将文章生成新的向量表示
        :param document:
        :return:
        """
        text = [word for word in document.lower().split()]

        # 生成文档集
        vec = self.dictionary.doc2bow(text)
        print(vec)
        return vec

    def tf_idf(self, doc_bow):
        """
        根据此类的文档集来检测文档相似度
        :param doc_bow:
        :return:
        """
        # initialize a model
        tfidf = models.TfidfModel(dictionary=self.dictionary)  # 初始化tf-idf模型, corpus 作为语料库

        # 使用tfidf模型将自身的词库转换成tf-idf表示
        corpus_tfidf = tfidf[self.corpus]
        for doc in corpus_tfidf:
            print(doc)

        # 使用模型tfidf，将doc_bow(由词,词频)表示转换成(词,tfidf)表示
        print(tfidf[doc_bow])

        # 检查和每个文档的相似度
        index = similarities.SparseMatrixSimilarity(tfidf[self.corpus], num_features=len(self.dictionary))
        sims = index[doc_bow]

        print(sims)


def tfidf_test():
    """
    测试tf-idf模型
    :return:
    """
    documents = ["Human machine interface for lab abc computer applications applications",
                 "A survey of user opinion of computer system response time",
                 "The EPS user interface management system",
                 "System and human system engineering testing of EPS",
                 "Relation of user perceived response time to error measurement",
                 "The generation of random binary unordered trees",
                 "The intersection graph of paths in trees",
                 "Graph minors IV Widths of trees and well quasi ordering",
                 "Graph minors A survey"]

    test_document = "Human like machine while interface is survey abc"

    manager = GensimTest(documents=documents)

    test_vec = manager.get_vec(test_document)

    manager.tf_idf(test_vec)

if __name__ == '__main__':
    # logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)  # 显示INFO等级以上日志

    tfidf_test()

