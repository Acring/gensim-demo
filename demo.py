from gensim import models
import logging
import os

"""
word2vec模块简单应用
"""


class CorpusIterator(object):
    """
    语料库迭代器
    打开语料库并逐行返回数据给Word2Vec
    """
    def __init__(self, path):
        """
        :param path: 文件地址
        """
        self.path = path

    def __iter__(self):
        """
        迭代器
        :return: 返回每行的分词
        """
        for line in open(self.path, mode="r", encoding="utf-8"):
            yield line.split(" ")  # 返回一个词语数组


class Manager(object):
    """

    """
    min_count = 100  # 最小记数
    size = 100  # 向量维度
    window = 5  # 窗口大小
    model = None

    def __init__(self, corpus_name, result_name, retrain=True):
        """
        :param corpus_name: 语料库地址
        :param result_name: 训练模型地址
        :param retrain: 是否允许多次训练
        """
        self.corpus_name = corpus_name
        self.result_name = result_name
        self.retrain = retrain

    def load_model(self, name=None):
        """
        加载模型
        :return:
        """
        if name:
            self.result_name = name

        try:
            self.model = models.Word2Vec.load(self.result_name)
        except Exception as e:
            logging.error("加载模型失败:", e)
            return

    def train_model(self):
        """
        训练模型
        :return:
        """
        sentence = CorpusIterator(self.corpus_name)

        if not os.path.exists(self.result_name):
            model = models.Word2Vec(sentence, min_count=self.min_count, size=self.size, window=self.window)
        elif self.retrain:
            logging.info("模型已存在, 再次训练")
            model = models.Word2Vec.load(self.result_name)
        else:
            logging.error("模型存在,禁止再次训练")
            return

        model.train(sentence, total_examples=model.corpus_count, epochs=model.iter)
        model.save(self.result_name)

        self.model = model

    def test_model(self, function):
        """
        测试模型
        :param function 测试函数(model)传入训练模型
        :return:
        """

        if not self.model:
            logging.ERROR("模型未加载或未训练")
            return
        if not hasattr(function, "__call__"):
            logging.ERROR("未传入测试函数")
        function(self.model)


def test_1(model):
    t1, t2 = 'man', 'woman'  # 测试两个词的相似度
    print(t1+"和"+t2+"的相似度为:"+str(model.wv.similarity(t1, t2)))

    t3, t4 = 'china', 'taiwan'
    print(t3+"和"+t4+"的相似度为:"+str(model.wv.similarity(t3, t4)))

    t5 = 'like'  # 测试和某个词最相近的词
    print("和"+t5+"相似度高的词语有:")
    for vac in model.wv.most_similar(t5):
        print(vac[0], end=" ")
    print('\n')

    t6 = ['breakfast', 'lunch', 'dinner', 'cat']  # 找出最不相似的词语
    print("在下列词语中最无关的是", end="\n")
    for vac in t6:
        print(vac, end=" ")
    print(" -> " + model.wv.doesnt_match(t6))


def test_2(model):
    t1, t2 = '干净', '整洁'  # 测试两个词的相似度
    print(t1+"和"+t2+"的相似度为:"+str(model.wv.similarity(t1, t2)))

    t3, t4 = '中国', '日本'
    print(t3+"和"+t4+"的相似度为:"+str(model.wv.similarity(t3, t4)))

    t5 = '喜欢'  # 测试和某个词最相近的词
    print("和`"+t5+"`相似度高的词语有:")
    for vac in model.wv.most_similar(t5):
        print(vac[0], end=" ")
    print('\n')

    t6 = ['喜欢', '害怕', '可爱', '汽车']  # 找出最不相似的词语
    print("在下列词语中最无关的是", end="\n")
    for vac in t6:
        print(vac, end=" ")
    print(" -> " + model.wv.doesnt_match(t6))


if __name__ == '__main__':

    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)  # 显示INFO等级以上日志

    manager = Manager(corpus_name="wiki_chinese_preprocessed.simplied.txt", result_name="train.model", retrain=False)

    manager.load_model()

    manager.test_model(test_2)


