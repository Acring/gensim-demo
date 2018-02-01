from gensim import corpora


def save_load(filename):
    """
    简单的保存和加载
    :return:
    """

    documents = []
    # 打开原始文本
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            documents.append(line.split(" "))

    # 生成词典
    dictionary = corpora.Dictionary(documents)

    print(dictionary)
    # 将词典保存到本地文件
    dictionary.save("tmp\\dictionary.dic")

    # 加载词典
    dictionary2 = corpora.Dictionary.load("tmp\\dictionary.dic")
    print(dictionary2)

    # 生成词库
    corpus = [dictionary.doc2bow(document) for document in documents]
    print(len(corpus))

    # 保存词库
    corpora.MmCorpus.serialize("tmp\\corpus.m", corpus)

    # 加载词库
    corpus2 = corpora.MmCorpus('tmp\\corpus.m')
    print(len(corpus2))

    return dictionary, corpus


def use_dic_cor(dictionary, corpus):
    """
    dictionary的简单使用
    :return:
    """
    # 获取某个词的id
    print(dictionary.token2id["中国"])

    # 用id获取某个词
    print(dictionary[46])

    # 和上面的效果相同
    print(dictionary.id2token[46])

    # 用id获取几个文档中出现过这个词
    print(dictionary.dfs[46])

    # 按词出现文档的次个数对词进行排序
    max_tuple = sorted(dictionary.dfs.items(), key=lambda x: x[1], reverse=True)
    # 出现次数第10个词语的id和出现次数
    print(max_tuple[10])
    # 出现次数第10个词语的实际字符表示
    print(dictionary[max_tuple[10][0]])

    # 新的文章按照词典转换成corpus的向量表示形式
    new_doc = "我 爱 中国"
    new_vec = dictionary.doc2bow(new_doc.split())
    print(new_vec)

    # 新的文章按照词典转换成下标的形式
    new_doc = "我 爱 中国"
    new_vec = dictionary.doc2idx(new_doc.split())
    print(new_vec)


def merge_dic(dic_1, dic_2):
    """
    合并两个词典
    :param dic_1:
    :param dic_2:
    :return:
    """
    dic2_to_dic1 = dic_1.merge_with(dic_2)
    print(dic2_to_dic1)

def main():
    filename = "wiki_chinese_preprocessed.simplied_1.txt"
    dictionary, corpus = save_load(filename)
    use_dic_cor(dictionary, corpus)
    filename = "wiki_chinese_preprocessed.simplied_2.txt"
    dictionary2, corpus2 = save_load(filename)
    merge_dic(dictionary, dictionary2)


if __name__ == '__main__':
    main()