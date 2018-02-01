import os
"""
将测试语料库wiki_chinese_preprocessed.simplied.txt截取一部分处理

"""


def main():
    mainfile = "wiki_chinese_preprocessed.simplied.txt"
    subfile = "wiki_chinese_preprocessed.simplied_2.txt"
    start = 100
    end = 200
    p = 0
    with open(mainfile, "r", encoding="utf-8") as f:
        with open(subfile, "w", encoding="utf-8") as w:
            for line in f:
                if start <= p <= end:
                    w.write(line)
                p += 1

if __name__ == '__main__':
    main()