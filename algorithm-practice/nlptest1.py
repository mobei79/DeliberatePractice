# -*- coding: utf-8 -*-
"""
@Time     :2021/11/4 16:40
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import os
from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller
# pip install pyltp -i https://pypi.tuna.tsinghua.edu.cn/simple 可以先下载好whl文件
#LTP语言平台：http://ltp.ai/index.html
#咱们使用的工具包,pyltp:https://pyltp.readthedocs.io/zh_CN/latest/api.html
#LTP附录：https://ltp.readthedocs.io/zh_CN/latest/appendix.html#id3
#安装方法：https://github.com/HIT-SCIR/pyltp
class LtpParser:
    def __init__(self):
        LTP_DIR = "./ltp_data_v3.4.0"
        self.segmentor = Segmentor()    # 分词
        self.segmentor.load(os.path.join(LTP_DIR, "cws.model"))

        self.postagger = Postagger()    # 词性标注
        self.postagger.load(os.path.join(LTP_DIR, "pos.model"))

        self.parser = Parser()  # 句法依存分析
        self.parser.load(os.path.join(LTP_DIR, "parser.model"))

        self.recognizer = NamedEntityRecognizer()   # 命名实体识别
        self.recognizer.load(os.path.join(LTP_DIR, "ner.model"))

        self.labeller = SementicRoleLabeller()  # 语义角色标注
        self.labeller.load(os.path.join(LTP_DIR, 'pisrl_win.model'))

    # 语义角色标注
    def format_labelrole(self, words, postags):
        print("分词----> words= {0}----len(words) = {1}".format(words, len(words)))
        print("词性标注----> postags= {0}----len(postags) = {1}".format(postags, len(postags)))
        arcs = self.parser.parse(words, postags)  # 建立依存句法分析树
        roles = self.labeller.label(words, postags, arcs)
        print("len(roles) = {0}----roles = {1}".format(len(roles), roles))
        roles_dict = {}
        for role in roles:
            print("谓语所在索引：role.index = {0}".format(role.index))
            roles_dict[role.index] = {arg.name:[arg.name,arg.range.start, arg.range.end] for arg in role.arguments}
        # {6: {'A0': ['A0', 0, 2], 'TMP': ['TMP', 3, 3], 'LOC': ['LOC', 4, 5], 'A1': ['A1', 8, 8]}}
        # 6：表示谓语（发表）所在序号；
        # A0：表示“施事者、主体、触发者”，0,2分别表示A0所在的起始索引、终止索引（此句中有2个A0，分别是“奥巴马”、“克林顿”，索引范围是是0-2）
        # TMP：表示“时间”，3, 3分别表示TMP所在的起始索引、终止索引（“昨晚”）
        # LOC：表示“地点”，4, 5分别表示LOC所在的起始索引、终止索引（“在”,“白宫”）
        # A1：表示“受事者”，8, 8分别表示LOC所在的起始索引、终止索引（“演说”）
        print("语义角色标注---->roles_dict = {0}".format(roles_dict))
        return roles_dict

    '''parser主函数'''
    def parser_main(self, sentence):
        # 分词
        words = list(self.segmentor.segment(sentence))
        # 词性标注
        postags = list(self.postagger.postag(words))
        # 语义角色标注
        roles_dict = self.format_labelrole(words, postags)

        return words, postags, roles_dict


if __name__ == '__main__':
    parse = LtpParser()
    sentence = '奥巴马与克林顿昨晚在白宫发表了演说'
    words, postags, roles_dict = parse.parser_main(sentence)
