import Levenshtein
import re

"""提纯数据"""


def findchinese(s):
    re_words = re.compile(u"[\u4e00-\u9fa5]+")
    res = re.findall(re_words, s)
    return ''.join(res)


def finddigtal(s):
    re_words = re.compile(u"[\d]+")
    res = re.findall(re_words, s)
    return ''.join(res)


def findenglish(s):
    re_words = re.compile(u"[\u0041-\u005a|\u0061-\u007a]+")
    res = re.findall(re_words, s)
    return ''.join(res)


"""公共子串相似度计算"""


def get_commonSubstring_sim(str1_l, str2_l):
    temp = 0
    for seq in str1_l:
        if seq in str2_l:
            temp = temp + 1
    return temp / ((len(str1_l) + len(str2_l)) / 2)


"""编辑距离相似度计算"""


def get_levenshtein_sim(str1, str2):
    res = Levenshtein.distance(str1, str2)
    mean_value = (len(str1) + len(str2)) / 2
    return 1 - (res / mean_value)


"""杰卡德相似度计算"""


def get_jaccard_sim(str1, str2):
    a = set(str1)
    b = set(str2)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


"""获取相似文本"""


def get_sim_content(talk_dict: dict, line: str):
    """
    :param keys: 文本key
    :param line: 输入文本
    :return: 相似文本，相似值
    """
    sim_value = -1
    sim_content = line
    desc = "self"
    # str1 = line.strip().split("_")  拼音文本
    str1 = line.strip()
    # str1_sub = [str1[i:i+2] for i in range(len(str1)-1)]
    for key in talk_dict.keys():
        # str2 = key.strip().split("_")  拼音文本
        str2 = key.strip()
        jaccard_sim_value = get_jaccard_sim(str1, str2)
        levensh_sim_value = 1 - get_levenshtein_sim(line, key)
        if jaccard_sim_value > sim_value:
            sim_content = key
            sim_value = jaccard_sim_value
            desc = "jac"
        if levensh_sim_value > sim_value:
            sim_content = key
            sim_value = levensh_sim_value
            desc = "lev"
    return sim_content, sim_value, desc


"""数字还原函数"""


def sub_num(text, num_project: dict):
    rx = re.compile('|'.join(map(re.escape, num_project)))

    def one_xlat(match):
        return num_project[match.group(0)]

    return rx.subn(one_xlat, text)  # 每遇到一次匹配就会调用回调函数


"""可疑联系方式检测"""


def sus_contact_detect(text, num_project: dict):
    (newtext, subNum) = sub_num(text, num_project)
    num_len = 0
    # only_num = finddigtal(newtext)
    pattern = re.compile('[\d]+')
    only_num = pattern.findall(newtext)
    if len(only_num) != 0:
        num_len = len(max(only_num, key=len))

    return newtext, num_len


# 以字为最小单位计算句子集合的重合度:（每句公共相交之和）/总字数
def naive_sim_char_2(message_list):
    word_pool = []
    for message in message_list:
        for word in message:
            word_pool.append(word)
    word_pool_length = len(word_pool) * 1.0
    if len(word_pool) == 0:
        return 0.0
    sim_score = 1 - len(set(word_pool)) / word_pool_length

    return sim_score


# naive_sim_char_2 改版
def naive_sim_char_3(message_dict):
    # message_dict : self.g_roletextDict[gameid][sceneid][openid]'
    L = []  # 存储单条文本
    L_N = []  # 存储该文本出现的总数
    for f in message_dict.keys():
        for c in message_dict[f].keys():
            L.append(c)
            L_N.append(c * len(message_dict[f][c]))

    if len(L) == 0:
        return 0.0

    set_word_pool_length = len(set("".join(L)))
    word_pool_length = len("".join(L_N)) * 1.0
    sim_score = 1 - set_word_pool_length / word_pool_length

    return sim_score


"""re匹配"""


def is_find_re_object(utext, re_list):
    '''
    @函数功能：返回re_list对给定文本的匹配结果。
    @输入：utext
    @输出：re_list [[compile(text), text], ..., ...,]
    正则匹配的时候，表达式和文本编码要一致，不然就只能匹配到英文的。
    双方都用utf8编码会大大提升速度。
    '''
    for re_object, source_text in re_list:
        if re_object.search(utext):
            return True
    return False


"""re取代"""


def is_filter_re_object(utext, re_list):
    '''
    @函数功能：返回re_list对给定文本的匹配结果。
    @输入：utext
    @输出：re_list [[compile(text), text], ..., ...,]
    正则匹配的时候，表达式和文本编码要一致，不然就只能匹配到英文的。
    双方都用utf8编码会大大提升速度。
    '''
    for re_object, source_text in re_list:
        utext = re_object.sub("", utext)
    return utext


"""读档"""


def read_txt(path):
    re_list = []
    with open(path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    for line in texts:
        line = line.strip()
        if line:
            re_list.append([re.compile(line), line])
    return re_list


if __name__ == '__main__':
    str1 = "怎么了"
    str2 = "怎么了???????"
    print(str2)
    print(get_jaccard_sim(str1, str2))

    num_project = {}
    symNum_path = './sym_num.txt'
    # 载入数字还原词典
    with open(symNum_path, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            a = line.strip().split('|')[0]
            b = line.strip().split('|')[1]
            num_project[a] = b

    line = "7⃣️8⃣️o9⃣️0快来加入我们幹啥妳想幹啥想打架麽oo0⃣️"
    line = "#{_INFOSER}出17世涅槃宝宝有现货接10-25世涅槃宝宝9星定制镜子有需要的dddddd#aB[二代咩小萌]#aE"
    line = "aB[洛阳1线(77,230)]#aE#LocationExtraData0*0*1*47*0*77.092346191406*230.68518066406"
    line = "三4⃣️五六7⃣️8，一二三27398729"
    line = "...？？ ...？？"
    print(sus_contact_detect(line, num_project))

    str1 = "wo_men_zen_me_le_1621"
    str2 = "wo_menzenme_312"
    str3 = "wo_hen_kai_xin_yu_ni_zai_yi_qi"
    str4 = "wo_hen_kai_xin_yu_ni"
    print(get_levenshtein_sim(findenglish(str1), findenglish(str2)))
    print(get_jaccard_sim(findenglish(str1), findenglish(str2)))
    new_str3 = str3.split('_')
    new_str4 = str4.split("_")
    str3_l = [new_str3[i:i + 2] for i in range(len(new_str3) - 1)]
    str4_l = [new_str4[i:i + 2] for i in range(len(new_str4) - 1)]
    print(str3_l)
    print(str4_l)
    temp = 0
    for seq in str4_l:
        if seq in str3_l:
            temp = temp + 1
    print(temp / ((len(str3_l) + len(str4_l)) / 2))
