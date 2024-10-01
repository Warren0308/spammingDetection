import time
from threshold import g_config
from datetime import datetime
import json
import os
from collections import defaultdict
try:
    import torch
    from FlagEmbedding import BGEM3FlagModel
    os.system("pip install -U FlagEmbedding")
except:

    os.system("pip install -i https://mirrors.cloud.tencent.com/pypi/simple --trusted-host mirrors.cloud.tencent.com pypi.org torch")
    os.system("pip install -i https://mirrors.cloud.tencent.com/pypi/simple --trusted-host mirrors.cloud.tencent.com pypi.org FlagEmbedding")
    os.system("pip install -i https://mirrors.cloud.tencent.com/pypi/simple --trusted-host mirrors.cloud.tencent.com pypi.org peft")
    import torch
    import peft
    from FlagEmbedding import BGEM3FlagModel


class InferenceService(object):
    def __init__(self, model_path=None):
        #self.model = torch.load('./bge_m3_complete_model.pth', map_location=torch.device('cpu'))
        self.model = BGEM3FlagModel('./bge_m3',  use_fp16=True)
        # 使用字典存储每个open_id在时间窗口内的聊天信息
        self.textDict = defaultdict(dict)
        # 使用集合来存储黑名单词，以提高查找速度
        self.black_dict = set()
        self.load_blacklist("blackFile.txt")
    """获取对应scene id的资料
    {
        open_id: {
            scene_id: {
                flag: {
                    message: {
                        "vector": {},
                        "windows": {
                            time1: {"start_time": None, "timestamps": []},
                            time2: {"start_time": None, "timestamps": []},
                            time3: {"start_time": None, "timestamps": []},
                        }
                    },
                    message:{
                    }
                },
                flag:{
                }
            },
            scene_id:{
            }
        }
    }
    """

    def load_blacklist(self, blacklist_file):
        """加载黑名单词典"""
        try:
            with open(blacklist_file, 'r', encoding='utf-8') as file:
                self.black_dict = {line.strip() for line in file if line.strip()}
        except FileNotFoundError:
            return False

    """加入黑库"""
    def save_to_blacklist(self, message):
        """将新消息追加到文件中并重新加载黑名单"""
        self.black_dict.add(message)
        try:
            with open("blackFile.txt", 'a', encoding='utf-8') as file:
                file.write(message + '\n')
        except IOError as e:
            return False

    """消除旧文本，加入新文本"""
    def clean_expired_data(self, open_id, scene_id, flag, similarity_message, message_time, duration):
        flag_data = self.textDict[open_id][scene_id][flag][similarity_message]["windows"][duration].copy()
        expired_time = message_time - duration
        valid_times = [t for t in
                       self.textDict[open_id][scene_id][flag][similarity_message]["windows"][duration]["timestamps"] if
                       t >= expired_time]
        if valid_times:
            start_time = min(valid_times) if min(valid_times) < message_time else 0
            flag_data["start_time"] = start_time
            flag_data["timestamps"] = valid_times
        else:
            flag_data["start_time"] = 0
            flag_data["timestamps"] = []
        self.textDict[open_id][scene_id][flag][similarity_message]["windows"][duration] = flag_data
        return True

    """根据文本长度分类文本"""
    def len_calcul(self, text):
        l_content = len(text)
        length_thresholds = [(6, "first"), (8, "second"), (10, "third"), (12, "fourth"),
                             (16, "fifth"), (20, "sixth"), (24, "seventh"), (28, "eighth"),
                             (32, "ninth"), (float('inf'), "tenth")]
        for threshold, flag in length_thresholds:
            if l_content <= threshold:
                return flag
        return False

    """处理每个open_id的数据"""
    def vectorChecking(self, open_id, scene_id, flag, message_time, vector, message):
        messages = self.textDict[open_id][scene_id][flag]
        similarities = [
            (msg, messages[msg]["vector"] @ vector.T) for msg in messages
        ]
        most_similar_message, highest_similarity = max(similarities, key=lambda x: x[1], default=(None, 0))
        return highest_similarity < 0.8, most_similar_message

    """文本向量化"""
    def message_vectorization(self, message):
        return self.model.encode(message, batch_size=15, max_length=50)['dense_vecs']

    """时间置换概念"""
    def time_management(self, open_id, scene_id, flag, similarity_message, message_time):
        for key, value in self.textDict[open_id][scene_id][flag][similarity_message]["windows"].items():
            start_time = value["start_time"]
            chat_duration = message_time - start_time
            if chat_duration < 0:
                return False
            elif chat_duration > key:
                self.clean_expired_data(open_id, scene_id, flag, similarity_message, message_time, key)
        return True

    def process_request(self, request, duration=None, duration2=None):

        start_time = time.time()
        """
        :param request:open_id&*#message&*#datetime(for test)
        :return: dict(message)
        """
        error = {0: "success"}
        result = {
            "open_id": "",  # 账号
            "scene_id": 0,  # 场景
            "spammingTime":{
                # duration:{ # 时段
                #     "no_similarity": 0,  # 相似文本的次数
                #     "threshold": 0,  # 阈值
                # },
            },
            "content": "",  # 文本内容
            "error": ""  # 出现问题才会有
        }

        try:
            request = request.decode("utf-8", "ignore")
        except:
            request = request

        # 空数据或数据不足不处理
        if request == "" or len(request.split('&*#')) < 3:
            result["error"] = "请求值有问题。"
            result = json.dumps(result, ensure_ascii=False)
            return result, error

        # 数据分析
        # 账号 ｜ 内容
        data = request.strip().split("&*#")
        open_id, message, scene_id = data[0], data[1], data[2]
        result["open_id"] = open_id
        result["scene_id"] = scene_id
        result["content"] = message
        message_time = datetime.now().timestamp()
        # 测试包含时间(testing)
        if len(request.split("&*#")) > 3:
            message_time = request.strip().split("&*#")[-1]
            message_time = datetime.strptime(message_time, '%Y-%m-%d %H:%M:%S').timestamp()

        # 排白文本不处理
        if message.strip() == "":
            result["error"] = "文本是空值。"
            result = json.dumps(result, ensure_ascii=False)
            return result, error

        # 未配置场景不处理
        scene_data = g_config.get(int(scene_id), {})
        if not scene_data:
            result["error"] = "场景配置错误。"
            result = json.dumps(result, ensure_ascii=False)
            return result, error

        # 文本长度设定
        flag = self.len_calcul(message)
        if not flag:
            result["error"] = "文本长度检测出现问题。"
            result = json.dumps(result, ensure_ascii=False)
            return result, error

        # 文本向量化设定
        vector = self.message_vectorization(message)
        if vector is False:
            result["error"] = "文本向量化出现问题。"
            result = json.dumps(result, ensure_ascii=False)
            return result, error

        # 检查黑库是否有类似文本
        if message.strip().lower() in self.black_dict:
            duration, threshold = next(iter(g_config.get(-1).items()))
            result["spammingTime"]["BlackWords"] = {
                "no_similarity": 36,
                "threshold": threshold
            }
            result["content"] = message
            result = json.dumps(result, ensure_ascii=False)
            return result, error

        # 第一个数据所以不需要进行相似度分析（账号，场景和文本长度进行对比了解有没有这个数据）
        if not self.textDict.get(open_id, {}).get(scene_id, {}).get(flag, {}):
            self.textDict.setdefault(open_id, {}).setdefault(scene_id, {}).setdefault(flag, {})
            self.textDict[open_id][scene_id][flag][message] = {
                'vector': vector,
                'windows': {}
            }
            # 把场景内设定的时段放入dict中(并根据设定时段输出结果）
            for duration, threshold in scene_data.items():
                self.textDict[open_id][scene_id][flag][message]['windows'][duration] = {
                    "start_time": message_time,
                    "timestamps": [message_time]
                }

            # 获取第一个数据
            duration, threshold = next(iter(scene_data.items()))
            duration = str(int(duration / 60)) + "分钟"
            result["spammingTime"][duration] = {
                "no_similarity": 0,  # Similar text count
                "threshold": threshold  # Threshold
            }
            result["content"] = message
            result = json.dumps(result, ensure_ascii=False)
            return result, error

        # 检查向量相似度
        new, similarity_message = self.vectorChecking(open_id, scene_id, flag, message_time, vector, message)
        # 新文本就直接输出新的结果
        if new:
            self.textDict[open_id][scene_id][flag][message] = {
                'vector': vector,
                'windows': {}
            }
            # 把场景内设定的时段放入dict中(并根据设定时段输出结果）
            for duration, threshold in scene_data.items():
                self.textDict[open_id][scene_id][flag][message]['windows'][duration] = {
                    "start_time": message_time,
                    "timestamps": [message_time]
                }
            # 获取第一个数据
            duration, threshold = next(iter(scene_data.items()))
            duration = str(int(duration / 60)) + "分钟"
            result["spammingTime"][duration] = {
                "no_similarity": 1,  # Similar text count
                "threshold": threshold  # Threshold
            }
            result["content"] = message
            result = json.dumps(result, ensure_ascii=False)
            # 记录结束时间
            end_time = time.time()

            # 计算执行时间
            execution_time = end_time - start_time

            print(f"代码执行时间：{execution_time} 秒")
            return result, error
        else:
            # 清除旧数据
            time_data = self.time_management(open_id, scene_id, flag, similarity_message, message_time)
            if not time_data:
                result["error"] = "时间窗口出现问题。"
                result = json.dumps(result, ensure_ascii=False)
                return result, error
            # 添加新数据 & 数据返回成功
            for duration, threshold in scene_data.items():

                self.textDict[open_id][scene_id][flag][similarity_message]['windows'][duration]["timestamps"].append(
                    message_time)
                no_similarity = len(
                    self.textDict[open_id][scene_id][flag][similarity_message]['windows'][duration]["timestamps"])
                duration = str(int(duration / 60)) + "分钟"
                result["spammingTime"][duration] = {
                    "no_similarity": no_similarity,  # Similar text count
                    "threshold": threshold  # Threshold
                }
            duration = next(reversed(scene_data))
            if len(self.textDict[open_id][scene_id][flag][similarity_message]['windows'][duration]["timestamps"]) >= next(iter(g_config.get(-1).items()))[1]:
                self.save_to_blacklist(message.strip().lower())
            result["content"] = message
            result = json.dumps(result, ensure_ascii=False)
            # 记录结束时间
            end_time = time.time()

            # 计算执行时间
            execution_time = end_time - start_time

            print(f"代码执行时间：{execution_time} 秒")
            return result, error


if __name__ == '__main__':
    # 使用示例
    model = InferenceService()

    print("第1分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:00:00"))
    print("第2分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:01:00"))
    print("第3分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:02:00"))
    print("第4分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:03:00"))
    print("第5分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:04:00"))
    print("第6分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:05:00"))
    print("第7分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:06:00"))
    print("第8分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:07:00"))
    print("第9分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:08:00"))
    print("第10分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:09:00"))
    print("第11分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:10:00"))
    print("第12分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:11:00"))
    print("第13分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:12:00"))
    print("第14分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:13:00"))
    print("第15分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:14:00"))
    print("第16分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:15:00"))
    print("第17分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:16:00"))
    print("第18分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:17:00"))
    print("第19分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:18:00"))
    print("第20分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:19:00"))
    print("第21分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:20:00"))
    print("第22分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:21:00"))
    print("第23分钟: ", model.process_request(
        "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:22:00"))
    print("第24分钟: ", model.process_request(
        "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:23:00"))
    print("第25分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:24:00"))
    print("第30分钟: ",
          model.process_request("28C073899E443002B6546B62BCC802&*#外挂群加加加32482933&*#2&*#2024-01-22 17:30:00"))
    print("第31分钟: ", model.process_request("28C073899E443002B6546B62BCC802&*#我叫黄胜衍&*#2&*#2024-01-22 17:31:00"))
    print("第32分钟: ",
          model.process_request("28C073899E443002B6546B62BCC802&*#外挂群加加加32482933&*#2&*#2024-01-22 17:32:00"))
    print("第33分钟: ",
          model.process_request("28C073899E443002B6546B62BCC802&*#外挂群加加加32482933&*#2&*#2024-01-22 17:33:00"))
    print("第36分钟: ",
          model.process_request("28C073899E443002B6546B62BCC802&*#外挂qun加加加32482935&*#2&*#2024-01-22 17:36:00"))
    print("第66分钟: ",
          model.process_request("28C073899E443002B6546B62BCC802&*#外挂qun加加加32482935&*#2&*#2024-01-22 18:06:00"))
    print("第67分钟: ",
          model.process_request("28C073899E443002B6546B62BCC802&*#外挂qun加加加32482935&*#2&*#2024-01-22 18:07:00"))
    print("第80分钟: ", model.process_request("28C073899E443002B6546B62BCC802&*#我叫黄胜衍&*#2&*#2024-01-22 18:20:00"))
    print("第120分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 19:24:00"))
