from collections import defaultdict
import faiss
from FlagEmbedding import BGEM3FlagModel

class InferenceService(object):
    def __init__(self, model_path=None):
        # self.model = torch.load('./bge_m3_complete_model.pth', map_location=torch.device('cpu'))
        self.model = BGEM3FlagModel('./bge_m3', use_fp16=True)
        self.textDict = defaultdict(dict)

    def vectorChecking(self, personal_data, vector):
        similarities = [
            (msg, personal_data[msg]["vector"] @ vector.T) for msg in personal_data
        ]
        most_similar_message, highest_similarity = max(similarities, key=lambda x: x[1], default=(None, 0))
        return highest_similarity < 0.8, most_similar_message

    def message_vectorization(self, message):
        return self.model.encode(message, batch_size=15, max_length=50)['dense_vecs']

    def len_calcul(self, text):
        l_content = len(text)
        length_thresholds = [(6, "first"), (8, "second"), (10, "third"), (12, "fourth"),
                             (16, "fifth"), (20, "sixth"), (24, "seventh"), (28, "eighth"),
                             (32, "ninth"), (float('inf'), "tenth")]
        for threshold, flag in length_thresholds:
            if l_content <= threshold:
                return flag
        return False

    def process_request(self,request):
        data = request.strip().split("&*#")
        open_id, message, scene_id = data[0], data[1], data[2]
        flag = self.len_calcul(message)
        vector = self.message_vectorization(message)
        personal_data = self.textDict[open_id][scene_id][flag]
        new, similarity_message = self.vectorChecking(personal_data, vector)

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