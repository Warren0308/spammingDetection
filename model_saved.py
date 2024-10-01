from FlagEmbedding import BGEM3FlagModel
import torch

model = BGEM3FlagModel('BAAI/bge-m3',use_fp16=True)
torch.save(model,'bge_m3_complete_model1.pth')