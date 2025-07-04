# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
