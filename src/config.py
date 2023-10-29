import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

img_size = 224
input_channel = 3
output_channel = 2
curvature = 0.005

img_list_file = "../GCN_edge_test/trainfnamelist.npy"
img_path = "../GCN_edge_test/DUTS_TR/"
epoch = 200
lr = 5e-6
batch_size = 2
aggNum = 16