import torch
import params
import argparse
from model import net_G

def load_generator(pretrained_file_path_G):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    G = net_G(args)
    if not torch.cuda.is_available():
        G.load_state_dict(torch.load(pretrained_file_path_G, map_location={'cuda:0': 'cpu'}))
    else:
        G.load_state_dict(torch.load(pretrained_file_path_G))

    G.to(params.device)
    G.eval()
    return G
