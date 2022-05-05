import torch
from models.pointnet2_cls import pointnet2_cls_ssg

def load_assessor(pretrained_file_path, dims=6, nclasses=40):
    device = torch.device('cuda')
    model = pointnet2_cls_ssg(dims, nclasses)
    model = model.to(device)
    model.load_state_dict(torch.load(pretrained_file_path))
    model.eval()
    return model