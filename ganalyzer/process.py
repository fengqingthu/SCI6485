import sys
import torch
from torch import optim
from torch import nn
import numpy as np
from generator import load_generator
from assessor import load_assessor

def process(z, G_path, Cls_path, iterations=1000, lr=0.0002):

    G = load_generator(G_path)
    Cls = load_assessor(Cls_path)

    # z_dim = 200
    # z = torch.zeros(1, z_dim)
    z.requires_grad = True

    target_score = torch.tensor(1.0)
    treshold = torch.tensor(0.95)

    optimizer = optim.Adam([z], lr)
    loss = nn.MSELoss()

    for iteration in range(iterations):
    
        optimizer.zero_grad()

        voxels = generator(z, G)
        points = convert(voxels)
        out_score = torch.tensor(classify(points, Cls))
        print('out_score: ', out_score)

        if out_score >= treshold:
            break

        loss_value = loss(out_score, target_score)

        print('loss: ', loss_value)

        loss_value.backward()
        optimizer.step()
        
        print("iteration:{}".format(iteration))
        print("out:{}".format(z))

# setup the generator, mapping z -> voxels
def generator(z, G):
    
    fake = G(z)
    samples = fake.unsqueeze(dim=0).detach().cpu().numpy()
    voxels = samples[0].__ge__(0.5)

    return voxels

# mapping voxels -> points
def convert(voxels, cube_len=64):
    vsize = 1/cube_len

    xarr = []
    yarr = []
    zarr = []
    index = 0
    for x in range(cube_len):
        for y in range(cube_len):
            for z in range(cube_len):
                if voxels[x,y,z]:
                    xarr.append(x*vsize)
                    yarr.append(y*vsize)
                    zarr.append(z*vsize)
                    index += 1
    
    xarr = np.array(xarr)
    yarr = np.array(yarr)
    zarr = np.array(zarr)
    data = np.column_stack((xarr, yarr, zarr))
    return data

# points -> outscore
def classify(points, model, npoints=1024, dims=6, nclasses=40):
    device = torch.device('cuda')
    inds = np.arange(len(xyz_points))
    np.random.shuffle(inds)
    xyz_points = xyz_points[inds, :]
    # normalize
    # xyz_points[:, :3] = pc_normalize(xyz_points[:, :3])
    data = [xyz_points]
    m = nn.Softmax(dim=1)
    xyz = data[:, :, :3]
    points = torch.zeros_like(xyz) # do not use normals
    with torch.no_grad():
        pred = model(xyz.to(device), points.to(device))
        pred = m(pred)
    return pred[0,26]

def main():
    args = sys.argv[1:]
    G_path = args[0]
    Cls_path = args[1]
    z_dim = 200
    z = torch.zeros(1, z_dim)
    process(z, G_path, Cls_path)

if __name__ == '__main__':
    main()