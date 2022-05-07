import os
import sys
import torch
from torch import optim
from torch import nn
import numpy as np
from torch._C import device
from torch.utils.data import DataLoader, TensorDataset
from generator import load_generator
from assessor import load_assessor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# np.set_printoptions(threshold=10)
def walk(z, G_path, Cls_path, iterations=1000, delta=0.1):
    #print('input z:', z)
    G = load_generator(G_path)
    Cls = load_assessor(Cls_path)
    m = nn.Softmax(dim=1)
    m.eval()
    # z_dim = 200
    # z = torch.zeros(1, z_dim)
    z = z.to(torch.device('cuda'))

    
    target_score = torch.tensor(1.0).to("cuda")
    treshold = 0.95 #torch.tensor(0.95).to("cuda")


    def f(z):
        #for i in range(10):
        voxels = generator(z, G)
        # print(voxels.shape)
        points = convert(voxels)
        # print(points.shape)
        
        out_score = classify(points, Cls, m)
          #print(out_score)
        return out_score

    ct = 0
    while True:
        torch.seed()
        z = torch.randn(1, 200).to(torch.device('cuda'))
        out_score = f(z).detach().to('cpu').numpy()
        print(out_score)
        
        if out_score > 0.95:
            visualize(ct, z, G)
            ct += 1
    return

    # for iteration in range(iterations):
        
    #     prevy = f(z).detach().to('cpu').numpy()
    #     print('current score:', prevy)
    #     visualize(iteration, z, G)

    #     mx = 0.
    #     transform = -1
    #     for dim in range(z.shape[1]):
    #         newz = torch.clone(z)
    #         newz[0,dim] += delta
    #         newy = f(newz).detach().to('cpu').numpy()
    #         if newy-prevy > mx:
    #             #print(newy)
    #             mx = newy-prevy
    #             transform = dim
    #     if newy < treshold and transform != -1:
    #         z[0,transform] += delta
    #         print("iteration {} z: {}".format(iteration, z))
            
    #     else:
    #         break
    print('output z:', z)
    print('output score:', f(z).detach().to('cpu').numpy())

def process(z, G_path, Cls_path, iterations=1000, lr=0.1):

    G = load_generator(G_path)
    Cls = load_assessor(Cls_path)
    m = nn.Softmax(dim=1)
    # z_dim = 200
    # z = torch.zeros(1, z_dim)
    z = z.to(torch.device('cuda'))
    z.requires_grad = True
    
    target_score = torch.tensor(1.0).to("cuda")
    treshold = torch.tensor(0.95).to("cuda")

    optimizer = optim.Adam([z], lr)
    loss = nn.MSELoss()

    for iteration in range(iterations):
    
        optimizer.zero_grad()

        voxels = generator(z, G)
        
        # print(voxels.shape)
        points = convert(voxels)
        out_score = torch.sum(points)/10000
        # print(points.shape)
        #out_score = classify(points, Cls, m)
    
        print('iteration {} out_score:{}'.format(iteration, out_score))

        if out_score >= treshold:
            break

        loss_value = loss(out_score, target_score)

        print('iteration {} loss:{}'.format(iteration, loss_value))

        z.retain_grad()
        loss_value.backward()
        optimizer.step()
        
        # print(z.grad)

    print("iteration:{}".format(iteration))
    print("out:{}".format(z))
    

def interpolate(source, target, G, segs=100):
    diff = target - source
    for i in range(segs):
        visualize(i, source + (i/segs)*diff, G)
    

def visualize(index, z, G):
    voxels = generator(z, G).to('cpu').numpy()
    
    fig = plt.figure(figsize=(16,16))
    x, y, z = voxels.nonzero()
    ax = fig.gca(projection="3d")
    ax.axes.set_xlim3d(left=0.2, right=63.8) 
    ax.axes.set_ylim3d(bottom=0.2, top=63.8) 
    ax.axes.set_zlim3d(bottom=0.2, top=63.8) 
    ax.scatter(x, y, z, zdir='z', c='red')
    # os.system('mkdir visualize')
    plt.savefig('visualize/{}.png'.format(index), bbox_inches='tight')
    plt.close()

    

# setup the generator, mapping z -> voxels
def generator(z, G):
    
    fake = G(z)
    samples = fake.unsqueeze(dim=0)#.detach().cpu().numpy()
    voxels = samples[0].__ge__(0.5)
    # voxels = torch.where(samples[0]>=0.5, 1., 0.)
    # diff = (voxels - samples[0]).detach()
    # voxels = samples[0] + diff

    return voxels

# mapping voxels -> points
def convert(voxels, cube_len=64):
    vsize = 1/cube_len
    res = torch.argwhere(voxels)*vsize
    return res

# points -> outscore
def classify(xyz_points, model, m, npoints=1024):
    #print(xyz_points)
    # device = torch.device('cuda')
    #inds = np.random.randint(0, len(xyz_points), size=(npoints, ))
    xyz_points = xyz_points[:npoints, :]
    # normalize
    # xyz_points[:, :3] = pc_normalize(xyz_points[:, :3])
    data = [xyz_points]
    data = torch.stack(data)
    #test_loader = DataLoader(dataset=TensorDataset(data),
    #                         batch_size=1, shuffle=False,
    #                         num_workers=1)
   
    #for batch in test_loader:
    #xyz = data[:, :, :3]
    points = torch.zeros_like(data) # do not use normals
    
    with torch.no_grad():

        torch.manual_seed(0)
        pred = model(data, points).detach()
        # print(torch.argmax(pred))
        pred = m(pred)
        # print('pred:', pred[0,26])
        
    return pred[0,26]
    

def main():
    args = sys.argv[1:]
    G_path = 'G.pth' #args[0]
    Cls_path = 'Cls.pth' #args[1]
    z_dim = 200
    z = torch.randn(1, z_dim)
    #process(z, G_path, Cls_path)
    walk(z, G_path, Cls_path)

if __name__ == '__main__':
    main()