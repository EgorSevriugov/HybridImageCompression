# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Hybrid approach
#
# $$\min\limits_{X,Z_1,...,Z_N} \sum\limits_{n=1}^3\alpha_n\|Z_n\|_*$$
# $$s.t. \|Y-f_{\theta}(X)\|_F^2 \leq \varepsilon^2$$
# $$X_{(n)} = Z_{n}$$
#
# For this problem we can derive augmented lagrangian:
#
# $$L = \sum\limits_{n=1}^3\alpha_n\|Z_{n}\|_* + i_D(X) + \frac{\lambda}{2}\left(\|X_{(n)} - Z_n - T_n\|_F^2 - \|T_n\|^2_F \right)$$
#
# Following lecture discription we can directly write update rules for $Z_n$, $T_n$, $X_n$:
#
# $$Z_n \leftarrow \mathrm{SVT}_\frac{\alpha_n}{\lambda}(X_{(n)} - \mathrm{Fold}_n(T_n))$$
#
# $$T_n \leftarrow T_n + Z_n - X_{(n)}$$
#
# $$X \leftarrow \min\limits_{s.t. \|Y - f_\theta(X)\|_F^2 \leq \varepsilon^2} \sum\limits_{n=1}^3\frac{\lambda}{2}\|X - \mathrm{Fold}_n(Z_n + T_n)\|_F^2$$
#
# $$\theta \leftarrow \min\limits_\theta\|Y - f_\theta(X)\|_F^2$$
#
# For we have several options:
#
# **Linear** $f_\theta(X) = \mathrm{unvec}(\theta \mathrm{vec}(X))$, $\theta$ is of shape $(W * H * 3, W * H * C)$, but we can represent it in tensor format of size $(W,H,3,W,H,C)$ and store in compressed format - CPD, Tucker, ...
#
# **Non linear** $f_\theta(X) = \mathrm{CNN}_\theta(X)$

# +
import torch
from torch.utils.data import DataLoader

from compressai import zoo

import torchmetrics as tm
import torchmetrics.functional as tmF

import neuralcompression.data as nc_data
import neuralcompression as nc
import neuralcompression.functional as ncF
import neuralcompression.metrics as ncm

import torchvision.transforms as tfms

from tqdm import tqdm

import math
from PIL import Image

import matplotlib.pyplot as plt

import numpy as np
import tensorly as tl
tl.set_backend('pytorch')
from copy import deepcopy
from tensorly.decomposition import tucker
import argparse


# -

def solve_X(X,f,Y,Z,T,eps,tol=1e-5):
    X_new = sum([tl.fold(T[n] + Z[n],n,X.shape) for n in range(len(Z))]) / 3
    if (f(X_new) - Y).norm() <= eps:
        return X_new, False
    else:
        while (X_new - X).norm() / X.norm() > tol:
            if (f(X_new * 0.5 + X * 0.5) - Y).norm() <= eps:
                X = X_new * 0.5 + X * 0.5
            else:
                X_new = X_new * 0.5 + X * 0.5
        return X, True


def solve_theta(X,f,Y, iters):
    for param in f.parameters():
        param.requires_grad=True
    optimizer = torch.optim.Adam(f.parameters(),lr=0.00002)
    best_loss = (f(X) - Y).norm()
    f_copy = deepcopy(f)
    for i in range(iters):
        loss = (f(X) - Y).norm()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss.item():
            f_copy = deepcopy(f)
    f.load_state_dict(f_copy.state_dict())
    for param in f.parameters():
        param.requires_grad=False
    end_loss = (f(X) - Y).norm().item()
    return end_loss


# +
def D_tau(X, tau):
    u, s, v = torch.svd(X)
    s_tau = torch.max(s-tau,torch.zeros_like(s))[None]
    return (u * s_tau) @ v.t()

def X_init(X,f,Y,iters=5000):
    X.requires_grad = True
    optimizer = torch.optim.Adam([X]+[param for param in f.parameters()], lr=0.0002)
    for param in f.parameters():
        param.requires_grad=True
    pbar = tqdm(range(iters))
    for i in pbar:
        loss = (f(X) - Y).norm()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(
            f"Loss {round(loss.item(),3)}"
        )
    print(f"|f(X)-Y| = {round(loss.item(),1)}")
    X.requires_grad=False
    for param in f.parameters():
        param.requires_grad=False
    return X

def opt_rank_tucker(X, f, Y, delta, alpha=None, lamda=1e6, iters=1000, device="cpu"):
    if alpha is None:
        alpha = tl.tensor(np.array(X.shape)) / np.sum(np.array(X.shape))
    print(f"Alpha {alpha}")
    Y = Y.to(device)
    
    #init all parameters
    
    Z = [tl.unfold(tl.copy(X),n).to(device) for n in range(len(X.shape))]
    T = [tl.zeros(z.shape).to(device) for z in Z]
    best_rank_sum = sum(tl.tensor(Y.shape) * alpha)
    best_rank = list(Y.shape)
    X_best = tl.copy(X)
    sum_s = []
    re_xy = []
    
    #perform iterations
    pbar = tqdm(range(iters))
    for i in pbar:
        
        #Solve Z
        for n in range(len(Z)):
            Z[n] = D_tau(tl.unfold(X,n) - T[n], alpha[n] / lamda)
            
        # Solve X
        X, flag = solve_X(X,f,Y,Z,T,delta)
        
        # Solve theta
        if flag:
            losses_theta = solve_theta(X,f,Y,500)
        else:
            losses_theta = solve_theta(X,f,Y,10)
            
        #Solve T
        for n in range(len(Z)):
            T[n] += Z[n] - tl.unfold(X,n)
        
        nnz = []
        sing_sum = []
        norms = []
        
        #Evaluate performance
        for n in range(len(Z)):
            svd_X = torch.svd(tl.unfold(X,n),compute_uv=False)[1]
            sing_sum.append(sum(svd_X) * alpha[n])
            nnz.append(torch.linalg.matrix_rank(tl.unfold(X,n),tol=1e-3*svd_X[0]).item())
            norms.append((torch.norm(tl.unfold(X,n) - Z[n]) / torch.norm(X)).cpu().item())
        if best_rank_sum > sum(tl.tensor(nnz) * alpha):
            X_best = tl.copy(X)
            best_rank_sum = sum(tl.tensor(nnz) * alpha)
            best_rank = nnz
        sum_s.append(sum(sing_sum).item())
        re_xy.append((torch.norm(f(X) - Y) / delta).item())
        pbar.set_description(
            f"R {nnz} || "
            f"sum(S) {round(sum(sing_sum).item(),4)} || "
            f"RE_Z 10^{round(math.log(np.mean(norms),10))} || "
            f"XY {round(losses_theta / delta,2)}"
        )
        if i >= 9 and np.std(sum_s[-10:]) < 0.1:
            break
        
    print(
        f"Best rank {best_rank}"
    )
    return X_best, best_rank, sum_s, re_xy


# -

# Pass download=True if dataset doesn't exist already
ds = nc_data.Kodak("kodak", transform = tfms.ToTensor())
dl = DataLoader(ds, batch_size = 1)
metrics = tm.MetricCollection(
    tm.PSNR(), 
    tm.MeanSquaredError(), 
    ncm.MultiscaleStructuralSimilarity()
)


# +
class Tanh_normal(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,X):
        return torch.nn.Tanh()(X) * 0.5 + 0.5

class F(torch.nn.Module):
    def __init__(self, ker_size):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3,64,kernel_size=ker_size,padding=ker_size // 2, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,3,kernel_size=ker_size,padding=ker_size // 2, bias=False)
        )
    def forward(self,X):
        return self.model(X[None])[0]


# -

# $$\varepsilon = \sqrt{C*H*W*10^{-\frac{\mathrm{PSNR}}{10}}}$$
#
# $$\min\sum\limits_{n=1}^3 \alpha_n\mathrm{Rank}(X_{(n)})$$
# $$s.t. \|Y-X\|_F \leq \varepsilon$$
#
# $$\min\sum\limits_{n=1}^3 \alpha_n\|X_{(n)}\|_*$$
# $$s.t. \|Y-X\|_F \leq \varepsilon$$
#
# $$\min\sum\limits_{n=1}^3 \alpha_n\|Z_{n}\|_*$$
# $$s.t. \|Y-X\|_F \leq \varepsilon$$
# $$X_{(n)} = Z_n$$
#
# $$L = \sum\limits_{n=1}^3 \alpha_n\|X_{(n)}\|_* + i_D(X) + \frac{\lambda}{2}(\|X_{(n)} - Z_n - T_n\|_F^2 - \|T_n\|_F^2)$$

def PSNR_from_MSE(mse):
    return 10 * math.log(1 / mse,10)


def X_init_tucker(X_shape,f,Y,R,device, iters=20000):
#     facs = tl.random.random_tucker(X_shape,R)
#     facs = [facs[0]] + facs[1]  
    X = X_init(torch.randn(X_shape,device=device) / np.prod(X_shape),f,Y)
    facs = tucker(X,R, init="random")
    facs = [facs[0]] + facs[1]
    opt_params = [opt.clone().detach().to(device).requires_grad_(True) for opt in facs]
    for param in f.parameters():
        param.requires_grad = True
    
    #optimization of f and tucker components of X
    optimizer = torch.optim.Adam(opt_params + [param for param in f.parameters()], lr=0.0002)
    
    pbar = tqdm(range(iters))
    best_loss = np.infty
    best_X = tl.tucker_to_tensor((opt_params[0],opt_params[1:]))
    
    for i in pbar:
        loss = torch.nn.MSELoss()(Y, f(tl.tucker_to_tensor((opt_params[0],opt_params[1:]))))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pbar.set_description(
            f"PSNR {round(PSNR_from_MSE(loss.item()),1)}"
        )
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_X = tl.copy(tl.tucker_to_tensor((opt_params[0],opt_params[1:]))).detach()
    for param in f.parameters():
        param.requires_grad = False
    return best_X.requires_grad_(False), (Y - f(best_X)).norm().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compression parameters"
    )

    parser.add_argument(
        "--PATH", type=str, help="path to the image"
    )
    
    parser.add_argument("--R", type=int, default=20, help="initial rank for algorithm")

    args = parser.parse_args()

ker_size=3
R = args.R
device = "cuda"

# +
bpp_sum = 0.
metrics_sum = {'PSNR': 0., 'MeanSquaredError': 0., 'MultiscaleStructuralSimilarity': 0.}

#load Image
img = tfms.ToTensor()(Image.open(args.PATH))
X_shape = list(img.shape)
torch.manual_seed(0)

#initialize f
f = F(ker_size).to(device)

#first part of dual problem
X,delta = X_init_tucker(X_shape,f,img.to(device),R,device, iters=10000)

#second part of dual problem
X = opt_rank_tucker(X.detach(),f,img,delta, lamda=1e0, iters=1000, device=device)

#metrics evaluation
for k,v in metrics(f(X[0])[None].detach().cpu(),img[None]).items():
    metrics_sum[k] = v.item()
bpp_sum = (sum([np.prod(param.shape) for param in f.parameters()]) + np.prod(X[1]) + sum(np.array(X[1]) * np.array(X_shape)))*32 / np.prod(img.shape[1:])

#saving compressed image and printing results
# tfms.ToPILImage()((X[0] + (X[0].reshape(3,-1)).min(dim=1)[:,None,None]) / ((X[0].reshape(3,-1)).max(dim=1)[:,None,None] - (X[0].reshape(3,-1)).min(dim=1)[:,None,None])).save("./preimage.png")
tfms.ToPILImage()(f(X[0]).detach().cpu().clamp(0,1)).save("./compressed_image.png")
print("Metrics:")
print(metrics_sum)
print(f"BPP: {round(bpp_sum,1)}")
