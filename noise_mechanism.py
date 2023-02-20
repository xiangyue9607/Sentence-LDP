import math

import numpy as np
import torch
from scipy import integrate


# The angular density f(x) = e^(-eps*x)*sin^(d-2)(x)
def f(x, d, eps):
    return math.exp(-eps * x) * math.pow(math.sin(x), (d - 2))


# Exactly follow Alg.1 of CCS21 paper -- DP for Directional Data
def PurArc(d, eps):
    a = 0
    b = math.pi
    u = np.random.uniform(0, 1)
    for i in range(1, 25):
        theta = (a + b) / 2
        y = integrate.quad(f, 0, theta, args=(d, eps))[0] / integrate.quad(f, 0, math.pi, args=(d, eps))[0]
        if y < u:
            a = theta
        elif y > u:
            b = theta
    return theta


def PurMech(sentence_embeddings, eps):
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, dim=-1)
    n = sentence_embeddings.size(0)
    d = sentence_embeddings.size(1)
    direct = torch.randn((n, d), device=sentence_embeddings.device, requires_grad=False)
    direct_update = direct - torch.sum(direct * sentence_embeddings, dim=-1, keepdim=True) * sentence_embeddings
    direct_update = torch.nn.functional.normalize(direct_update, dim=-1)
    theta = torch.tensor([[PurArc(d, eps)] * d for _ in range(n)], dtype=torch.float, device=sentence_embeddings.device,
                         requires_grad=False)
    noisy_embedd = torch.cos(theta) * sentence_embeddings + torch.sin(theta) * direct_update
    return noisy_embedd


def LapMech(sentence_embeddings, eps):
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, dim=-1)
    n = sentence_embeddings.size(0)
    d = sentence_embeddings.size(1)
    scale = torch.distributions.gamma.Gamma(d, eps).sample((n,)).to(sentence_embeddings.device)
    noise = torch.randn((n, d), device=sentence_embeddings.device, requires_grad=False)
    noise = torch.nn.functional.normalize(noise, dim=-1)
    sentence_embeddings = sentence_embeddings + scale.unsqueeze(-1) * noise
    return sentence_embeddings
