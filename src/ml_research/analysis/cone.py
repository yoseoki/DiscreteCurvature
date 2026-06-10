import numpy as np
import cupy as cp
import copy
import fnnlsEigen as fe
from tqdm import tqdm
import torch

def nnls_torch(A, b, lr=1e-3, steps=1000):
    A = torch.as_tensor(A, device="cuda", dtype=torch.float64)
    b = torch.as_tensor(b, device="cuda", dtype=torch.float64)

    x = torch.randn(
        (A.shape[1],),
        device=A.device,
        dtype=A.dtype,
        requires_grad=True
    )

    # 초기 non-negativity projection
    with torch.no_grad():
        x.clamp_(min=0)

    for _ in range(steps):
        loss = torch.norm(A @ x - b) ** 2
        loss.backward()

        with torch.no_grad():
            x -= lr * x.grad
            x.clamp_(min=0)

        x.grad.zero_()

    return x.detach()

class WeightCONE():
    def __init__(self):
        pass

    def nmf_basic(self, dataArray, max_iter=50):
        d = dataArray.shape[0]
        n = dataArray.shape[1]
        r = n
        print("(d, n, r) = ({:02d}, {:02d}, {:02d})".format(d, n, r))
        W = np.random.rand(d, r)
        W = np.asarray(W, dtype=np.float64)
        H = np.random.rand(r, n)
        H = np.asarray(H, dtype=np.float64)
        
        for it in tqdm(range(max_iter)):
            for i in range(d):
                b = dataArray[i, :].get()
                b = np.asarray(b, dtype=np.float64)
                W[i, :] = fe.fnnls(H, b)
            
            for j in range(n):
                b = dataArray[:, j].get()
                b = np.asarray(b, dtype=np.float64)
                H[:, j] = fe.fnnls(W, b)

        return W
    
    def nmf_projection(self, B, x):
        w = fe.fnnls(B, x)
        w = np.expand_dims(w, axis=-1)
        return np.squeeze(B @ w)
    
class ConeDiff():
    def __init__(self):
        self.weightCONE = WeightCONE()

    def orthogonal_complement(self, vectors, tol=1e-10):
        U, S, Vh = np.linalg.svd(vectors.T, full_matrices=True)

        rank = np.sum(S > tol)
        return Vh[rank:].T

    def calc_similarity(self, basis1, basis2):

        b1 = basis1
        b2 = basis2
        r = min(basis1.shape[1], basis2.shape[1])
        sim_container = []

        for _ in tqdm(range(r)):
            y = 0.5 * (b1[:, 0] + b2[:, 0])
            while True:
                py1 = self.weightCONE.nmf_projection(b1, y)
                py2 = self.weightCONE.nmf_projection(b2, y)
                y_next = 0.5 * (py1 + py2)
                if np.linalg.norm(y - y_next) < 1e-6:
                    sim = np.dot(py1, py2) / (np.linalg.norm(py1) * np.linalg.norm(py2))
                    sim_container.append(sim**2)
                    break
                y = y_next

            vectors = np.concatenate((np.expand_dims(py1, axis=-1), np.expand_dims(py2, axis=-1)), axis=1)
            P = self.orthogonal_complement(vectors)
            # print(P.shape)
            b1 = np.transpose(P) @ b1
            b2 = np.transpose(P) @ b2

        return 2 * (len(sim_container) - sum(sim_container))