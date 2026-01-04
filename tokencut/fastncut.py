import numpy as np
import torch


def _pinv(matrix: torch.Tensor) -> torch.Tensor:
    # Prefer GPU for speed; fall back to CPU if MAGMA/GPU SVD fails.
    device = matrix.device
    linalg = getattr(torch, "linalg", None)
    if linalg is not None and hasattr(linalg, "pinv"):
        try:
            return linalg.pinv(matrix)
        except Exception:
            pass
    else:
        try:
            return torch.pinverse(matrix)
        except Exception:
            pass

    matrix_cpu = matrix.detach().cpu()
    if linalg is not None and hasattr(linalg, "pinv"):
        pinv_cpu = linalg.pinv(matrix_cpu)
    else:
        pinv_cpu = torch.pinverse(matrix_cpu)
    return pinv_cpu.to(device)


class FastNcut:
    def __init__(
        self,
        A=np.array([[2, 0, 0], [0, 1, 0], [0, 0, 0]]),
        const=np.array([[0, 1]]),
        max_iter=1000,
        opt_tol=1e-12,
    ):
        self.A = A
        self.const = const
        self.max_iter = max_iter
        self.opt_tol = opt_tol
        self.B, self.c = self._init_const(self.const)
        self.BB_inv = np.linalg.pinv(
            self.B @ self.B.T
        )  # Pre-Compute and Store for Reuse

    def _init_const(self, const):
        const_num = len(const)
        B = np.zeros([const_num, len(self.A)])
        for i, pair in enumerate(const):
            B[i][pair] = [1, -1]
        c = np.zeros([const_num, 1])
        return B, c

    def _projected_powermethod(self):
        A = torch.from_numpy(self.A).to(torch.float32).cuda()
        B = torch.from_numpy(self.B).to(torch.float32).cuda()
        c = torch.from_numpy(self.c).to(torch.float32).cuda()

        P = (
            torch.eye(len(A), dtype=torch.float32).cuda()
            - B.T @ _pinv(B @ B.T) @ B
        )
        PA = P @ A  # Pre-Compute PA

        k = 0
        n_0 = B.T @ _pinv(B @ B.T) @ c
        ganma = torch.sqrt(1 - torch.norm(n_0) ** 2)

        if torch.count_nonzero(B) == 0:
            v = ganma * PA @ n_0 / torch.norm(PA @ n_0) + n_0
        else:
            v = torch.rand(len(A), 1, dtype=torch.float32).cuda()

        obj = v.T @ A @ v
        obj_old = obj

        while k < self.max_iter:
            v /= torch.norm(v)
            u = ganma * PA @ v / torch.norm(PA @ v)
            v = u + n_0
            k += 1
            obj = v.T @ A @ v
            if self.opt_tol > abs(obj - obj_old):
                break
            obj_old = obj
        return v.cpu().numpy(), k

    def fit(self, X):
        const_num = 1
        const_eig_vec, iter_num = self._projected_powermethod()
        return const_eig_vec
