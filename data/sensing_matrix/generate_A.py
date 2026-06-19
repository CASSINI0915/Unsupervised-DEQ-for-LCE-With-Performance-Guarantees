import math
from pathlib import Path

import torch
from scipy.io import savemat


def to_numpy(tensor):
    """Convert a torch tensor to a numpy array for scipy savemat.
    
    :param tensor: Tensor that may live on GPU or have a conjugate view bit.
    :return: CPU numpy array with conjugation resolved.
    """
    return tensor.detach().cpu().resolve_conj().numpy()

# ----------------------------
# Parameters
# ----------------------------
M = 128
N = 256

# ----------------------------
# 1) Complex Gaussian i.i.d. Matrix (MxN)
#    Each entry ~ CN(0, 1/M)
#    That means real/imag parts each ~ N(0, 1/(2M)).
# ----------------------------
# Create a float tensor for real and imaginary parts, then combine
A_gaussian = (1.0 / math.sqrt(2.0 * M)) * (
    torch.randn(M, N, dtype=torch.float32) +
    1j * torch.randn(M, N, dtype=torch.float32)
)
# Ensure PyTorch sees it as a complex tensor
A_gaussian = A_gaussian.to(torch.complex64)

# ----------------------------
# 2) Random Bernoulli Matrix (MxN)
#    Each entry = ±1/sqrt(M) with probability 1/2
# ----------------------------
A_bernoulli = (1.0 / math.sqrt(M)) * (
    2.0 * torch.randint(0, 2, (M, N), dtype=torch.float32) - 1.0
)

# ----------------------------
# 3) Row-Orthonormal Matrix (MxN)
#    Derived from a complex Gaussian matrix by QR decomposition
#    so that the final matrix has orthonormal rows.
# ----------------------------
# Perform QR on A^H for a complex-valued matrix.
# Q will be size (N x M), so Q^H is (M x N).
Q, _ = torch.linalg.qr(A_gaussian.conj().T)
A_orth = Q.conj().T  # This has orthonormal rows if M <= N

# ----------------------------
# Save all to a .mat file
# ----------------------------
out_dir = Path(__file__).resolve().parent / "matrices"
out_dir.mkdir(parents=True, exist_ok=True)
savemat(out_dir / "A_gaussian256128.mat", {"A": to_numpy(A_gaussian)})
savemat(out_dir / "A_bernoulli256128.mat", {"A": to_numpy(A_bernoulli)})
savemat(out_dir / "A_unitary256128.mat", {"A": to_numpy(A_orth)})




