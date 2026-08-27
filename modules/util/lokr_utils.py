import torch


def factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    """
    Factorizes the provided number into the product of two numbers.
    Copied from https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/lokr.py#L11
    """
    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        return m, n
    if factor == -1:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n


def make_kron(w1, w2, scale=1.0):
    """
    Kronecker product of two tensors.
    """
    if len(w2.shape) == 4: # For Conv2d
        w1 = w1.unsqueeze(2).unsqueeze(2)
    w2 = w2.contiguous()
    rebuild = torch.kron(w1, w2)
    return rebuild * scale


def vl_rearrange(g: torch.Tensor, out_l: int, out_k: int, in_m: int, in_n: int) -> torch.Tensor:
    """
    Van Loan-Pitsianis rearrangement of a (out_l*out_k, in_m*in_n) matrix.

    Index convention matches make_kron(w1, w2) with w1 of shape (out_l, in_m)
    and w2 of shape (out_k, in_n), so that
        vl_rearrange(make_kron(w1, w2)) == vec(w1) @ vec(w2).T
    (vec = row-major flatten). The best kron(w1, w2) approximation of g in the
    Frobenius norm therefore corresponds to the top singular pair of the
    rearranged matrix.
    """
    return g.reshape(out_l, out_k, in_m, in_n).permute(0, 2, 1, 3).reshape(out_l * in_m, out_k * in_n)


def nearest_kron_factors(g: torch.Tensor, out_l: int, out_k: int, in_m: int, in_n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the nearest Kronecker product factors of a (out_l*out_k, in_m*in_n)
    matrix g: the (w1, w2) minimizing ||g - kron(w1, w2)||_F, via SVD of the
    Van Loan rearrangement. The leading singular value is split evenly between
    the factors. Returns (w1, w2, sigma) with w1 (out_l, in_m), w2 (out_k, in_n).
    """
    r = vl_rearrange(g, out_l, out_k, in_m, in_n)
    u, s, vh = torch.linalg.svd(r.float(), full_matrices=False)
    sigma = s[0]
    sqrt_sigma = sigma.sqrt()
    w1 = (u[:, 0] * sqrt_sigma).reshape(out_l, in_m)
    w2 = (vh[0, :] * sqrt_sigma).reshape(out_k, in_n)
    return w1, w2, sigma


def rebuild_tucker(t, wa, wb):
    """
    Rebuilds tensor from Tucker decomposition for convolutional layers.
    t: [r, r, k1, k2]
    wa: [r, b]
    wb: [r, d]
    rebuild: [b, d, k1, k2]
    """
    rebuild = torch.einsum('i j k l, i p, j r -> p r k l', t, wa, wb) # [c, d, k1, k2]
    return rebuild
