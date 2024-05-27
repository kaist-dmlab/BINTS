import torch

# Convenience l2_norm function
def l2_norm(x, axis=None):
    """
    Takes an input tensor and returns the l2 norm along the specified axis
    """
    square_sum = torch.sum(torch.square(x), axis=axis, keepdim=True)
    norm = torch.sqrt(torch.clamp(square_sum, min=torch.finfo(x.dtype).eps))
    return norm

def pairwise_cosine_sim(A_B):
    """
    A [batch x n x d] tensor of n rows with d dimensions
    B [batch x m x d] tensor of n rows with d dimensions

    Returns:
    D [batch x n x m] tensor of cosine similarity scores between each point i<n, j<m
    """
    A, B = A_B
    A_mag = l2_norm(A, axis=2)
    B_mag = l2_norm(B, axis=2)
    num = torch.bmm(A, B.permute(0, 2, 1))
    den = torch.bmm(A_mag, B_mag.permute(0, 2, 1))
    dist_mat = num / den
    return dist_mat