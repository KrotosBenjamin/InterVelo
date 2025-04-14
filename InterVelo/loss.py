from typing import Optional
import torch

def pearson(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.BoolTensor] = None,
) -> torch.Tensor:
    """
    The pearson correlation coefficient between two vectors. by default, the
    first dimension is the sample dimension.

    Args:
    x (torch.Tensor):
        The first vector, (N, D).
    y (torch.Tensor):
        The second vector, (N, D).
    mask (torch.BoolTensor):
        The mask showing valid data positions, (N, D).

    Returns: (torch.Tensor), (D,)
    """
    if mask is None:
        x = x - torch.mean(x, dim=0)
        y = y - torch.mean(y, dim=0)
        x = x / (torch.std(x, dim=0) + 1e-9)
        y = y / (torch.std(y, dim=0) + 1e-9)
        return torch.mean(x * y, dim=0)  # (D,)
    else:
        assert mask.dtype == torch.bool
        mask = mask.detach().float()
        num_valid_data = torch.sum(mask, dim=0)  # (D,)

        y = y * mask
        x = x * mask
        x = x - torch.sum(x, dim=0) / (num_valid_data + 1e-9)
        y = y - torch.sum(y, dim=0) / (num_valid_data + 1e-9)
        y = y * mask  # make the invalid data to zero again to ignore
        x = x * mask
        x = x / torch.sqrt(torch.sum(torch.pow(x, 2), dim=0) + 1e-9)
        y = y / torch.sqrt(torch.sum(torch.pow(y, 2), dim=0) + 1e-9)
        return torch.sum(x * y, dim=0)  # (D,)
    
def direction_loss(
    velocity: torch.Tensor,
    spliced_counts: torch.Tensor,
    unspliced_counts: torch.Tensor,
    coeff_u: float = 1.0,
    coeff_s: float = 1.0,
    reduce: bool = True,
) -> torch.Tensor:
    """
    The constraint for the direction of the velocity vectors. Large ratio of u/s
    should have positive direction, for each gene.

    Args:
    velocity (torch.Tensor):
        The predicted velocity vectors, (batch_size, genes).
    spliced_counts (torch.Tensor):
        The number of spliced reads, (batch_size, genes).
    unspliced_counts (torch.Tensor):
        The number of unspliced reads, (batch_size, genes).
    velocity_u (torch.Tensor):
        The predicted velocity vectors for unspliced reads, (batch_size, genes).
    coeff_u (float):
        The coefficient for the unspliced reads.
    coeff_s (float):
        The coefficient for the spliced reads.
    reduce (bool):
        Whether to reduce the loss to a scalar.

    Returns: (torch.Tensor), (1,)
    """

    # 3. Intereting for the gliogenic cells of the GABAInterneuraons ====
    mask1 = unspliced_counts > 0
    mask2 = spliced_counts > 0
    corr = coeff_u * pearson(velocity, unspliced_counts, mask1) - coeff_s * pearson(
        velocity, spliced_counts, mask2
    )

    if not reduce:
        if torch.mean(corr / (coeff_u + coeff_s)) >= 0 :
            reverse = False
        else:
            reverse = True
        return corr / (coeff_u + coeff_s), reverse  # range [-1, 1], shape (genes,)

    loss = coeff_u + coeff_s - torch.mean(corr)  # to maximize the correlation
    loss = loss / (coeff_u + coeff_s)
    # should not use correlation to u, since the alpha coefficient is unknown and
    # it definitely varies along the phase portrait.
    # if velocity_u is not None:
    #     corr_unpliced = pearson(velocity_u, unspliced_counts, mask)
    #     loss_u = 1 + torch.mean(corr_unpliced)  # mininize the corr_unpliced
    #     loss = 0.5 * (loss + loss_u)
        
    return loss