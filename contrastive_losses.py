import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the contrastive loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        cos_sim = F.cosine_similarity(output1, output2, dim=-1)
        loss = torch.mean((1 - target) * torch.pow(cos_sim, 2) + (target) * torch.pow(torch.clamp(self.margin - cos_sim, min=0.0), 2))
        return loss

# class RankingContrastiveLoss(nn.Module):
#     def __init__(self, temperature=1.0, margin=0.2):
#         super(RankingContrastiveLoss, self).__init__()
#         self.temperature = temperature
#         self.margin = margin

#     def forward(self, cosine_similarity, adjacency_matrix):
        
#         batch_size, _, _ = cosine_similarity.size()

#         # Reshape cosine similarity and adjacency matrix to (batch_size, num_nodes^2)
#         cosine_similarity_flat = cosine_similarity.view(batch_size, -1)
#         adjacency_matrix_flat = adjacency_matrix.view(batch_size, -1)

#         # Compute masks for positive and negative pairs
#         pos_mask = adjacency_matrix_flat > 0
#         neg_mask = adjacency_matrix_flat == 0

#         # Compute positive loss
#         pos_cos_sim = cosine_similarity_flat.masked_select(pos_mask)
#         pos_cos_sim = pos_cos_sim.view(batch_size, -1)  # Reshape to maintain batch dimension
#         pos_cos_sim = pos_cos_sim.unsqueeze(2)  # Add singleton dimension for broadcasting
#         pos_loss = torch.log(1 + torch.exp(-pos_cos_sim / self.temperature))
#         total_pos_pairs = pos_loss.size(1)

#         # Compute negative loss
#         neg_cos_sim = cosine_similarity_flat.masked_select(neg_mask)
#         neg_cos_sim = neg_cos_sim.view(batch_size, -1)  # Reshape to maintain batch dimension
#         neg_cos_sim = neg_cos_sim.unsqueeze(1)  # Add singleton dimension for broadcasting
#         neg_loss = torch.log(1 + torch.exp(neg_cos_sim / self.temperature))
#         total_neg_pairs = neg_loss.size(2)

#         # Normalize by the total number of pairs
#         total_pairs = total_pos_pairs + total_neg_pairs

#         # Compute total loss
#         total_loss = torch.mean(-torch.log(pos_loss / (pos_loss + neg_loss))) / total_pairs

#         return total_loss

class RankingContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(RankingContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, cosine_similarity, adjacency_matrix):
        # Flatten the cosine similarity and adjacency matrix to 2D tensors
        cosine_similarity_flat = cosine_similarity.view(cosine_similarity.size(0), -1)
        adjacency_matrix_flat = adjacency_matrix.view(adjacency_matrix.size(0), -1)
        
        # Masks for positive and negative pairs
        pos_mask = adjacency_matrix_flat > 0
        neg_mask = adjacency_matrix_flat == 0

        # Positive loss: the goal is to bring these pairs closer, maximizing their similarity
        pos_cos_sim = cosine_similarity_flat[pos_mask]
        pos_loss = torch.mean(1 - pos_cos_sim)  # Encourages high similarity for positives

        # Negative loss: hinge loss that enforces a margin to keep negative pairs distant
        neg_cos_sim = cosine_similarity_flat[neg_mask]
        neg_loss = torch.mean(F.relu(neg_cos_sim - self.margin))  # Margin-based hinge loss

        # Final loss: Sum of positive and negative loss components
        total_loss = pos_loss + neg_loss  # Adding both components
        
        return total_loss


class TemporalContrastiveLoss(nn.Module):
    def __init__(self, temperature=1.0, softplus_w=10e3):
        super(TemporalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.softplus_w = softplus_w

    def _calculate_weight(self, temporal_similarity_matrix):
        sigma = torch.std(temporal_similarity_matrix)
        gaussian_weight = torch.exp(-0.5 * (temporal_similarity_matrix - torch.mean(temporal_similarity_matrix))**2 / sigma**2)
        return gaussian_weight / gaussian_weight.sum()
    
    def forward(self, temporal_similarity_matrix):
        # Initialize loss
        loss = 0
        total_frames = temporal_similarity_matrix.size(0)

        for i in range(total_frames):
            loss_i = 0
            # Calculate the prior Gaussian weight
            weight = self._calculate_weight(temporal_similarity_matrix)

            # Compute the loss for the i-th reference frame
            for j in range(total_frames):
                if j != i:
                    sim_ij = torch.cosine_similarity(temporal_similarity_matrix[i], temporal_similarity_matrix[j], dim=0) / self.temperature
                    exp_sim = torch.exp(sim_ij)
                    sum_exp_sim = torch.sum(exp_sim)
                    NLL = -torch.log(exp_sim / sum_exp_sim)
                    
                    loss_i += F.kl_div(NLL, weight[i], reduction="none")

            # Aggregate the loss for the i-th reference frame
            loss_i = (loss_i * weight[i]).sum()
            loss = loss_i + torch.log( 1 + torch.exp(self.softplus_w*loss_i) ) # softplus function

        return loss