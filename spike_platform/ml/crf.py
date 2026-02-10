"""
Linear-chain Conditional Random Field (CRF) layer for sequence labeling.

Enforces valid transition constraints between tags (e.g., approach → jump → swing → land).
Used on top of LSTM emissions to produce globally consistent tag sequences.
"""

import torch
import torch.nn as nn


class LinearChainCRF(nn.Module):
    """Linear-chain CRF for sequence labeling."""

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        # Transition scores: transitions[i][j] = score of transitioning from tag i to tag j
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log-likelihood loss.

        Args:
            emissions: (B, T, num_tags) — logits from LSTM
            tags: (B, T) — ground truth tag indices
            mask: (B, T) — boolean mask (True = valid, False = padding)

        Returns:
            Scalar loss (mean over batch)
        """
        gold_score = self._score_sentence(emissions, tags, mask)
        forward_score = self._forward_algorithm(emissions, mask)
        # NLL = log Z - score(gold)
        return (forward_score - gold_score).mean()

    def _score_sentence(self, emissions, tags, mask):
        """Score of the gold tag sequence."""
        B, T, C = emissions.shape
        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        for t in range(1, T):
            m = mask[:, t].float()
            emit = emissions[:, t].gather(1, tags[:, t].unsqueeze(1)).squeeze(1)
            trans = self.transitions[tags[:, t - 1], tags[:, t]]
            score += (emit + trans) * m

        # End transition: find last valid position per sequence
        last_idx = mask.long().sum(dim=1) - 1  # (B,)
        last_tags = tags.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]
        return score

    def _forward_algorithm(self, emissions, mask):
        """Compute log partition function (forward algorithm)."""
        B, T, C = emissions.shape
        # alpha[b, c] = log-sum-exp of all paths ending at tag c at current timestep
        alpha = self.start_transitions + emissions[:, 0]  # (B, C)

        for t in range(1, T):
            m = mask[:, t].unsqueeze(1).float()  # (B, 1)
            emit = emissions[:, t].unsqueeze(1)  # (B, 1, C)
            trans = self.transitions.unsqueeze(0)  # (1, C, C)
            # alpha_next[b, j] = logsumexp_i(alpha[b, i] + trans[i, j] + emit[b, j])
            scores = alpha.unsqueeze(2) + trans + emit  # (B, C, C)
            alpha_next = torch.logsumexp(scores, dim=1)  # (B, C)
            alpha = alpha_next * m + alpha * (1 - m)

        alpha += self.end_transitions
        return torch.logsumexp(alpha, dim=1)  # (B,)

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        """Viterbi decoding — find best tag sequence.

        Args:
            emissions: (B, T, num_tags)
            mask: (B, T) — boolean mask

        Returns:
            List of B tag sequences (variable length)
        """
        B, T, C = emissions.shape
        # Viterbi forward
        viterbi = self.start_transitions + emissions[:, 0]  # (B, C)
        backpointers = []

        for t in range(1, T):
            m = mask[:, t].unsqueeze(1).float()  # (B, 1)
            emit = emissions[:, t]  # (B, C)
            trans = self.transitions.unsqueeze(0)  # (1, C, C)
            # scores[b, i, j] = viterbi[b, i] + trans[i, j]
            scores = viterbi.unsqueeze(2) + trans  # (B, C, C)
            best_scores, best_tags = scores.max(dim=1)  # (B, C)
            viterbi_next = best_scores + emit
            viterbi = viterbi_next * m + viterbi * (1 - m)
            backpointers.append(best_tags)

        viterbi += self.end_transitions

        # Backtrack
        best_paths = []
        best_last = viterbi.argmax(dim=1)  # (B,)

        for b in range(B):
            seq_len = mask[b].long().sum().item()
            path = [best_last[b].item()]
            for t in range(len(backpointers) - 1, -1, -1):
                if t + 1 >= seq_len:
                    continue
                path.append(backpointers[t][b, path[-1]].item())
            path.reverse()
            best_paths.append(path[:seq_len])

        return best_paths
