import torch
import torch.nn as nn


class LanguageModelCriterion(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super(LanguageModelCriterion, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        nll = -input.gather(2, target.long().unsqueeze(2)).squeeze(2)
        if self.label_smoothing > 0:
            smooth = -input.mean(dim=2)
            output = ((1 - self.label_smoothing) * nll + self.label_smoothing * smooth) * mask
        else:
            output = nll * mask
        return torch.sum(output) / torch.sum(mask)


class CriterionWrapper(nn.Module):
    """Wraps LanguageModelCriterion to handle reports_ids/masks slicing."""
    def __init__(self, label_smoothing=0.0):
        super(CriterionWrapper, self).__init__()
        self.criterion = LanguageModelCriterion(label_smoothing=label_smoothing)

    def forward(self, output, reports_ids, reports_masks):
        return self.criterion(output, reports_ids[:, 1:], reports_masks[:, 1:])


class LossWrapper(nn.Module):
    def __init__(self):
        super(LossWrapper, self).__init__()
        self.criterion = LanguageModelCriterion()
        self.criterion_mlc = nn.BCELoss()

    def forward(self, output, output_mlc, reports_ids, reports_masks, label):
        loss = self.criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
        loss_mlc = self.criterion_mlc(output_mlc, label)
        return loss + loss_mlc


def compute_loss(output, reports_ids, reports_masks):
    criterion = LanguageModelCriterion()
    loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
    return loss


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)

        input = input.reshape(-1)
        reward = reward.reshape(-1)
        mask = (seq > 0).to(input)
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).reshape(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output
