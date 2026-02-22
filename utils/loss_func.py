import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def sig_loss(x, y):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (x * y).sum(1)
    loss = torch.sigmoid(-loss)
    loss = loss.mean()
    return loss


def compute_mlm_loss(logits, input_ids, masked_input_ids, ignore_index=-100):
    """
    Compute masked language modeling loss over masked token positions only.

    Args:
        logits: token prediction logits [batch, seq_len, vocab_size]
        input_ids: original (unmasked) token ids [batch, seq_len]
        masked_input_ids: masked input ids [batch, seq_len]
        ignore_index: label value to ignore in cross-entropy loss
    """
    loss_fct = CrossEntropyLoss(ignore_index=ignore_index)
    # Labels: original token id at masked positions, ignore_index elsewhere
    labels = input_ids.clone().long()
    labels[masked_input_ids == input_ids] = ignore_index
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    return loss