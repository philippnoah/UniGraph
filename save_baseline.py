"""Save a checkpoint of the untrained model (epoch 0, no training steps)."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
from utils.functions import build_args, set_random_seed
from model import UniGraph

args = build_args()
set_random_seed(args.seed)
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

model = UniGraph(args).to(device)

save_dir = args.checkpoint_path if args.checkpoint_path else "checkpoints"
os.makedirs(save_dir, exist_ok=True)
path = os.path.join(save_dir, "pretrain_epoch_0_baseline.pt")
torch.save({"epoch": -1, "model_state_dict": model.state_dict(), "pretrain_loss": None}, path)
print(f"Saved untrained baseline to {path}")
