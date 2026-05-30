import torch

from open_mythos import (
    mythos_1b,
    OpenMythos,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

cfg = mythos_1b()
model = OpenMythos(cfg).to(device)

total = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total:,}")
