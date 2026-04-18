"""Identify the 26 False Negative candidates from DenseNet3D test set."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'pipeline'))
sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'pipeline' / 'fase2'))
sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'pipeline' / 'fase2' / 'models'))

from expert3_densenet3d import Expert3DenseNet3D

# Paths
CHECKPOINT_PATH = PROJECT_ROOT / 'checkpoints' / 'expert_03_densenet3d' / 'best.pt'
PATCHES_BASE = PROJECT_ROOT / 'datasets' / 'luna_lung_cancer' / 'patches'
CSV_PATH = PROJECT_ROOT / 'datasets' / 'luna_lung_cancer' / 'candidates_V2' / 'candidates_V2.csv'

# Load label map
cand_df = pd.read_csv(CSV_PATH)
label_map = dict(zip(cand_df.index, cand_df['class']))

class LUNA16Simple(Dataset):
    def __init__(self, patches_dir, label_map):
        self.samples = []
        for f in sorted(Path(patches_dir).glob('candidate_*.npy')):
            try:
                idx = int(f.stem.split('_')[1])
            except (IndexError, ValueError):
                continue
            lbl = label_map.get(idx, -1)
            if lbl >= 0:
                self.samples.append((f, lbl))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        vol = np.load(path).astype(np.float32)
        return torch.from_numpy(vol).unsqueeze(0), label, path.stem

# Load model
ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
cfg = ckpt['config']
model = Expert3DenseNet3D(
    in_channels=1, num_classes=2, growth_rate=32,
    block_layers=[4, 8, 16, 12], init_features=64,
    spatial_dropout_p=cfg['spatial_dropout_3d'],
    fc_dropout_p=cfg['dropout_fc'],
)
state = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
model.load_state_dict(state)
model.eval()

# Test loader
test_ds = LUNA16Simple(PATCHES_BASE / 'test', label_map)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)

print(f"Test samples: {len(test_ds)}")

# Inference with stems
all_labels = []
all_probs = []
all_stems = []

with torch.no_grad():
    for batch_idx, (volumes, labels, stems) in enumerate(test_loader):
        logits = model(volumes)
        probs = F.softmax(logits, dim=1)[:, 1]
        all_labels.extend(labels.numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_stems.extend(stems)
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{len(test_loader)}")

labels_arr = np.array(all_labels)
probs_arr = np.array(all_probs)
preds_arr = (probs_arr >= 0.5).astype(int)

# Find FN: label=1, pred=0
fn_mask = (labels_arr == 1) & (preds_arr == 0)
fn_stems = [all_stems[i] for i in range(len(all_stems)) if fn_mask[i]]

print(f"\nTotal FN: {fn_mask.sum()}")
print(f"\n{'='*60}")
print("FALSE NEGATIVE candidates (label=1, pred=0):")
print(f"{'='*60}")
for i, stem in enumerate(fn_stems):
    prob = probs_arr[[j for j in range(len(all_stems)) if fn_mask[j]][i]]
    print(f"  {i+1:2d}. {stem}.npy  (prob={prob:.4f})")

# Save to file
output_path = PROJECT_ROOT / 'notebooks' / 'evaluacion_expert3' / 'false_negatives_test.txt'
with open(output_path, 'w') as f:
    for stem in fn_stems:
        f.write(f"{stem}.npy\n")
print(f"\nSaved to: {output_path}")
