import os
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbones.iresnet import iresnet50
from datasets import CasiaWebFace, BinDataset

# numpy bool alias compatibility for older code that expects np.bool
if not hasattr(np, "bool"):
    np.bool = np.bool_

# ----------------------
# Configuration
# ----------------------
WEIGHTS_DIR = "/home/seidel/r50_webface_COSFace035"
CASIA_ROOT = "/data/ozgur/casia_training"
CASIA_NUM_CLASSES = 10572
VAL_DIR = "/home/seidel/validation"
VAL_TARGETS = [
    "lfw",
    "cfp_fp",
    "cfp_ff",
    "agedb_30",
    "cplfw",
    "calfw",
]

OUTPUT_GRADS_DIR = "output/gradients"
OUTPUT_INFL_DIR = "output/tracin"


# ----------------------
# Loss and grad helpers
# ----------------------

def contrastive_loss(features1, features2, labels, margin=1.0):
    """Scalar contrastive loss for a batch or single pair.

    Args:
        features1, features2: [B, D] embeddings
        labels: [B] with values 1 (same) or 0 (different)
        margin: margin for negative pairs
    Returns: scalar loss (mean over batch)
    """
    euclidean_distance = F.pairwise_distance(features1, features2)
    loss = labels * euclidean_distance.pow(2) + (1 - labels) * (
        torch.clamp(margin - euclidean_distance, min=0.0).pow(2)
    )
    return loss.mean()


def _flatten_grads_from_list(grad_list):
    return torch.cat([g.reshape(-1) for g in grad_list if g is not None])


def _pair_grad(backbone, img1, img2, label01, device):
    """Compute flattened gradient vector for a single pair loss w.r.t. backbone params."""
    backbone.zero_grad(set_to_none=True)
    img1 = img1.unsqueeze(0).to(device, non_blocking=True)
    img2 = img2.unsqueeze(0).to(device, non_blocking=True)
    # ensure label tensor type matches computation dtype
    lbl = torch.as_tensor(label01, device=device, dtype=img1.dtype).view(1)
    f1 = backbone(img1)
    f2 = backbone(img2)
    loss = contrastive_loss(f1, f2, lbl)
    grads = torch.autograd.grad(
        loss,
        [p for p in backbone.parameters() if p.requires_grad],
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    flat = _flatten_grads_from_list(grads).detach().cpu()
    return flat


# ----------------------
# Checkpoint utilities
# ----------------------

def _select_evenly_spaced_checkpoints_generic(filenames, num_select):
    """Support filenames like 'backbone_31614.pth' or '31614backbone.pth'."""
    def extract_iter(f):
        m = re.search(r"(\d+)", f)
        return int(m.group(1)) if m else -1

    files_sorted = sorted(filenames, key=extract_iter)
    if num_select >= len(files_sorted):
        return files_sorted
    idxs = np.linspace(0, len(files_sorted) - 1, num=num_select, dtype=int)
    idxs = sorted(set(map(int, idxs.tolist())))
    while len(idxs) < num_select:
        for i in range(len(files_sorted)):
            if i not in idxs:
                idxs.append(i)
                if len(idxs) == num_select:
                    break
    idxs.sort()
    return [files_sorted[i] for i in idxs]


def load_backbone_weights(backbone, checkpoint_path, device):
    """Load backbone weights from a checkpoint that may store raw state_dict or a dict."""
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and any(
        k.startswith("layer") or k.startswith("conv") or k.startswith("bn") or k == "fc.weight"
        for k in state.keys()
    ):
        backbone.load_state_dict(state)
    elif isinstance(state, dict) and "state_dict" in state:
        backbone.load_state_dict(state["state_dict"])
    else:
        backbone.load_state_dict(state)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(True)


# ----------------------
# Train pairs and gradient caching
# ----------------------

def _pairs_from_batch(images, labels, imageindices, max_pairs_per_batch=32):
    """Return a small, balanced list of (i, j, y01, img_idx_i, img_idx_j)."""
    by_label = defaultdict(list)
    for i, y in enumerate(labels.tolist()):
        by_label[y].append(i)

    pos, neg = [], []
    # positive pairs (consecutive within each label)
    for idxs in by_label.values():
        if len(idxs) > 1:
            for k in range(len(idxs) - 1):
                i, j = idxs[k], idxs[k + 1]
                pos.append((i, j, 1, int(imageindices[i]), int(imageindices[j])))

    # negative pairs (simple cross-label sampling)
    labs = list(by_label.keys())
    for a in range(len(labs)):
        if not by_label[labs[a]]:
            continue
        i = by_label[labs[a]][0]
        b = len(labs) - 1 - a
        if b < 0:
            continue
        if by_label[labs[b]]:
            j = by_label[labs[b]][0]
            if labels[i] != labels[j]:
                neg.append((i, j, 0, int(imageindices[i]), int(imageindices[j])))

    # balance and cap
    pairs = []
    p_needed = max_pairs_per_batch // 2
    n_needed = max_pairs_per_batch - p_needed
    pairs.extend(pos[:p_needed])
    pairs.extend(neg[:n_needed])
    return pairs


def calc_grads_train(
    checkpoints,
    backbone,
    device,
    max_train_pairs_per_ckpt=2000,
    batch_size=64,
    num_workers=4,
):
    """Compute and cache gradient vectors for a subset of train pairs per checkpoint."""
    train_dataset = CasiaWebFace(CASIA_ROOT, local_rank=0, num_classes=CASIA_NUM_CLASSES)
    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    os.makedirs(OUTPUT_GRADS_DIR, exist_ok=True)

    with torch.autograd.set_grad_enabled(True):
        for ci, cp in enumerate(checkpoints):
            ckpt_path = os.path.join(WEIGHTS_DIR, cp) if not os.path.isabs(cp) else cp
            print(f"[{ci+1}/{len(checkpoints)}] Loading checkpoint: {ckpt_path}")
            load_backbone_weights(backbone, ckpt_path, device)

            saved = 0
            for images, labels, imageindices in tqdm(train_dl, desc=f"train-pairs@{cp}"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                pairs = _pairs_from_batch(images, labels, imageindices, max_pairs_per_batch=32)
                for i, j, y01, gi, gj in pairs:
                    grad_vec = _pair_grad(
                        backbone,
                        images[i].detach().cpu(),
                        images[j].detach().cpu(),
                        y01,
                        device,
                    )
                    save_path = os.path.join(
                        OUTPUT_GRADS_DIR, f"{os.path.basename(cp)}__pair_{saved:07d}.pt"
                    )
                    torch.save(
                        {
                            "ckpt": os.path.basename(cp),
                            "train_img_i": gi,
                            "train_img_j": gj,
                            "label01": int(y01),
                            "grad": grad_vec,
                        },
                        save_path,
                    )
                    saved += 1
                    if saved >= max_train_pairs_per_ckpt:
                        break
                if saved >= max_train_pairs_per_ckpt:
                    break


# ----------------------
# TracIn influence computation
# ----------------------

def tracin_influence_for_test_pair(backbone, cp, test_pair, label_bool, device):
    """Compute per-training-image influence for one test pair at one checkpoint.

    Returns: dict {train_image_index: influence_value}
    """
    # Compute test gradient once
    img1, img2 = test_pair
    lbl01 = 1 if bool(label_bool) else 0
    test_grad = _pair_grad(backbone, img1, img2, lbl01, device)

    # Accumulate dots per training image by streaming over saved gradients
    infl = defaultdict(float)
    grad_files = [
        os.path.join(OUTPUT_GRADS_DIR, f)
        for f in os.listdir(OUTPUT_GRADS_DIR)
        if f.startswith(f"{os.path.basename(cp)}__pair_")
    ]
    for gf in grad_files:
        rec = torch.load(gf)
        g = rec["grad"]
        dot = float(torch.dot(test_grad, g))
        # assign to both images in the pair
        infl[rec["train_img_i"]] += dot * 0.5
        infl[rec["train_img_j"]] += dot * 0.5
    return infl


def save_topk_influence(infl_map, out_csv_path, topk=100):
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    # Sort by absolute influence desc
    items = sorted(infl_map.items(), key=lambda kv: abs(kv[1]), reverse=True)[:topk]
    with open(out_csv_path, "w") as f:
        f.write("train_image_index,influence\n")
        for idx, val in items:
            f.write(f"{idx},{val}\n")


# ----------------------
# Main
# ----------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve checkpoints
    all_weights = [
        f
        for f in os.listdir(WEIGHTS_DIR)
        if os.path.isfile(os.path.join(WEIGHTS_DIR, f)) and "backbone" in f and f.endswith(".pth")
    ]
    if not all_weights:
        raise RuntimeError(f"No backbone checkpoints found in {WEIGHTS_DIR}")
    backbone_weights = _select_evenly_spaced_checkpoints_generic(all_weights, 3)

    backbone = iresnet50().to(device)

    # Precompute and cache training pair gradients per checkpoint (pairwise loss, no header)
    need_train_grads = not os.path.exists(OUTPUT_GRADS_DIR) or not os.listdir(OUTPUT_GRADS_DIR)
    if need_train_grads:
        print("No existing gradients found. Calculating training pair gradients...")
        calc_grads_train(backbone_weights, backbone, device)

    # For each validation set and checkpoint, compute per-image TracIn
    for d in VAL_TARGETS:
        test_dataset = BinDataset(os.path.join(VAL_DIR, d + ".bin"))
        test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

        for cp in backbone_weights:
            ckpt_path = os.path.join(WEIGHTS_DIR, cp) if not os.path.isabs(cp) else cp
            print(f"Eval TracIn on {d} @ {ckpt_path}")
            load_backbone_weights(backbone, ckpt_path, device)

            out_dir = os.path.join(OUTPUT_INFL_DIR, d, os.path.basename(cp))
            os.makedirs(out_dir, exist_ok=True)

            for pair_idx, (orig_pair, flipped_pair, label_bool) in enumerate(
                tqdm(test_dl, desc=f"{d}@{cp}")
            ):
                # Use original pair only (optionally average with flipped)
                img1, img2 = orig_pair
                infl_map = tracin_influence_for_test_pair(
                    backbone, cp, (img1[0], img2[0]), label_bool[0].item(), device
                )
                save_topk_influence(
                    infl_map,
                    os.path.join(out_dir, f"pair_{pair_idx:06d}.csv"),
                    topk=100,
                )


if __name__ == "__main__":
    main()
