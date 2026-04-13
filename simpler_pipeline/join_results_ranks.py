import os
import torch
import numpy as np

results_dir = "simpler_pipeline/results/ExperimentsClassDeno/Experiments4/30_angles_sigma_0.3_1e-1_5e-2"
name =  "30_angles_sigma_0.3_1e-1_5e-2_10batch"


def load_and_merge_results(results_dir, name, world_size=2):
    merged = {}

    for rank in range(world_size):
        path = os.path.join(results_dir, f"results_mnist_ct_{name}_rank{rank}.pt")
        r = torch.load(path, map_location="cpu")

        for k, v in r.items():
            if k not in merged:
                if isinstance(v, list):
                    merged[k] = []
                else:
                    merged[k] = v
            if isinstance(v, list):
                merged[k].extend(v)

    return merged

def summarize_metric(metric_list):
    """
    metric_list: lista de longitud N_batches
                 cada elemento es una lista de longitud N_iters
    Devuelve media y std por iteración.
    """
    arr = np.array(metric_list, dtype=np.float64)   # [N_batches, N_iters]
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    return {
        "all": arr,
        "mean": mean,
        "std": std,
        "mean_final": arr[:, -1].mean(),
        "std_final": arr[:, -1].std(),
    }

def summarize_digit_distributions(prob_list, true_digits):
    """
    prob_list: lista de N muestras, cada una un tensor [10]
    true_digits: lista de N enteros
    """
    top1 = []
    top2 = []
    top3 = []
    true_prob = []
    true_rank = []
    nll = []
    entropy = []
    margin12 = []

    for i in range(len(prob_list)):
        p = prob_list[i]
        if not isinstance(p, torch.Tensor):
            p = torch.tensor(p, dtype=torch.float32)
        p = p.float().flatten()

        y = int(true_digits[i])

        vals, idxs = torch.sort(p, descending=True)

        pos = (idxs == y).nonzero(as_tuple=False)
        if pos.numel() == 0:
            raise ValueError(f"True digit {y} not found in prediction indices for sample {i}. Shape of p: {p.shape}")

        rank = pos[0].item() + 1
        tp = p[y].item()
        ent = -(p * torch.log(p.clamp_min(1e-12))).sum().item()
        nll_val = -np.log(max(tp, 1e-12))
        m12 = (vals[0] - vals[1]).item() if len(vals) > 1 else float("nan")

        top1.append(1.0 if rank <= 1 else 0.0)
        top2.append(1.0 if rank <= 2 else 0.0)
        top3.append(1.0 if rank <= 3 else 0.0)
        true_prob.append(tp)
        true_rank.append(rank)
        nll.append(nll_val)
        entropy.append(ent)
        margin12.append(m12)

    return {
        "top1_mean": np.mean(top1),
        "top2_mean": np.mean(top2),
        "top3_mean": np.mean(top3),
        "true_prob_mean": np.mean(true_prob),
        "true_rank_mean": np.mean(true_rank),
        "nll_mean": np.mean(nll),
        "entropy_mean": np.mean(entropy),
        "margin12_mean": np.mean(margin12),
        "top1_all": np.array(top1),
        "top2_all": np.array(top2),
        "top3_all": np.array(top3),
        "true_prob_all": np.array(true_prob),
        "true_rank_all": np.array(true_rank),
        "nll_all": np.array(nll),
        "entropy_all": np.array(entropy),
        "margin12_all": np.array(margin12),
    }

def summarize_center_distributions(center_list, true_centers, ymin=14, ymax=85, xmin=14, xmax=85, radii=(1,2,3,5)):
    """
    center_list: lista de N muestras, cada una tensor [H, W]
    true_centers: lista de N muestras con (cy, cx)
    """
    prob_true = []
    argmax_acc = []
    entropy = []
    spread_true = []
    spread_mean = []
    mass_in_radius = {r: [] for r in radii}

    for i in range(len(center_list)):
        p = center_list[i]
        if not isinstance(p, torch.Tensor):
            p = torch.tensor(p, dtype=torch.float32)
        p = p.float()

        cy_true, cx_true = true_centers[i]
        cy_true = int(cy_true)
        cx_true = int(cx_true)

        H, W = p.shape
        ys = torch.arange(H).view(H, 1).float()
        xs = torch.arange(W).view(1, W).float()

        prob_true.append(p[cy_true, cx_true].item())

        flat_idx = torch.argmax(p).item()
        cy_hat = flat_idx // W
        cx_hat = flat_idx % W
        argmax_acc.append(1.0 if (cy_hat == cy_true and cx_hat == cx_true) else 0.0)

        entropy.append((-(p * torch.log(p.clamp_min(1e-12))).sum()).item())

        dist_true = torch.sqrt((ys - cy_true)**2 + (xs - cx_true)**2)
        spread_true.append((p * dist_true).sum().item())

        Ey = (p * ys).sum().item()
        Ex = (p * xs).sum().item()
        dist_mean = torch.sqrt((ys - Ey)**2 + (xs - Ex)**2)
        spread_mean.append((p * dist_mean).sum().item())

        for r in radii:
            mask = ((ys - cy_true)**2 + (xs - cx_true)**2) <= (r**2)
            mass_in_radius[r].append(p[mask].sum().item())

    out = {
        "prob_true_mean": np.mean(prob_true),
        "argmax_acc_mean": np.mean(argmax_acc),
        "entropy_mean": np.mean(entropy),
        "spread_true_mean": np.mean(spread_true),
        "spread_mean_mean": np.mean(spread_mean),
        "prob_true_all": np.array(prob_true),
        "argmax_acc_all": np.array(argmax_acc),
        "entropy_all": np.array(entropy),
        "spread_true_all": np.array(spread_true),
        "spread_mean_all": np.array(spread_mean),
    }
    for r in radii:
        out[f"mass_r{r}_mean"] = np.mean(mass_in_radius[r])
        out[f"mass_r{r}_all"] = np.array(mass_in_radius[r])

    return out

results = load_and_merge_results(results_dir, name, world_size=1)

summary = {}

metric_keys = [
    "both_psnr", "sirt_psnr", "clas_psnr", "deno_psnr",
    "both_ssim", "sirt_ssim", "clas_ssim", "deno_ssim",
    "both_psnr_crop", "sirt_psnr_crop", "clas_psnr_crop", "deno_psnr_crop",
    "both_ssim_crop", "sirt_ssim_crop", "clas_ssim_crop", "deno_ssim_crop",
]

for key in metric_keys:
    summary[key] = summarize_metric(results[key])

print("=== RECONSTRUCTION METRICS ===")

for key in metric_keys:
    print(f"{key}:")
    print(f"  final mean = {summary[key]['mean_final']:.4f}")
    print(f"  final std  = {summary[key]['std_final']:.4f}")

true_digits = []
for x in results["gt_digits"]:
    if isinstance(x, torch.Tensor):
        true_digits.append(int(x.item()))
    else:
        true_digits.append(int(x))

digit_summary = {}

digit_summary["both"] = summarize_digit_distributions(results["both_digits"], true_digits)
digit_summary["sirt"] = summarize_digit_distributions(results["sirt_digits"], true_digits)

print("\n=== DIGIT METRICS ===")
for mode in ["both", "sirt"]:
    s = digit_summary[mode]
    print(f"{mode}:")
    print(f"  top1 acc      = {s['top1_mean']:.4f}")
    print(f"  top2 acc      = {s['top2_mean']:.4f}")
    print(f"  top3 acc      = {s['top3_mean']:.4f}")
    print(f"  true prob     = {s['true_prob_mean']:.4f}")
    print(f"  true rank     = {s['true_rank_mean']:.4f}")
    print(f"  nll           = {s['nll_mean']:.4f}")
    print(f"  entropy       = {s['entropy_mean']:.4f}")
    print(f"  margin12      = {s['margin12_mean']:.4f}")

true_centers = []
for c in results["gt_centers"]:
    if isinstance(c, torch.Tensor):
        true_centers.append((int(c[0].item()), int(c[1].item())))
    else:
        true_centers.append((int(c[0]), int(c[1])))

center_summary = {}

center_summary["both"] = summarize_center_distributions(results["both_centers"], true_centers)
center_summary["sirt"] = summarize_center_distributions(results["sirt_centers"], true_centers)

print("\n=== CENTER METRICS ===")
for mode in ["both", "sirt"]:
    s = center_summary[mode]
    print(f"{mode}:")
    print(f"  prob_true     = {s['prob_true_mean']:.6f}")
    print(f"  argmax_acc    = {s['argmax_acc_mean']:.4f}")
    print(f"  entropy       = {s['entropy_mean']:.4f}")
    print(f"  spread_true   = {s['spread_true_mean']:.4f}")
    print(f"  spread_mean   = {s['spread_mean_mean']:.4f}")
    print(f"  mass_r1       = {s['mass_r1_mean']:.6f}")
    print(f"  mass_r2       = {s['mass_r2_mean']:.6f}")
    print(f"  mass_r3       = {s['mass_r3_mean']:.6f}")
    print(f"  mass_r5       = {s['mass_r5_mean']:.6f}")