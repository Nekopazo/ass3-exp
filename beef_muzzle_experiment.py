#!/usr/bin/env python3
import os

# Set thread env before importing numpy/sklearn/torch
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")

import json
import time
import random
import zipfile
import hashlib
import argparse
import urllib.request
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models

from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.random_projection import GaussianRandomProjection
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore", category=FutureWarning)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RGBImageDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        with Image.open(p) as im:
            im = im.convert("RGB")
            x = self.transform(im)
        return x, idx


def download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[skip] zip exists: {dst}")
        return

    print(f"[download] {url} -> {dst}")
    with urllib.request.urlopen(url) as r:
        total = int(r.headers.get("Content-Length", 0))
        with open(dst, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="download") as pbar:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))


def unzip_dataset(zip_path: Path, target_dir: Path) -> None:
    if target_dir.exists() and any(target_dir.glob("cattle_*/*.jpg")):
        print(f"[skip] extracted dataset exists: {target_dir}")
        return

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[unzip] {zip_path} -> {target_dir.parent}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir.parent)


def build_manifest(raw_root: Path) -> pd.DataFrame:
    rows = []
    for id_dir in sorted(raw_root.glob("cattle_*")):
        if not id_dir.is_dir():
            continue
        cattle_id = id_dir.name
        for p in sorted(id_dir.glob("*.jpg")):
            rows.append(
                {
                    "image_path": str(p.resolve()),
                    "cattle_id": cattle_id,
                    "file_name": p.name,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No images found under: {raw_root}")
    df = df.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    return df


def check_image_readable(path: str) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def quality_check(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bad = []
    empty = []

    for p in tqdm(df["image_path"].tolist(), desc="quality-check"):
        if not os.path.exists(p):
            bad.append((p, "missing"))
            continue
        if os.path.getsize(p) == 0:
            empty.append((p, "empty"))
            continue
        if not check_image_readable(p):
            bad.append((p, "corrupted"))

    bad_df = pd.DataFrame(bad, columns=["image_path", "issue"]) if bad else pd.DataFrame(columns=["image_path", "issue"])
    empty_df = pd.DataFrame(empty, columns=["image_path", "issue"]) if empty else pd.DataFrame(columns=["image_path", "issue"])
    issues = pd.concat([bad_df, empty_df], ignore_index=True)

    usable = df[~df["image_path"].isin(issues["image_path"])].copy().reset_index(drop=True)
    assert len(usable) + len(issues) == len(df), "usable + issues != total"
    assert usable["image_path"].nunique() == len(usable), "Duplicate paths in usable set"
    return usable, issues


def build_id_bucket_folds(df: pd.DataFrame, n_folds: int, seed: int):
    rng = np.random.default_rng(seed)
    per_id_parts = {}

    for cid, g in df.groupby("cattle_id"):
        idx = g.index.to_numpy()
        rng.shuffle(idx)
        parts = np.array_split(idx, n_folds)
        per_id_parts[cid] = parts

    folds = []
    for k in range(n_folds):
        test_idx = []
        train_idx = []
        for _, parts in per_id_parts.items():
            for i, part in enumerate(parts):
                if i == k:
                    test_idx.extend(part.tolist())
                else:
                    train_idx.extend(part.tolist())

        folds.append((np.array(sorted(train_idx), dtype=int), np.array(sorted(test_idx), dtype=int)))

    return folds


def get_backbone(name: str, device: torch.device):
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        dim = model.fc.in_features
        model.fc = nn.Identity()
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        dim = model.classifier[-1].in_features
        model.classifier = nn.Identity()
    elif name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        dim = model.classifier[0].in_features
        model.classifier = nn.Identity()
    else:
        raise ValueError(name)

    for p in model.parameters():
        p.requires_grad = False

    model.eval().to(device)
    return model, dim


@torch.no_grad()
def extract_embeddings(paths, backbone_name, device, transform, batch_size: int, num_workers: int):
    model, dim = get_backbone(backbone_name, device)
    ds = RGBImageDataset(paths, transform)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    feats = np.zeros((len(paths), dim), dtype=np.float32)
    for xb, idxb in tqdm(dl, desc=f"embed-{backbone_name}"):
        xb = xb.to(device, non_blocking=True)
        out = model(xb)
        if out.ndim > 2:
            out = torch.flatten(out, 1)
        feats[idxb.numpy()] = out.detach().cpu().numpy().astype(np.float32)

    return feats, dim


def texture_worker(args):
    path, lbp_p, lbp_r, lbp_method, lbp_bins, glcm_distances, glcm_angles, glcm_props = args

    with Image.open(path) as im:
        im = im.convert("RGB")
        im = im.resize((224, 224), resample=Image.Resampling.BICUBIC)
        gray = np.array(im.convert("L"), dtype=np.uint8)

    lbp = local_binary_pattern(gray, P=lbp_p, R=lbp_r, method=lbp_method)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, lbp_bins + 1), range=(0, lbp_bins), density=False)
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist = lbp_hist / max(lbp_hist.sum(), 1.0)

    glcm = graycomatrix(
        gray,
        distances=glcm_distances,
        angles=glcm_angles,
        levels=256,
        symmetric=True,
        normed=True,
    )

    glcm_feats = []
    for prop in glcm_props:
        val = graycoprops(glcm, prop)
        val = val.mean(axis=1)
        glcm_feats.append(val.astype(np.float32))

    glcm_feats = np.concatenate(glcm_feats, axis=0)
    tex = np.concatenate([lbp_hist, glcm_feats], axis=0).astype(np.float32)
    if tex.shape[0] != 34:
        raise RuntimeError(f"Texture dim must be 34, got {tex.shape[0]}")
    return tex


def get_dr_configs(x_train_scaled: np.ndarray, y_train: np.ndarray, pca_ks, rp_ks):
    n_train, n_feat = x_train_scaled.shape
    c = len(np.unique(y_train))
    d_lda = min(c - 1, n_feat, n_train - c)

    # Strict rule: no skipped config is allowed
    lda_base_ks = [1, 8, 16, 32, 64, 128]
    assert d_lda >= max(lda_base_ks), f"LDA not feasible with required ks, d_lda={d_lda}"
    if min(n_train, n_feat) < 256:
        raise RuntimeError(f"PCA k=256 infeasible: min(n_train,n_feat)={min(n_train, n_feat)}")

    configs = [("pca", k) for k in pca_ks]
    configs += [("lda", k) for k in lda_base_ks + [d_lda]]
    configs += [("rp", k) for k in rp_ks]

    # Deduplicate while preserving order.
    dedup = []
    seen = set()
    for method, k in configs:
        key = (method, k)
        if key in seen:
            continue
        seen.add(key)
        dedup.append((method, k))
    return dedup, d_lda


def fit_transform_dr(method, k, x_train, y_train, x_test, seed):
    if method == "none":
        return x_train, x_test
    if method == "pca":
        obj = PCA(n_components=k, random_state=seed)
        return obj.fit_transform(x_train), obj.transform(x_test)
    if method == "lda":
        obj = LinearDiscriminantAnalysis(n_components=k)
        return obj.fit_transform(x_train, y_train), obj.transform(x_test)
    if method == "rp":
        obj = GaussianRandomProjection(n_components=k, random_state=seed)
        return obj.fit_transform(x_train), obj.transform(x_test)
    raise ValueError(method)


def train_predict(clf_name, x_train, y_train, x_test, seed, knn_neighbors, cpu_threads):
    if clf_name == "linearsvc":
        clf = LinearSVC(
            C=1.0,
            loss="squared_hinge",
            penalty="l2",
            dual=False,
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        )
    elif clf_name == "logreg":
        clf = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        )
    elif clf_name == "knn":
        clf = KNeighborsClassifier(
            n_neighbors=knn_neighbors,
            metric="cosine",
            weights="distance",
            algorithm="brute",
            n_jobs=cpu_threads,
        )
    else:
        raise ValueError(clf_name)

    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return clf_name, pred


def run_single_dr_experiment(
    fold_id,
    backbone,
    dr_method,
    k,
    x_train_s,
    x_test_s,
    y_train,
    y_test,
    test_idx,
    test_image_paths,
    d_lda,
    seed,
    knn_neighbors,
    cpu_threads,
    clf_workers,
    artifacts_dir,
    features_dir,
    results_dir,
):
    k_name = str(k)
    x_train_dr, x_test_dr = fit_transform_dr(dr_method, k, x_train_s, y_train, x_test_s, seed)

    expected_dim = int(k)
    assert x_train_dr.shape[1] == expected_dim
    assert x_test_dr.shape[1] == expected_dim

    dr_meta_path = artifacts_dir / "dr" / f"fold{fold_id}_{backbone}_{dr_method}_{k_name}_meta.json"
    dr_meta_path.write_text(
        json.dumps(
            {
                "fold": fold_id,
                "backbone": backbone,
                "dr": dr_method,
                "k": k_name,
                "d_lda": int(d_lda),
                "out_dim": int(expected_dim),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    x_train_nol2 = x_train_dr.astype(np.float32)
    x_test_nol2 = x_test_dr.astype(np.float32)
    x_train_l2 = normalize(x_train_dr, norm="l2").astype(np.float32)
    x_test_l2 = normalize(x_test_dr, norm="l2").astype(np.float32)

    train_norm_l2 = np.linalg.norm(x_train_l2, axis=1)
    assert np.allclose(train_norm_l2, 1.0, atol=1e-4)

    final_base = features_dir / "final" / f"fold{fold_id}_{backbone}_{dr_method}_{k_name}"
    np.save(final_base.with_name(final_base.name + "_train_nol2.npy"), x_train_nol2)
    np.save(final_base.with_name(final_base.name + "_test_nol2.npy"), x_test_nol2)
    np.save(final_base.with_name(final_base.name + "_train_l2.npy"), x_train_l2)
    np.save(final_base.with_name(final_base.name + "_test_l2.npy"), x_test_l2)

    clf_inputs = {
        "linearsvc": (x_train_nol2, x_test_nol2),
        "logreg": (x_train_nol2, x_test_nol2),
        "knn": (x_train_l2, x_test_l2),
    }

    max_workers = min(clf_workers, len(clf_inputs))
    pred_results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {}
        start_ts = {}
        for clf_name, (Xtr, Xte) in clf_inputs.items():
            fut = ex.submit(train_predict, clf_name, Xtr, y_train, Xte, seed, knn_neighbors, cpu_threads)
            fut_map[fut] = clf_name
            start_ts[clf_name] = time.time()

        pending = set(fut_map.keys())
        while pending:
            done, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            if not done:
                running = [fut_map[f] for f in pending]
                tqdm.write(f"[fold{fold_id}-{backbone}-{dr_method}-{k_name}] running: {running}")
                continue

            for fut in done:
                clf_name, pred = fut.result()
                pred_results[clf_name] = pred
                elapsed = time.time() - start_ts[clf_name]
                tqdm.write(f"[fold{fold_id}-{backbone}-{dr_method}-{k_name}] done: {clf_name} ({elapsed:.2f}s)")

    metric_rows = []
    for clf_name in ["linearsvc", "logreg", "knn"]:
        pred = pred_results[clf_name]
        assert len(pred) == len(y_test)
        assert set(np.unique(pred)).issubset(set(np.unique(y_train)))

        test_labels = np.unique(y_test)
        acc = accuracy_score(y_test, pred)
        macro_f1 = f1_score(y_test, pred, average="macro", labels=test_labels)

        pred_df = pd.DataFrame(
            {
                "sample_index": test_idx,
                "image_path": test_image_paths,
                "y_true": y_test,
                "y_pred": pred,
            }
        )
        pred_path = results_dir / "fold_metrics" / f"pred_fold{fold_id}_{backbone}_{dr_method}_{k_name}_{clf_name}.csv"
        pred_df.to_csv(pred_path, index=False)

        metric_rows.append(
            {
                "fold": fold_id,
                "backbone": backbone,
                "feature": "EplusT",
                "dr": dr_method,
                "k": k_name,
                "clf": clf_name,
                "accuracy": float(acc),
                "macro_f1": float(macro_f1),
            }
        )

    # Save per-config metrics immediately for crash-safe progress tracking.
    metric_df = pd.DataFrame(metric_rows)
    metric_path = results_dir / "fold_metrics" / f"metric_fold{fold_id}_{backbone}_{dr_method}_{k_name}.csv"
    metric_df.to_csv(metric_path, index=False)

    return {"dr_method": dr_method, "k_name": k_name, "metric_rows": metric_rows, "jobs_done": 3}


def run_single_fold_experiments(
    fold_id,
    train_idx,
    test_idx,
    usable_manifest,
    backbones,
    embedding_cache,
    texture_all,
    backbone_dims,
    artifacts_dir,
    features_dir,
    results_dir,
    pca_ks,
    rp_ks,
    seed,
    knn_neighbors,
    cpu_threads,
    clf_workers,
    exp_workers,
):
    train_df = usable_manifest.loc[train_idx].reset_index(drop=True)
    test_df = usable_manifest.loc[test_idx].reset_index(drop=True)

    y_train = train_df["cattle_id"].to_numpy()
    y_test = test_df["cattle_id"].to_numpy()

    fold_rows = []
    fold_jobs_done = 0

    for b in backbones:
        e = embedding_cache[b]
        x_train_e = e[train_idx]
        x_test_e = e[test_idx]

        x_train_t = texture_all[train_idx]
        x_test_t = texture_all[test_idx]

        x_train = np.concatenate([x_train_e, x_train_t], axis=1).astype(np.float32)
        x_test = np.concatenate([x_test_e, x_test_t], axis=1).astype(np.float32)

        assert x_train.shape[1] == backbone_dims[b] + 34
        assert x_test.shape[1] == backbone_dims[b] + 34
        assert np.isfinite(x_train).all() and np.isfinite(x_test).all()

        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_train)
        x_test_s = scaler.transform(x_test)

        scaler_meta_path = artifacts_dir / "scaler" / f"fold{fold_id}_{b}_scaler_meta.json"
        scaler_meta_path.write_text(
            json.dumps(
                {
                    "fold": fold_id,
                    "backbone": b,
                    "n_train": int(x_train.shape[0]),
                    "n_test": int(x_test.shape[0]),
                    "n_features": int(x_train.shape[1]),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        dr_configs, d_lda = get_dr_configs(x_train_s, y_train, pca_ks, rp_ks)

        with ThreadPoolExecutor(max_workers=min(exp_workers, len(dr_configs))) as dr_pool:
            fut_map = {}
            for dr_method, k in dr_configs:
                fut = dr_pool.submit(
                    run_single_dr_experiment,
                    fold_id,
                    b,
                    dr_method,
                    k,
                    x_train_s,
                    x_test_s,
                    y_train,
                    y_test,
                    test_idx,
                    test_df["image_path"].values,
                    d_lda,
                    seed,
                    knn_neighbors,
                    cpu_threads,
                    clf_workers,
                    artifacts_dir,
                    features_dir,
                    results_dir,
                )
                fut_map[fut] = (dr_method, k)

            pending = set(fut_map.keys())
            while pending:
                done, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                if not done:
                    running = [f"{m}-{k}" for f in pending for (m, k) in [fut_map[f]]]
                    tqdm.write(f"[fold{fold_id}-{b}] running dr configs: {running}")
                    continue

                for fut in done:
                    res = fut.result()
                    fold_rows.extend(res["metric_rows"])
                    fold_jobs_done += res["jobs_done"]
                    tqdm.write(f"[fold{fold_id}-{b}] done dr: {res['dr_method']}-{res['k_name']}")

    return {"fold_id": fold_id, "metric_rows": fold_rows, "jobs_done": fold_jobs_done}


def main():
    parser = argparse.ArgumentParser(description="Beef muzzle frozen CNN experiment (py script)")
    parser.add_argument("--root", type=str, default=".", help="Project root")
    parser.add_argument("--download", action="store_true", help="Download+unzip dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--texture-workers", type=int, default=16)
    parser.add_argument("--clf-workers", type=int, default=16)
    parser.add_argument("--exp-workers", type=int, default=3, help="Parallel DR experiment workers per fold/backbone")
    parser.add_argument("--fold-workers", type=int, default=4, help="Parallel fold workers")
    parser.add_argument("--cpu-threads", type=int, default=16)
    args = parser.parse_args()

    torch.set_num_threads(args.cpu_threads)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(max(1, min(4, args.cpu_threads)))

    set_seed(args.seed)

    root = Path(args.root).resolve()
    data_dir = root / "data"
    raw_root = data_dir / "raw" / "BeefCattle_Muzzle_Individualized"
    zip_path = data_dir / "BeefCattle_Muzzle_database.zip"

    splits_dir = root / "splits"
    features_dir = root / "features"
    artifacts_dir = root / "artifacts"
    results_dir = root / "results"
    logs_dir = root / "logs"

    for d in [data_dir, splits_dir, features_dir, artifacts_dir, results_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    (features_dir / "embedding").mkdir(parents=True, exist_ok=True)
    (features_dir / "texture").mkdir(parents=True, exist_ok=True)
    (features_dir / "final").mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "scaler").mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "dr").mkdir(parents=True, exist_ok=True)
    (results_dir / "fold_metrics").mkdir(parents=True, exist_ok=True)
    (results_dir / "summary").mkdir(parents=True, exist_ok=True)

    data_url = "https://zenodo.org/records/6324361/files/BeefCattle_Muzzle_database.zip?download=1"

    if args.download:
        download_file(data_url, zip_path)
        unzip_dataset(zip_path, raw_root)

    manifest = build_manifest(raw_root)
    usable_manifest, issues_df = quality_check(manifest)

    id_counts = usable_manifest.groupby("cattle_id").size().sort_values()
    assert id_counts.min() >= 4, f"Found ID with <4 images: min={id_counts.min()}"

    usable_manifest.to_csv(data_dir / "manifest_usable.csv", index=False)
    issues_df.to_csv(data_dir / "manifest_issues.csv", index=False)

    folds_a = build_id_bucket_folds(usable_manifest, args.folds, args.seed)
    folds_b = build_id_bucket_folds(usable_manifest, args.folds, args.seed)

    for fold_id, ((tr_a, te_a), (tr_b, te_b)) in enumerate(zip(folds_a, folds_b), start=1):
        assert np.array_equal(tr_a, tr_b), f"Split reproducibility fail (train) fold {fold_id}"
        assert np.array_equal(te_a, te_b), f"Split reproducibility fail (test) fold {fold_id}"

    all_test_cover = {}
    for fold_id, (train_idx, test_idx) in enumerate(folds_a, start=1):
        train_set = set(train_idx.tolist())
        test_set = set(test_idx.tolist())
        assert train_set.isdisjoint(test_set), f"Train/Test overlap in fold {fold_id}"

        train_df = usable_manifest.loc[train_idx].copy()
        test_df = usable_manifest.loc[test_idx].copy()
        train_df.to_csv(splits_dir / f"fold{fold_id}_train.csv", index=False)
        test_df.to_csv(splits_dir / f"fold{fold_id}_test.csv", index=False)

        for cid, cnt in test_df.groupby("cattle_id").size().items():
            all_test_cover[cid] = all_test_cover.get(cid, 0) + int(cnt)

    id_total = usable_manifest.groupby("cattle_id").size().to_dict()
    for cid, total in id_total.items():
        assert all_test_cover.get(cid, 0) == int(total), f"Coverage mismatch for {cid}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    rgb_transform = T.Compose(
        [
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    backbones = ["resnet50", "efficientnet_b0", "mobilenet_v3_small"]
    all_paths = usable_manifest["image_path"].tolist()
    embedding_cache = {}
    backbone_dims = {}

    for b in backbones:
        feat_path = features_dir / "embedding" / f"{b}_all.npy"
        dim_path = features_dir / "embedding" / f"{b}_dim.json"

        if feat_path.exists() and dim_path.exists():
            emb = np.load(feat_path)
            dim = json.loads(dim_path.read_text())["dim"]
        else:
            emb, dim = extract_embeddings(
                all_paths,
                backbone_name=b,
                device=device,
                transform=rgb_transform,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            np.save(feat_path, emb)
            dim_path.write_text(json.dumps({"dim": int(dim)}, ensure_ascii=False, indent=2))

        assert emb.shape[0] == len(usable_manifest)
        assert emb.shape[1] == dim
        embedding_cache[b] = emb
        backbone_dims[b] = dim

    # Texture extraction (multiprocessing)
    lbp_p, lbp_r, lbp_method, lbp_bins = 8, 1, "uniform", 10
    glcm_distances = [1, 2, 3, 4]
    glcm_angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm_props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]

    tex_path = features_dir / "texture" / "texture_all.npy"
    if tex_path.exists():
        texture_all = np.load(tex_path)
    else:
        tasks = [
            (p, lbp_p, lbp_r, lbp_method, lbp_bins, glcm_distances, glcm_angles, glcm_props)
            for p in all_paths
        ]
        with ProcessPoolExecutor(max_workers=args.texture_workers) as ex:
            feats = list(tqdm(ex.map(texture_worker, tasks, chunksize=32), total=len(tasks), desc=f"texture-mp({args.texture_workers})"))
        texture_all = np.asarray(feats, dtype=np.float32)
        np.save(tex_path, texture_all)

    assert texture_all.shape == (len(usable_manifest), 34)
    assert np.isfinite(texture_all).all()

    pca_ks = [1, 8, 16, 32, 64, 128, 256]
    rp_ks = [1, 8, 16, 32, 64, 128]
    knn_neighbors = 4

    fold_metric_rows = []
    global_job_pbar = tqdm(total=None, desc="all-clf-jobs")
    with ThreadPoolExecutor(max_workers=min(args.fold_workers, args.folds)) as fold_pool:
        fut_map = {}
        for fold_id, (train_idx, test_idx) in enumerate(folds_a, start=1):
            fut = fold_pool.submit(
                run_single_fold_experiments,
                fold_id,
                train_idx,
                test_idx,
                usable_manifest,
                backbones,
                embedding_cache,
                texture_all,
                backbone_dims,
                artifacts_dir,
                features_dir,
                results_dir,
                pca_ks,
                rp_ks,
                args.seed,
                knn_neighbors,
                args.cpu_threads,
                args.clf_workers,
                args.exp_workers,
            )
            fut_map[fut] = fold_id

        pending = set(fut_map.keys())
        while pending:
            done, pending = wait(pending, timeout=20, return_when=FIRST_COMPLETED)
            if not done:
                running = [f"fold{fut_map[f]}" for f in pending]
                tqdm.write(f"[fold-parallel] running: {running}")
                continue

            for fut in done:
                res = fut.result()
                fold_metric_rows.extend(res["metric_rows"])
                global_job_pbar.update(res["jobs_done"])
                tqdm.write(f"[fold-parallel] done: fold{res['fold_id']}")
    global_job_pbar.close()

    fold_metrics = pd.DataFrame(fold_metric_rows)
    fold_metrics_path = results_dir / "fold_metrics" / "fold_metrics_all.csv"
    fold_metrics.to_csv(fold_metrics_path, index=False)

    grp = fold_metrics.groupby(["backbone", "feature", "dr", "k", "clf"]).size()
    assert (grp == args.folds).all(), "Some configs do not have exactly 4 folds"

    summary = (
        fold_metrics.groupby(["backbone", "feature", "dr", "k", "clf"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            n_folds=("fold", "count"),
        )
    )

    summary["accuracy_mean±std"] = summary.apply(lambda r: f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}", axis=1)
    summary["macro_f1_mean±std"] = summary.apply(lambda r: f"{r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f}", axis=1)

    summary_path = results_dir / "summary" / "summary_all.csv"
    summary.to_csv(summary_path, index=False)

    summary_macro = summary.sort_values(["macro_f1_mean", "accuracy_mean"], ascending=[False, False]).reset_index(drop=True)
    summary_acc = summary.sort_values(["accuracy_mean", "macro_f1_mean"], ascending=[False, False]).reset_index(drop=True)
    summary_macro.to_csv(results_dir / "summary" / "summary_sorted_by_macro_f1.csv", index=False)
    summary_acc.to_csv(results_dir / "summary" / "summary_sorted_by_accuracy.csv", index=False)

    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_url": data_url,
        "raw_root": str(raw_root),
        "num_images": int(len(usable_manifest)),
        "num_ids": int(usable_manifest["cattle_id"].nunique()),
        "seed": args.seed,
        "n_folds": args.folds,
        "backbones": backbones,
        "feature": "EplusT",
        "dr": {"none": False, "pca": pca_ks, "lda": "[1,8,16,32,64,128,d_lda]", "rp": rp_ks},
        "classifiers": {
            "linearsvc": {
                "C": 1.0,
                "loss": "squared_hinge",
                "penalty": "l2",
                "dual": False,
                "class_weight": "balanced",
                "max_iter": 5000,
                "random_state": args.seed,
            },
            "logreg": {
                "penalty": "l2",
                "C": 1.0,
                "solver": "lbfgs",
                "multi_class": "multinomial",
                "class_weight": "balanced",
                "max_iter": 1000,
                "random_state": args.seed,
            },
            "knn": {
                "n_neighbors": 4,
                "metric": "cosine",
                "weights": "distance",
                "algorithm": "brute",
                "n_jobs": args.cpu_threads,
            },
        },
        "macro_f1_labels": "labels appearing in current fold test set",
        "texture_workers": args.texture_workers,
        "fold_workers": args.fold_workers,
        "exp_workers": args.exp_workers,
        "clf_parallel_workers": args.clf_workers,
        "cpu_threads": args.cpu_threads,
    }

    (logs_dir / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    print(f"Saved: {fold_metrics_path}")
    print(f"Saved: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
