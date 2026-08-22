"""Hard-Negative candidate clustering & representative extraction tool."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import torch

from wingbeat_ml.sed.models.verifier import Stage2Verifier


def sample_hard_negative_candidates(
    candidates_csv: str | Path,
    atst_checkpoint: str | Path,
    output_csv: str | Path,
    max_per_file: int = 200,
    n_clusters: int = 100,
    pca_components: int = 50,
) -> pd.DataFrame:
    """Cluster Stage-1 candidate proposals and select nearest real candidate to each centroid."""
    df = pd.read_csv(candidates_csv)
    if df.empty:
        return pd.DataFrame()

    # Apply per-file cap to prevent single noisy recording from dominating
    sampled_rows = []
    for _, group in df.groupby("recording_file_id"):
        if len(group) > max_per_file:
            sampled_rows.append(group.sample(n=max_per_file, random_state=42))
        else:
            sampled_rows.append(group)

    capped_df = pd.concat(sampled_rows, ignore_index=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verifier = Stage2Verifier(atst_checkpoint=atst_checkpoint, input_sample_rate=16000).to(device)
    verifier.eval()

    embeddings = []
    valid_indices = []

    # Extract pooled ATST embeddings for each candidate clip
    for idx, row in capped_df.iterrows():
        path = Path(row["audio_path"])
        if not path.is_file():
            continue
        start_s = float(row["start_s"])
        end_s = float(row["end_s"])
        # Crop 2 s candidate window
        mid_s = (start_s + end_s) / 2.0
        crop_start = max(0.0, mid_s - 1.0)

        try:
            from wingbeat_ml.sed.inference.long_recording import load_long_audio_pcm
            signal, sr = load_long_audio_pcm(path, target_sample_rate=16000)
            crop_start_sample = int(crop_start * sr)
            crop_samples = int(2.0 * sr)
            crop_signal = signal[crop_start_sample : crop_start_sample + crop_samples]
            if len(crop_signal) < crop_samples:
                crop_signal = np.pad(crop_signal, (0, crop_samples - len(crop_signal)))

            crop_tensor = torch.from_numpy(crop_signal).float().unsqueeze(0).to(device)
            with torch.no_grad():
                feats = verifier.encoder(crop_tensor)
                pooled = verifier._pool_features(feats).squeeze(0).cpu().numpy()
            embeddings.append(pooled)
            valid_indices.append(idx)
        except Exception:
            continue

    if not embeddings:
        return pd.DataFrame()

    X = np.stack(embeddings)
    valid_df = capped_df.iloc[valid_indices].copy().reset_index(drop=True)

    # PCA reduction + MiniBatchKMeans clustering
    n_components = min(pca_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)

    actual_clusters = min(n_clusters, X_reduced.shape[0])
    kmeans = MiniBatchKMeans(n_clusters=actual_clusters, random_state=42, batch_size=256)
    cluster_labels = kmeans.fit_predict(X_reduced)
    centers = kmeans.cluster_centers_

    # Find nearest real candidate x_k* to each cluster centroid mu_k
    representatives = []
    for k in range(actual_clusters):
        cluster_indices = np.where(cluster_labels == k)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_vecs = X_reduced[cluster_indices]
        dists = np.linalg.norm(cluster_vecs - centers[k], axis=1)
        best_in_cluster = cluster_indices[np.argmin(dists)]
        rep_row = valid_df.iloc[best_in_cluster].to_dict()
        rep_row["cluster_id"] = k
        rep_row["cluster_size"] = len(cluster_indices)
        representatives.append(rep_row)

    out_df = pd.DataFrame(representatives)
    if output_csv:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)

    return out_df
