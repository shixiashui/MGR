"""Versioned, train-only feature and sparse graph index for SARAF retrieval."""

import hashlib
import json
import os
import tempfile
import time
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F


SCHEMA_VERSION = 1


def _feature_batch(windows, spectral_bins=32):
    """Return compact frequency and distribution-statistics descriptors."""
    x = windows.float()
    eps = 1e-6

    mean = x.mean(dim=1)
    centered = x - mean.unsqueeze(1)
    std = centered.square().mean(dim=1).sqrt().clamp_min(eps)
    z = centered / std.unsqueeze(1)
    metrics = [
        mean,
        std,
        z.pow(3).mean(dim=1),
        z.pow(4).mean(dim=1) - 3.0,
        (x.amax(dim=1) - x.amin(dim=1)) / std,
        (z[:, 1:, :] * z[:, :-1, :]).mean(dim=1),
        (centered * torch.linspace(-1.0, 1.0, x.shape[1], device=x.device).view(1, -1, 1)).mean(dim=1) / std,
    ]
    stats_by_channel = torch.stack(metrics, dim=1)  # B, metrics, C
    stats = torch.cat(
        [stats_by_channel.mean(dim=2), stats_by_channel.std(dim=2, unbiased=False)],
        dim=1,
    )

    spectrum = torch.fft.rfft(x, dim=1).abs().square()  # B, F, C
    spectrum = spectrum / spectrum.sum(dim=1, keepdim=True).clamp_min(eps)
    avg_power = spectrum.mean(dim=2)
    channel_spread = spectrum.std(dim=2, unbiased=False)
    avg_power = F.adaptive_avg_pool1d(avg_power.unsqueeze(1), spectral_bins).squeeze(1)
    channel_spread = F.adaptive_avg_pool1d(channel_spread.unsqueeze(1), spectral_bins).squeeze(1)
    freq = torch.cat([avg_power, channel_spread], dim=1)
    return freq, stats


def _batched_features(windows, batch_size=256):
    freq_parts, stat_parts = [], []
    for start in range(0, len(windows), batch_size):
        freq, stats = _feature_batch(windows[start:start + batch_size])
        freq_parts.append(freq.cpu())
        stat_parts.append(stats.cpu())
    return torch.cat(freq_parts), torch.cat(stat_parts)


def _fit_normalizer(features):
    mean = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False).clamp_min(1e-5)
    return mean, scale


def _transform(features, mean, scale):
    return F.normalize((features - mean) / scale, dim=1, eps=1e-8)


def _build_knn(features, k=16, block_size=256, backend='auto'):
    """Build a sparse cosine kNN graph, using HNSW when requested and available."""
    vectors = F.normalize(features.float(), dim=1, eps=1e-8).cpu()
    n = vectors.shape[0]
    k = min(int(k), max(n - 1, 1))
    if backend not in ('auto', 'exact', 'hnsw'):
        raise ValueError('graph backend must be auto, exact, or hnsw')
    use_hnsw = backend == 'hnsw' or (backend == 'auto' and n >= 5000)
    if use_hnsw:
        try:
            import hnswlib
        except ImportError:
            if backend == 'hnsw':
                raise RuntimeError("graph_backend='hnsw' requires the optional hnswlib package")
            use_hnsw = False
            if n >= 5000:
                warnings.warn(
                    'hnswlib is unavailable; building an exact graph in blocks. '
                    'Install hnswlib or pass --graph_backend exact for an explicit exact build.',
                    RuntimeWarning,
                )
    if use_hnsw:
        index = hnswlib.Index(space='cosine', dim=vectors.shape[1])
        index.init_index(max_elements=n, ef_construction=120, M=16, random_seed=42)
        index.add_items(vectors.numpy(), num_threads=-1)
        index.set_ef(max(64, 4 * k))
        labels, distances = index.knn_query(vectors.numpy(), k=min(k + 1, n), num_threads=-1)
        neighbors, weights = [], []
        for node_id, (row_labels, row_distances) in enumerate(zip(labels, distances)):
            selected = [(int(label), 1.0 - float(distance))
                        for label, distance in zip(row_labels, row_distances)
                        if int(label) != node_id][:k]
            if len(selected) < k:
                selected.extend([(node_id, 0.0)] * (k - len(selected)))
            neighbors.append([item[0] for item in selected])
            weights.append([item[1] for item in selected])
        neighbors = torch.tensor(neighbors, dtype=torch.int32)
        weights = torch.tensor(weights, dtype=torch.float16)
        sample = torch.linspace(0, n - 1, min(256, n), dtype=torch.long)
        exact = vectors[sample] @ vectors.T
        exact[torch.arange(len(sample)), sample] = -float('inf')
        exact_ids = torch.topk(exact, k=k, dim=1).indices
        approximate_ids = neighbors[sample].long()
        matches = (exact_ids.unsqueeze(2) == approximate_ids.unsqueeze(1)).any(dim=2)
        recall = matches.float().sum().item() / (len(sample) * k)
        return neighbors, weights, recall

    all_indices, all_weights = [], []
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        scores = vectors[start:end] @ vectors.T
        rows = torch.arange(end - start)
        scores[rows, torch.arange(start, end)] = -float('inf')
        weights, indices = torch.topk(scores, k=k, dim=1)
        all_indices.append(indices.to(torch.int32))
        all_weights.append(weights.clamp_min(0).to(torch.float16))
    return torch.cat(all_indices), torch.cat(all_weights), 1.0


def _source_fingerprint(dataset, windows, seq_len, pred_len, graph_k, spectral_bins):
    source = os.path.abspath(os.path.join(dataset.root_path, dataset.data_path))
    try:
        st = os.stat(source)
        file_hash = hashlib.sha256()
        with open(source, 'rb') as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                file_hash.update(chunk)
        file_meta = {
            'source': source,
            'size': st.st_size,
            'mtime_ns': st.st_mtime_ns,
            'sha256': file_hash.hexdigest(),
        }
    except OSError:
        file_meta = {'source': source, 'size': None, 'mtime_ns': None}
    identity = {
        'schema': SCHEMA_VERSION,
        'file': file_meta,
        'dataset': getattr(dataset.args, 'data', None),
        'frequency': getattr(dataset, 'freq', None),
        'features': getattr(dataset, 'features', None),
        'target': getattr(dataset, 'target', None),
        'scale': getattr(dataset, 'scale', None),
        'augmentation_ratio': getattr(dataset.args, 'augmentation_ratio', 0),
        'seed': getattr(dataset.args, 'seed', 2021),
        'windows': list(windows.shape),
        'seq_len': int(seq_len),
        'pred_len': int(pred_len),
        'graph_k': int(graph_k),
        'spectral_bins': int(spectral_bins),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode('utf-8')).hexdigest()[:24]
    return identity, digest


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=str(path.parent), suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _atomic_torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=str(path.parent), suffix='.tmp')
    os.close(fd)
    try:
        torch.save(value, temp_path)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def load_or_build_index(dataset, windows, seq_len, pred_len, cache_root,
                        graph_k=16, spectral_bins=32, graph_backend='auto', build_graph=True):
    """Persist train-only views and typed graph edges; return their cache path."""
    identity, digest = _source_fingerprint(dataset, windows, seq_len, pred_len, graph_k, spectral_bins)
    if not build_graph:
        effective_backend = 'none'
    elif graph_backend == 'auto':
        effective_backend = 'hnsw' if len(windows) >= 5000 and _has_hnswlib() else 'exact'
        if len(windows) >= 5000 and effective_backend == 'exact':
            warnings.warn(
                'hnswlib is unavailable; building an exact graph in blocks. '
                'Install hnswlib or pass --graph_backend exact for an explicit exact build.',
                RuntimeWarning,
            )
    else:
        effective_backend = graph_backend
    identity['graph_backend'] = effective_backend
    identity['build_graph'] = bool(build_graph)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode('utf-8')).hexdigest()[:24]
    root = Path(cache_root) / digest
    manifest_path = root / 'manifest.json'
    features_path = root / 'features.pt'
    graph_path = root / 'graph.pt'
    if manifest_path.exists() and features_path.exists() and graph_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if manifest.get('identity') == identity:
                return {
                    'path': str(root),
                    'features': torch.load(features_path, map_location='cpu', weights_only=True),
                    'graph': torch.load(graph_path, map_location='cpu', weights_only=True),
                    'manifest': manifest,
                }
        except Exception as exc:
            warnings.warn(f'Ignoring an invalid SARAF retrieval cache at {root}: {exc}', RuntimeWarning)

    build_started = time.perf_counter()
    freq_raw, stats_raw = _batched_features(windows, batch_size=256)
    freq_mean, freq_scale = _fit_normalizer(freq_raw)
    stats_mean, stats_scale = _fit_normalizer(stats_raw)
    freq = _transform(freq_raw, freq_mean, freq_scale)
    stats = _transform(stats_raw, stats_mean, stats_scale)
    if build_graph:
        freq_edges, freq_weights, freq_recall = _build_knn(freq, k=graph_k, backend=effective_backend)
        stat_edges, stat_weights, stat_recall = _build_knn(stats, k=graph_k, backend=effective_backend)
    else:
        freq_edges = torch.empty((len(windows), 0), dtype=torch.int32)
        stat_edges = torch.empty((len(windows), 0), dtype=torch.int32)
        freq_weights = torch.empty((len(windows), 0), dtype=torch.float16)
        stat_weights = torch.empty((len(windows), 0), dtype=torch.float16)
        freq_recall = stat_recall = 1.0

    features = {
        'frequency': freq,
        'statistics': stats,
        'frequency_mean': freq_mean,
        'frequency_scale': freq_scale,
        'statistics_mean': stats_mean,
        'statistics_scale': stats_scale,
    }
    graph = {
        'frequency_indices': freq_edges,
        'frequency_weights': freq_weights,
        'statistics_indices': stat_edges,
        'statistics_weights': stat_weights,
    }
    source = identity['file']['source']
    records = []
    raw_window_count = int(getattr(dataset, 'raw_window_count', len(windows)))
    for i in range(len(windows)):
        is_source_window = i < raw_window_count
        raw_start = int(getattr(dataset, 'data_offset', 0)) + i if is_source_window else None
        node_key = f"{identity['file']['source']}|train|{raw_start if is_source_window else 'aug-' + str(i)}|{seq_len}|{pred_len}"
        node_id = hashlib.sha256(node_key.encode('utf-8')).hexdigest()[:20]
        records.append({
            'window_id': node_id,
            'split': 'train',
            'source_file': source,
            'raw_start': raw_start,
            'raw_input_end': raw_start + int(seq_len) if is_source_window else None,
            'raw_target_end': raw_start + int(seq_len) + int(pred_len) if is_source_window else None,
            'synthetic_augmentation': not is_source_window,
            'scaled': bool(getattr(dataset, 'scale', True)),
        })
    manifest = {
        'schema_version': SCHEMA_VERSION,
        'identity': identity,
        'n_windows': len(windows),
        'window_length': int(seq_len),
        'prediction_length': int(pred_len),
        'raw_view': 'source_file plus raw_start/raw_input_end/raw_target_end references; raw values remain in the source file',
        'derived_views': ['frequency', 'statistics'],
        'edge_types': ['frequency_knn', 'statistics_knn'] if build_graph else [],
        'edge_k': int(graph_k),
        'graph_backend': ('hnswlib' if effective_backend == 'hnsw' else 'exact_blocked')
            if build_graph else 'not_built',
        'sampled_graph_recall_at_k': (freq_recall + stat_recall) / 2.0,
        'index_build_seconds': time.perf_counter() - build_started,
        'scaler_mean': getattr(getattr(dataset, 'scaler', None), 'mean_', None).tolist()
            if getattr(getattr(dataset, 'scaler', None), 'mean_', None) is not None else None,
        'scaler_scale': getattr(getattr(dataset, 'scaler', None), 'scale_', None).tolist()
            if getattr(getattr(dataset, 'scaler', None), 'scale_', None) is not None else None,
        'channel_names': getattr(dataset, 'channel_names', None),
        'windows': records,
    }
    _atomic_torch_save(features_path, features)
    _atomic_torch_save(graph_path, graph)
    manifest['cache_bytes'] = (
        os.path.getsize(features_path)
        + os.path.getsize(graph_path)
        + len(json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8'))
    )
    _atomic_json(manifest_path, manifest)
    return {'path': str(root), 'features': features, 'graph': graph, 'manifest': manifest}


def _has_hnswlib():
    try:
        import hnswlib  # noqa: F401
        return True
    except ImportError:
        return False


def transform_query(windows, features, batch_size=256):
    freq_parts, stat_parts = [], []
    for start in range(0, len(windows), batch_size):
        freq, stats = _feature_batch(windows[start:start + batch_size])
        freq_parts.append(freq.cpu())
        stat_parts.append(stats.cpu())
    freq = torch.cat(freq_parts).to(features['frequency_mean'].device)
    stats = torch.cat(stat_parts).to(features['statistics_mean'].device)
    freq = _transform(freq, features['frequency_mean'], features['frequency_scale'])
    stats = _transform(stats, features['statistics_mean'], features['statistics_scale'])
    return freq, stats
