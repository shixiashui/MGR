import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader


class RetrievalTool():
    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        n_period=3,
        temperature=0.1,
        topm=20,
        with_dec=False,
        return_key=False,
        use_multiview=False,
        use_graph=False,
        n_fft=8,
        graph_hop=1,
    ):
        period_num = [16, 8, 4, 2, 1]
        period_num = period_num[-1 * n_period:]

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels

        self.n_period = n_period
        self.period_num = sorted(period_num, reverse=True)

        self.temperature = temperature
        self.topm = topm

        self.with_dec = with_dec
        self.return_key = return_key

        # Multi-view params
        self.use_multiview = use_multiview
        self.use_graph = use_graph
        self.n_fft = n_fft
        self.graph_hop = graph_hop

    def prepare_dataset(self, train_data):
        train_data_all = []
        y_data_all = []

        for i in range(len(train_data)):
            td = train_data[i]
            train_data_all.append(td[1])

            if self.with_dec:
                y_data_all.append(td[2][-(train_data.pred_len + train_data.label_len):])
            else:
                y_data_all.append(td[2][-train_data.pred_len:])

        self.train_data_all = torch.tensor(np.stack(train_data_all, axis=0)).float()
        self.train_data_all_mg, _ = self.decompose_mg(self.train_data_all)

        self.y_data_all = torch.tensor(np.stack(y_data_all, axis=0)).float()
        self.y_data_all_mg, _ = self.decompose_mg(self.y_data_all)

        self.n_train = self.train_data_all.shape[0]

        # Multi-view: frequency-domain and statistical features
        if self.use_multiview:
            self.freq_mem = self._compute_freq_view(self.train_data_all_mg)
            self.stat_mem = self._compute_stat_view(self.train_data_all_mg)

        # Graph: build temporal proximity adjacency
        if self.use_graph:
            self._build_temporal_graph(self.n_train)

    def _compute_freq_view(self, data_mg):
        """
        Compute frequency-domain view via FFT on period-decomposed data.
        data_mg: (G, T, S, C) where S = seq_len
        Returns: (G, T, n_fft, C)
        """
        freq = torch.fft.rfft(data_mg, dim=2)  # (G, T, S//2+1, C)
        freq_amp = freq.abs()  # (G, T, S//2+1, C)
        freq_amp = freq_amp[:, :, :self.n_fft, :]  # (G, T, n_fft, C)
        return freq_amp

    def _compute_stat_view(self, data_mg):
        """
        Compute statistical view: mean, variance, autocorrelation.
        data_mg: (G, T, S, C) where S = seq_len
        Returns: (G, T, 3, C)
        """
        mean_val = data_mg.mean(dim=2)  # (G, T, C)
        var_val = data_mg.var(dim=2)  # (G, T, C)
        centered = data_mg - mean_val.unsqueeze(2)  # (G, T, S, C)
        autocorr = (centered[:, :, :-1, :] * centered[:, :, 1:, :]).mean(dim=2)
        autocorr = autocorr / (var_val + 1e-8)  # (G, T, C)
        stat = torch.stack([mean_val, var_val, autocorr], dim=2)  # (G, T, 3, C)
        return stat

    def _build_temporal_graph(self, n_samples):
        """
        Build temporal proximity graph adjacency.
        Each sample connects to its neighbors within graph_hop distance.
        """
        self.adjacency = []
        for i in range(n_samples):
            neighbors = []
            for h in range(1, self.graph_hop + 1):
                if i - h >= 0:
                    neighbors.append(i - h)
                if i + h < n_samples:
                    neighbors.append(i + h)
            self.adjacency.append(torch.tensor(neighbors, dtype=torch.long))

    def _graph_expand(self, topm_index):
        """
        Expand top-M candidates using temporal graph neighbors.
        topm_index: (G * B, M)
        Returns: (G * B, K) expanded candidates
        """
        rows, _ = topm_index.shape
        expanded = []

        for r in range(rows):
            candidates = topm_index[r]
            expanded_set = set(candidates.tolist())
            for c in candidates:
                c_idx = c.item()
                if c_idx < len(self.adjacency):
                    for nb in self.adjacency[c_idx]:
                        expanded_set.add(nb.item())
            expanded.append(torch.tensor(list(expanded_set), dtype=torch.long))

        max_len = max(len(e) for e in expanded)
        padded = torch.zeros(rows, max_len, dtype=torch.long, device=topm_index.device)
        for r, e in enumerate(expanded):
            padded[r, :len(e)] = e.to(topm_index.device)

        return padded

    def decompose_mg(self, data_all, remove_offset=True):
        data_all = copy.deepcopy(data_all)  # T, S, C

        mg = []
        for g in self.period_num:
            cur = data_all.unfold(dimension=1, size=g, step=g).mean(dim=-1)
            cur = cur.repeat_interleave(repeats=g, dim=1)

            mg.append(cur)

        mg = torch.stack(mg, dim=0)  # G, T, S, C

        if remove_offset:
            offset = []
            for i, data_p in enumerate(mg):
                cur_offset = data_p[:, -1:, :]
                mg[i] = data_p - cur_offset
                offset.append(cur_offset)
        else:
            offset = None

        offset = torch.stack(offset, dim=0)

        return mg, offset

    def periodic_batch_corr(self, data_all, key, in_bsz=512):
        _, bsz, features = key.shape
        _, train_len, _ = data_all.shape

        bx = key - torch.mean(key, dim=2, keepdim=True)

        iters = math.ceil(train_len / in_bsz)

        sim = []
        for i in range(iters):
            start_idx = i * in_bsz
            end_idx = min((i + 1) * in_bsz, train_len)

            cur_data = data_all[:, start_idx:end_idx].to(key.device)
            ax = cur_data - torch.mean(cur_data, dim=2, keepdim=True)

            cur_sim = torch.bmm(F.normalize(bx, dim=2), F.normalize(ax, dim=2).transpose(-1, -2))
            sim.append(cur_sim)

        sim = torch.cat(sim, dim=2)

        return sim

    def _view_retrieval(self, sim, bsz, channels):
        """
        Perform retrieval for a single view given similarity scores.
        sim: (G, B, T)
        Returns: pred_from_retrieval (G, B, P, C)
        """
        sim = sim.reshape(self.n_period * bsz, self.n_train)  # G X B, T

        # Graph expansion
        if self.use_graph:
            topm_index = torch.topk(sim, self.topm, dim=1).indices
            expanded_index = self._graph_expand(topm_index)

            rows = torch.arange(expanded_index.size(0)).unsqueeze(-1).to(sim.device)
            expanded_scores = sim[rows, expanded_index]

            if expanded_index.size(1) > self.topm:
                topm_indices = torch.topk(expanded_scores, self.topm, dim=1).indices
                rows_idx = torch.arange(expanded_index.size(0)).unsqueeze(-1).to(sim.device)
                final_topm_index = expanded_index[rows_idx, topm_indices]
                final_scores = expanded_scores[rows_idx, topm_indices]
            else:
                final_topm_index = expanded_index
                final_scores = expanded_scores
        else:
            final_topm_index = torch.topk(sim, self.topm, dim=1).indices
            final_scores = None

        ranking_sim = torch.ones_like(sim) * float('-inf')

        rows = torch.arange(sim.size(0)).unsqueeze(-1).to(sim.device)
        if final_scores is not None:
            ranking_sim[rows, final_topm_index] = final_scores
        else:
            ranking_sim[rows, final_topm_index] = sim[rows, final_topm_index]

        ranking_sim = ranking_sim.reshape(self.n_period, bsz, self.n_train)  # G, B, T

        ranking_prob = F.softmax(ranking_sim / self.temperature, dim=2)
        ranking_prob = ranking_prob.detach().cpu()  # G, B, T

        y_data_all = self.y_data_all_mg.flatten(start_dim=2)  # G, T, P * C

        pred = torch.bmm(ranking_prob, y_data_all).reshape(self.n_period, bsz, -1, channels)
        return pred

    def retrieve(self, x, index, train=True):
        index = index.to(x.device)

        bsz, seq_len, channels = x.shape
        assert (seq_len == self.seq_len, channels == self.channels)

        x_mg, mg_offset = self.decompose_mg(x)  # G, B, S, C

        # --- View 1: Time-domain ---
        sim_time = self.periodic_batch_corr(
            self.train_data_all_mg.flatten(start_dim=2),  # G, T, S * C
            x_mg.flatten(start_dim=2),  # G, B, S * C
        )  # G, B, T

        # --- View 2: Frequency-domain ---
        if self.use_multiview:
            x_freq = self._compute_freq_view(x_mg)  # G, B, n_fft, C
            sim_freq = self.periodic_batch_corr(
                self.freq_mem.flatten(start_dim=2),  # G, T, n_fft * C
                x_freq.flatten(start_dim=2),  # G, B, n_fft * C
            )  # G, B, T

            # --- View 3: Statistical ---
            x_stat = self._compute_stat_view(x_mg)  # G, B, 3, C
            sim_stat = self.periodic_batch_corr(
                self.stat_mem.flatten(start_dim=2),  # G, T, 3 * C
                x_stat.flatten(start_dim=2),  # G, B, 3 * C
            )  # G, B, T
        else:
            sim_freq = None
            sim_stat = None

        # --- Apply training mask ---
        if train:
            sliding_index = torch.arange(2 * (self.seq_len + self.pred_len) - 1).to(x.device)
            sliding_index = sliding_index.unsqueeze(dim=0).repeat(len(index), 1)
            sliding_index = sliding_index + (index - self.seq_len - self.pred_len + 1).unsqueeze(dim=1)

            sliding_index = torch.where(sliding_index >= 0, sliding_index, 0)
            sliding_index = torch.where(sliding_index < self.n_train, sliding_index, self.n_train - 1)

            self_mask = torch.zeros((bsz, self.n_train)).to(x.device)
            self_mask = self_mask.scatter_(1, sliding_index, 1.)
            self_mask = self_mask.unsqueeze(dim=0).repeat(self.n_period, 1, 1)

            sim_time = sim_time.masked_fill_(self_mask.bool(), float('-inf'))
            if sim_freq is not None:
                sim_freq = sim_freq.masked_fill_(self_mask.bool(), float('-inf'))
            if sim_stat is not None:
                sim_stat = sim_stat.masked_fill_(self_mask.bool(), float('-inf'))

        # --- Per-view retrieval with graph expansion ---
        pred_time = self._view_retrieval(sim_time, bsz, channels)

        if self.use_multiview:
            pred_freq = self._view_retrieval(sim_freq, bsz, channels)
            pred_stat = self._view_retrieval(sim_stat, bsz, channels)
        else:
            pred_freq = None
            pred_stat = None

        result = {'time': pred_time.to(x.device)}
        if pred_freq is not None:
            result['freq'] = pred_freq.to(x.device)
        if pred_stat is not None:
            result['stat'] = pred_stat.to(x.device)

        return result

    def retrieve_all(self, data, train=False, device=torch.device('cpu')):
        assert (self.train_data_all_mg is not None)

        rt_loader = DataLoader(
            data,
            batch_size=1024,
            shuffle=False,
            num_workers=8,
            drop_last=False
        )

        all_results = {'time': []}
        if self.use_multiview:
            all_results['freq'] = []
            all_results['stat'] = []

        with torch.no_grad():
            for index, batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(rt_loader):
                result = self.retrieve(batch_x.float().to(device), index, train=train)
                for view_key in result:
                    result[view_key] = result[view_key].cpu()
                    all_results[view_key].append(result[view_key])

        for view_key in all_results:
            all_results[view_key] = torch.cat(all_results[view_key], dim=1)

        return all_results