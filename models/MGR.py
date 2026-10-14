import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Retrieval import RetrievalTool


class Model(nn.Module):
    """
    RAFT with Multi-View Memory Encoding and Graph-Enhanced Retrieval.
    """

    def __init__(self, configs, individual=False):
        super(Model, self).__init__()
        self.device = torch.device(f'cuda:{configs.gpu}')
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        self.channels = configs.enc_in

        self.linear_x = nn.Linear(self.seq_len, self.pred_len)

        self.n_period = configs.n_period
        self.topm = configs.topm

        # New configs for multi-view and graph
        self.use_multiview = getattr(configs, 'use_multiview', False)
        self.use_graph = getattr(configs, 'use_graph', False)
        self.n_fft = getattr(configs, 'n_fft', 8)
        self.graph_hop = getattr(configs, 'graph_hop', 1)

        self.rt = RetrievalTool(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            channels=self.channels,
            n_period=self.n_period,
            topm=self.topm,
            use_multiview=self.use_multiview,
            use_graph=self.use_graph,
            n_fft=self.n_fft,
            graph_hop=self.graph_hop,
        )

        self.period_num = self.rt.period_num[-1 * self.n_period:]

        module_list = [
            nn.Linear(self.pred_len // g, self.pred_len)
            for g in self.period_num
        ]
        self.retrieval_pred = nn.ModuleList(module_list)
        self.linear_pred = nn.Linear(2 * self.pred_len, self.pred_len)

        # Learnable late fusion weights for multi-view
        # view_fusion_weights: [w_time, w_freq, w_stat]
        num_views = 3 if self.use_multiview else 1
        self.view_fusion_weights = nn.Parameter(torch.ones(num_views) / num_views)

    def prepare_dataset(self, train_data, valid_data, test_data):
        self.rt.prepare_dataset(train_data)

        self.retrieval_dict = {}

        print('Doing Train Retrieval')
        train_rt = self.rt.retrieve_all(train_data, train=True, device=self.device)

        print('Doing Valid Retrieval')
        valid_rt = self.rt.retrieve_all(valid_data, train=False, device=self.device)

        print('Doing Test Retrieval')
        test_rt = self.rt.retrieve_all(test_data, train=False, device=self.device)

        del self.rt
        torch.cuda.empty_cache()

        # Store per-view results
        self.retrieval_dict['train'] = {k: v.detach() for k, v in train_rt.items()}
        self.retrieval_dict['valid'] = {k: v.detach() for k, v in valid_rt.items()}
        self.retrieval_dict['test'] = {k: v.detach() for k, v in test_rt.items()}

    def _fuse_views(self, view_results):
        """
        Fuse multi-view retrieval results using learnable weights.
        view_results: dict of view_key -> (G, B, P, C)
        Returns: (G, B, P, C) fused result
        """
        if not self.use_multiview:
            return view_results['time']

        weights = F.softmax(self.view_fusion_weights, dim=0)  # (3,)
        fused = None

        view_keys = ['time', 'freq', 'stat']
        for i, vk in enumerate(view_keys):
            if vk in view_results:
                weighted = view_results[vk] * weights[i]
                if fused is None:
                    fused = weighted
                else:
                    fused = fused + weighted

        return fused

    def encoder(self, x, index, mode):
        index = index.to(self.device)

        bsz, seq_len, channels = x.shape
        assert (seq_len == self.seq_len, channels == self.channels)

        x_offset = x[:, -1:, :].detach()
        x_norm = x - x_offset

        x_pred_from_x = self.linear_x(x_norm.permute(0, 2, 1)).permute(0, 2, 1)  # B, P, C

        # Get per-view retrieval results
        view_results = {}
        for vk in self.retrieval_dict[mode]:
            view_results[vk] = self.retrieval_dict[mode][vk][:, index].to(self.device)

        # Late fusion across views
        pred_from_retrieval = self._fuse_views(view_results)  # G, B, P, C

        retrieval_pred_list = []

        # Compress repeating dimensions
        for i, pr in enumerate(pred_from_retrieval):
            assert ((bsz, self.pred_len, channels) == pr.shape)
            g = self.period_num[i]
            pr = pr.reshape(bsz, self.pred_len // g, g, channels)
            pr = pr[:, :, 0, :]

            pr = self.retrieval_pred[i](pr.permute(0, 2, 1)).permute(0, 2, 1)
            pr = pr.reshape(bsz, self.pred_len, self.channels)

            retrieval_pred_list.append(pr)

        retrieval_pred_list = torch.stack(retrieval_pred_list, dim=1)
        retrieval_pred_list = retrieval_pred_list.sum(dim=1)

        pred = torch.cat([x_pred_from_x, retrieval_pred_list], dim=1)
        pred = self.linear_pred(pred.permute(0, 2, 1)).permute(0, 2, 1).reshape(bsz, self.pred_len, self.channels)

        pred = pred + x_offset

        return pred

    def forecast(self, x_enc, index, mode):
        return self.encoder(x_enc, index, mode)

    def imputation(self, x_enc, index, mode):
        return self.encoder(x_enc, index, mode)

    def anomaly_detection(self, x_enc, index, mode):
        return self.encoder(x_enc, index, mode)

    def classification(self, x_enc, index, mode):
        enc_out = self.encoder(x_enc, index, mode)
        output = enc_out.reshape(enc_out.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, index, mode='train'):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, index, mode)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, index, mode)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, index, mode)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, index, mode)
            return dec_out  # [B, N]
        return None