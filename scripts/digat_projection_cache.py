"""Exact evaluation-only cache of DIGAT layer-zero linear projections.

Only W/ffn1/ffn2 projections of initial news/topic vectors are cached.
Candidate-conditioned attention and all subsequent graph layers are unchanged.
"""
import hashlib
from pathlib import Path
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from naml_cached_eval import _array_hash_into, _file_hash, _json_bytes, _save_json, _save_npy


class DigatLayerZeroCache:
    def __init__(self, model, news, directory, model_hash, device, batch_size=512):
        self.model, self.graph, self.device = model, model.graph_encoder, device
        self.news_count = len(news)
        if self.graph.training or type(self.graph).__name__ != "DIGAT":
            raise ValueError("Projection cache requires the official DIGAT in eval mode")
        if self.graph.graph_depth < 1:
            raise ValueError("DIGAT requires at least one interaction layer")
        digest = hashlib.sha256()
        _array_hash_into(digest, "news_embeddings", news)
        self.provenance = {"schema": 1, "model_state_sha256": model_hash,
                           "news_embeddings_sha256": digest.hexdigest(),
                           "implementation_sha256": _file_hash(__file__),
                           "torch_version": torch.__version__,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "device": str(device),
                           "cache_scope": "only layer0 W/ffn1/ffn2 of initial vectors"}
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        names = ("news_W", "news_ffn1", "news_ffn2", "user_W", "user_ffn1", "user_ffn2")
        manifest_path = self.directory / "manifest.json"
        started = time.monotonic()
        self.loaded = manifest_path.exists()
        if self.loaded:
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
            if metadata["provenance"] != self.provenance:
                raise ValueError("Projection cache provenance mismatch")
            for name, expected in metadata["files"].items():
                if _file_hash(self.directory / name) != expected:
                    raise ValueError("Projection cache file changed: " + name)
        else:
            if any(self.directory.iterdir()):
                raise ValueError("Incomplete projection cache directory")
            with torch.inference_mode():
                for name in names:
                    side, projection = name.split("_", 1)
                    layer = getattr(self.graph, side + "_graph_attention_" + projection)[0]
                    count = len(news) + (self.graph.category_num - 1 if side == "user" else 0)
                    output = np.empty((count, self.graph.news_embedding_dim), dtype=np.float32)
                    for start in range(0, len(news), batch_size):
                        end = min(start + batch_size, len(news))
                        output[start:end] = layer(torch.as_tensor(news[start:end], device=device)).cpu().numpy()
                    if side == "user":
                        output[len(news):] = layer(self.graph.topic_node_embedding).cpu().numpy()
                    if not np.isfinite(output).all():
                        raise ValueError("Nonfinite projection cache")
                    _save_npy(self.directory / (name + ".npy"), output)
            _save_json(manifest_path, {"provenance": self.provenance,
                       "files": {name + ".npy": _file_hash(self.directory / (name + ".npy")) for name in names}})
        self.values = {}
        self.bytes = 0
        for name in names:
            array = np.load(self.directory / (name + ".npy"), allow_pickle=False)
            count = len(news) + (self.graph.category_num - 1 if name.startswith("user_") else 0)
            if array.dtype != np.float32 or array.shape != (count, self.graph.news_embedding_dim):
                raise ValueError("Invalid projection cache shape/dtype")
            self.bytes += array.nbytes
            self.values[name] = torch.as_tensor(array, device=device)
        self.elapsed_seconds = time.monotonic() - started

    def _layer_zero(self, side, embeddings, graph, context, indices):
        g = self.graph
        h = self.values[side + "_W"][indices]
        k1 = self.values[side + "_ffn1"][indices].unsqueeze(1)
        k2 = self.values[side + "_ffn2"][indices].unsqueeze(2)
        k3 = getattr(g, side + "_graph_attention_ffn3")[0](context).view(len(embeddings), 1, 1, g.news_embedding_dim)
        a = getattr(g, side + "_graph_attention_a")[0](F.relu(k3 + k1 + k2, inplace=True)).squeeze(3)
        weights = F.softmax(g.leaky_relu(a).masked_fill(graph == 0, -1e9), dim=2)
        # Both original dropout modules are identities in eval mode.
        return F.relu(torch.bmm(weights, h), inplace=True) + embeddings

    def score(self, news_vectors, contexts, histories, candidate_ids, news_nodes,
              user_graph, category_mask, category_indices, news_graph, news_graph_mask):
        g = self.graph
        if self.model.training or g.training:
            raise ValueError("Layer-zero projection cache cannot be used during training")
        user_news = news_vectors[histories]
        candidate_news = news_vectors[news_nodes]
        batch = len(candidate_ids)
        user_embeddings = torch.cat((user_news, g.topic_node_embedding.unsqueeze(0).expand(batch, -1, -1)), dim=1)
        news_context = contexts[candidate_ids]
        user_context = g.compute_user_graph_context(user_embeddings, category_mask, category_indices, news_context)
        topic_ids = torch.arange(g.category_num - 1, device=self.device) + self.news_count
        user_nodes = torch.cat((histories, topic_ids.unsqueeze(0).expand(batch, -1)), dim=1)
        candidate_news = self._layer_zero("news", candidate_news, news_graph, user_context, news_nodes)
        user_embeddings = self._layer_zero("user", user_embeddings, user_graph, news_context, user_nodes)
        news_context = news_context + g.compute_news_graph_context(candidate_news, news_graph_mask)
        user_context = user_context + g.compute_user_graph_context(user_embeddings, category_mask, category_indices, news_context)
        for index in range(1, g.graph_depth):
            candidate_news = g.compute_news_graph_embeddings(index, candidate_news, news_graph, user_context)
            user_embeddings = g.compute_user_graph_embeddings(index, user_embeddings, user_graph, news_context)
            news_context = news_context + g.compute_news_graph_context(candidate_news, news_graph_mask)
            user_context = user_context + g.compute_user_graph_context(user_embeddings, category_mask, category_indices, news_context)
        return (news_context * user_context).sum(dim=1)
