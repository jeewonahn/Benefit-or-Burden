"""NNR batch alignment for personalized queries and cross-view cell states.

Apply the same alignment in training and cached evaluation and record the
selected implementation in the run configuration.
"""
from types import MethodType
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _pne_forward(self, title_text, title_mask, title_entity, content_text, content_mask,
                 content_entity, category, subCategory, user_embedding):
    batch_size, news_num = title_text.shape[:2]
    mask = title_mask.reshape(batch_size * news_num, self.max_sentence_length)
    w = self.dropout(self.word_embedding(title_text)).reshape(
        batch_size * news_num, self.max_sentence_length, self.word_embedding_dim)
    c = self.dropout_(self.conv(w.permute(0, 2, 1)).permute(0, 2, 1))
    query = F.relu(self.dense(user_embedding), inplace=True).repeat_interleave(news_num, dim=0)
    representation = self.personalizedAttention(c, query, mask).reshape(
        batch_size, news_num, self.cnn_kernel_num)
    return self.feature_fusion(representation, category, subCategory)


def _cne_forward(self, title_text, title_mask, title_entity, content_text, content_mask,
                 content_entity, category, subCategory, user_embedding):
    batch_size, news_num = title_text.shape[:2]
    count = batch_size * news_num
    # Clone masks: avoid modifying shared corpus feature arrays during inference.
    title_mask = title_mask.reshape(count, self.max_title_length).clone()
    content_mask = content_mask.reshape(count, self.max_content_length).clone()
    title_mask[:, 0] = 1
    content_mask[:, 0] = 1
    title_lengths, title_order = torch.sort(title_mask.sum(dim=1).long(), descending=True)
    content_lengths, content_order = torch.sort(content_mask.sum(dim=1).long(), descending=True)
    title_inverse = torch.argsort(title_order)
    content_inverse = torch.argsort(content_order)
    title = self.dropout(self.word_embedding(title_text)).reshape(count, self.max_title_length, self.word_embedding_dim)
    content = self.dropout(self.word_embedding(content_text)).reshape(count, self.max_content_length, self.word_embedding_dim)
    packed_title = pack_padded_sequence(title[title_order], title_lengths.cpu(), batch_first=True)
    packed_content = pack_padded_sequence(content[content_order], content_lengths.cpu(), batch_first=True)
    title_h, (_, title_c) = self.title_lstm(packed_title)
    content_h, (_, content_c) = self.content_lstm(packed_content)
    title_m = torch.cat([title_c[0], title_c[1]], dim=1)
    content_m = torch.cat([content_c[0], content_c[1]], dim=1)
    title_h, _ = pad_packed_sequence(title_h, batch_first=True, total_length=self.max_title_length)
    content_h, _ = pad_packed_sequence(content_h, batch_first=True, total_length=self.max_content_length)
    # Cell states must first return to original article order, then adopt the
    # other text view's sorted order. Legacy code directly mixed sorted orders.
    content_for_title = content_m[content_inverse][title_order]
    title_for_content = title_m[title_inverse][content_order]
    title_gate = torch.sigmoid(self.title_H(title_h) + self.title_M(content_for_title).unsqueeze(1))
    content_gate = torch.sigmoid(self.content_H(content_h) + self.content_M(title_for_content).unsqueeze(1))
    title_h = (title_h * title_gate)[title_inverse]
    content_h = (content_h * content_gate)[content_inverse]
    title_self = self.title_self_attention(title_h, title_mask)
    content_self = self.content_self_attention(content_h, content_mask)
    title_cross = self.title_cross_attention(title_h, content_self, title_mask)
    content_cross = self.content_cross_attention(content_h, title_self, content_mask)
    representation = torch.cat([title_self + title_cross, content_self + content_cross], dim=1).reshape(
        batch_size, news_num, self.hidden_dim * 4)
    return self.feature_fusion(representation, category, subCategory)


def apply_batch_alignment_fixes(model):
    """Return explicit implementation adjustments; safe to call repeatedly."""
    kind = type(model.news_encoder).__name__
    adjustments = []
    if kind == 'PNE':
        model.news_encoder.forward = MethodType(_pne_forward, model.news_encoder)
        adjustments.append('PNE: align personalized query to each user/news with repeat_interleave')
    elif kind == 'CNE':
        model.news_encoder.forward = MethodType(_cne_forward, model.news_encoder)
        adjustments.append('CNE: align cross-view cell states by original article identity before sorted-view gating')
    model._tors_batch_alignment_fixes = adjustments
    return adjustments
