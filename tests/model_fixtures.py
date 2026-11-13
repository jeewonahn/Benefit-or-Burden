"""Small deterministic-shape fixtures for CPU model and cache tests."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from nnr_artifacts import MODELS

def tiny_config(model):
    tree = ast.parse((ROOT/'NNR-main/config.py').read_text(encoding='utf-8'))
    values = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
            if node.args and isinstance(node.args[0], ast.Constant):
                for kw in node.keywords:
                    if kw.arg == 'default': values[node.args[0].value.lstrip('-')] = ast.literal_eval(kw.value)
    values.update(dataset='synthetic_smoke_only', vocabulary_size=16, user_num=4,
        category_num=2, subCategory_num=3, word_embedding_dim=50,
        max_title_length=5, max_abstract_length=7, max_history_num=3,
        cnn_kernel_num=8, attention_dim=8, head_num=2, head_dim=4,
        user_embedding_dim=4, category_embedding_dim=2, subCategory_embedding_dim=2,
        hidden_dim=4, personalized_embedding_dim=4, gcn_layer_num=1,
        dropout_rate=0.0, batch_size=2, negative_sample_num=2)
    values['news_encoder'], values['user_encoder'] = MODELS[model]
    return SimpleNamespace(**values)

def synthetic_inputs(torch, cfg):
    batch, history, candidates = 2, 3, 3
    def news(count):
        category = torch.randint(0, cfg.category_num, (batch,count))
        subcategory = torch.randint(0, cfg.subCategory_num, (batch,count))
        title = torch.randint(1,cfg.vocabulary_size,(batch,count,cfg.max_title_length))
        content = torch.randint(1,cfg.vocabulary_size,(batch,count,cfg.max_abstract_length))
        return (category,subcategory,title,torch.ones_like(title),torch.zeros_like(title),
                content,torch.ones_like(content),torch.zeros_like(content))
    user = news(history); candidate = news(candidates)
    graph = torch.eye(history+cfg.category_num).unsqueeze(0).repeat(batch,1,1)
    return (torch.tensor([1,2]),*user,torch.ones(batch,history,dtype=torch.bool),graph,
            torch.ones(batch,cfg.category_num+1,dtype=torch.bool),user[0].clone(),*candidate)
