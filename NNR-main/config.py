import os
import argparse
import sys
import time
import torch
import random
import numpy as np
import json
from prepare_MIND_dataset import prepare_MIND_200k, prepare_MIND_small, prepare_MIND_large, prepare_MIND_codetest


class Config:
    def parse_argument(self):
        parser = argparse.ArgumentParser(description='Neural news recommendation')
        # General config
        parser.add_argument('--mode', type=str, default='train', choices=['train', 'dev', 'test'], help='Mode')
        parser.add_argument('--news_encoder', type=str, default='CNE', choices=['CNE', 'CNN', 'MHSA', 'KCNN', 'HDC', 'NAML', 'PNE', 'DAE', 'Inception', 'NAML_Title', 'NAML_Content', 'CNE_Title', 'CNE_Content', 'CNE_wo_CS', 'CNE_wo_CA'], help='News encoder')
        parser.add_argument('--user_encoder', type=str, default='SUE', choices=['SUE', 'LSTUR', 'MHSA', 'ATT', 'CATT', 'FIM', 'PUE', 'GRU', 'OMAP', 'SUE_wo_GCN', 'SUE_wo_HCA'], help='User encoder')
        parser.add_argument('--dev_model_path', type=str, default='', help='Dev model path')
        parser.add_argument('--test_model_path', type=str, default='', help='Test model path')
        parser.add_argument('--test_output_file', type=str, default='', help='Specific test output file')
        parser.add_argument('--device_id', type=int, default=0, help='Device ID of GPU')
        parser.add_argument('--seed', type=int, default=0, help='Seed for random number generator')
        parser.add_argument('--config_file', type=str, default='', help='Config file path')
        # Dataset config
        parser.add_argument('--dataset', type=str, default='codetest', choices=['200k', 'small', 'large', 'codetest'], help='Dataset type')
        parser.add_argument('--train_type', type=str, default='imp', choices=['imp', 'rs50'], help='Train behavior file type')
        parser.add_argument('--eval_type', type=str, default='imp', choices=['imp', 'tt', 'rs30', 'rs50', 'rs60', 'rs90', 'rs120', 'rs150'], help='Dev/Test behavior file type')
        parser.add_argument('--data_root', default='', help='Explicit preprocessed dataset root; never auto-download when supplied')
        parser.add_argument('--embedding_root', default='', help='Directory containing local GloVe vectors')
        parser.add_argument('--dev_type', default=None, choices=['imp', 'tt', 'rs30', 'rs50', 'rs60', 'rs90', 'rs120', 'rs150'])
        parser.add_argument('--test_type', default=None, choices=['imp', 'tt', 'rs30', 'rs50', 'rs60', 'rs90', 'rs120', 'rs150'])
        parser.add_argument('--tokenizer', type=str, default='MIND', choices=['MIND', 'NLTK'], help='Sentence tokenizer')
        parser.add_argument('--word_threshold', type=int, default=3, help='Word threshold')
        parser.add_argument('--max_title_length', type=int, default=32, help='Sentence truncate length for title')
        parser.add_argument('--max_abstract_length', type=int, default=128, help='Sentence truncate length for abstract')
        # Training config
        parser.add_argument('--negative_sample_num', type=int, default=4, help='Negative sample number of each positive sample')
        parser.add_argument('--max_history_num', type=int, default=50, help='Maximum number of history news for each user')
        parser.add_argument('--epoch', type=int, default=1, help='Training epoch')
        parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
        parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        parser.add_argument('--weight_decay', type=float, default=0, help='Optimizer weight decay')
        parser.add_argument('--gradient_clip_norm', type=float, default=4, help='Gradient clip norm (non-positive value for no clipping)')
        parser.add_argument('--world_size', type=int, default=1, help='World size of multi-process GPU training')
        # Dev config
        parser.add_argument('--dev_criterion', type=str, default='avg', choices=['auc', 'mrr', 'ndcg5', 'ndcg10', 'avg'], help='Validation criterion to select model')
        parser.add_argument('--early_stopping_epoch', type=int, default=3, help='Epoch number of stop training after dev result does not improve')
        # Model config
        parser.add_argument('--word_embedding_dim', type=int, default=300, choices=[50, 100, 200, 300], help='Word embedding dimension')
        parser.add_argument('--entity_embedding_dim', type=int, default=100, choices=[100], help='Entity embedding dimension')
        parser.add_argument('--context_embedding_dim', type=int, default=100, choices=[100], help='Context embedding dimension')
        parser.add_argument('--cnn_method', type=str, default='naive', choices=['naive', 'group3', 'group4', 'group5'], help='CNN group')
        parser.add_argument('--cnn_kernel_num', type=int, default=400, help='Number of CNN kernel')
        parser.add_argument('--cnn_window_size', type=int, default=3, help='Window size of CNN kernel')
        parser.add_argument('--attention_dim', type=int, default=200, help="Attention dimension")
        parser.add_argument('--head_num', type=int, default=20, help='Head number of multi-head self-attention')
        parser.add_argument('--head_dim', type=int, default=20, help='Head dimension of multi-head self-attention')
        parser.add_argument('--user_embedding_dim', type=int, default=50, help='User embedding dimension')
        parser.add_argument('--category_embedding_dim', type=int, default=50, help='Category embedding dimension')
        parser.add_argument('--subCategory_embedding_dim', type=int, default=50, help='SubCategory embedding dimension')
        parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
        parser.add_argument('--no_self_connection', default=False, action='store_true', help='Whether the graph contains self-connection')
        parser.add_argument('--no_adjacent_normalization', default=False, action='store_true', help='Whether normalize the adjacent matrix')
        parser.add_argument('--gcn_normalization_type', type=str, default='symmetric', choices=['symmetric', 'asymmetric'], help='GCN normalization for adjacent matrix A (\"symmetric\" for D^{-\\frac{1}{2}}AD^{-\\frac{1}{2}}; \"asymmetric\" for D^{-\\frac{1}{2}}A)')
        parser.add_argument('--gcn_layer_num', type=int, default=4, help='Number of GCN layer')
        parser.add_argument('--no_gcn_residual', default=False, action='store_true', help='Whether apply residual connection to GCN')
        parser.add_argument('--gcn_layer_norm', default=False, action='store_true', help='Whether apply layer normalization to GCN')
        parser.add_argument('--hidden_dim', type=int, default=200, help='Encoder hidden dimension')
        parser.add_argument('--Alpha', type=float, default=0.1, help='Reconstruction loss weight for DAE')
        parser.add_argument('--long_term_masking_probability', type=float, default=0.1, help='Probability of masking long-term representation for LSTUR')
        parser.add_argument('--personalized_embedding_dim', type=int, default=200, help='Personalized embedding dimension for NPA')
        parser.add_argument('--HDC_window_size', type=int, default=3, help='Convolution window size of HDC for FIM')
        parser.add_argument('--HDC_filter_num', type=int, default=150, help='Convolution filter num of HDC for FIM')
        parser.add_argument('--conv3D_filter_num_first', type=int, default=32, help='3D matching convolution filter num of the first layer for FIM ')
        parser.add_argument('--conv3D_kernel_size_first', type=int, default=3, help='3D matching convolution kernel size of the first layer for FIM')
        parser.add_argument('--conv3D_filter_num_second', type=int, default=16, help='3D matching convolution filter num of the second layer for FIM ')
        parser.add_argument('--conv3D_kernel_size_second', type=int, default=3, help='3D matching convolution kernel size of the second layer for FIM')
        parser.add_argument('--maxpooling3D_size', type=int, default=3, help='3D matching pooling size for FIM ')
        parser.add_argument('--maxpooling3D_stride', type=int, default=3, help='3D matching pooling stride for FIM')
        parser.add_argument('--OMAP_head_num', type=int, default=3, help='Head num of OMAP for Hi-Fi Ark')
        parser.add_argument('--HiFi_Ark_regularizer_coefficient', type=float, default=0.1, help='Coefficient of regularization loss for Hi-Fi Ark')
        parser.add_argument('--click_predictor', type=str, default='dot_product', choices=['dot_product', 'mlp', 'sigmoid', 'FIM'], help='Click predictor')

        # Precedence: dataset defaults < JSON settings < explicit CLI flags.
        values = vars(parser.parse_args())
        explicit = {arg.split('=', 1)[0][2:] for arg in sys.argv[1:] if arg.startswith('--')}
        cli = dict(values)
        if values['config_file']:
            with open(values['config_file'], encoding='utf-8') as stream:
                supplied = json.load(stream)
            if not isinstance(supplied, dict):
                parser.error('Config JSON must be an object')
            unknown = set(supplied) - set(values)
            if unknown:
                parser.error('Unknown config keys: ' + ', '.join(sorted(unknown)))
            values.update(supplied)
            values.update({key: cli[key] for key in explicit if key in cli})
        else:
            supplied = {}
        presets = {'small': {'dropout_rate': 0.25, 'gcn_layer_num': 3},
                   'codetest': {'dropout_rate': 0.25, 'gcn_layer_num': 3},
                   '200k': {'dropout_rate': 0.2, 'gcn_layer_num': 4, 'epoch': 8},
                   'large': {'dropout_rate': 0.1, 'gcn_layer_num': 4, 'epoch': 6}}
        if values['dataset'] not in presets:
            parser.error('Unsupported dataset')
        for key, value in presets[values['dataset']].items():
            if key not in supplied and key not in explicit:
                values[key] = value
        for action in parser._actions:
            if action.dest not in values or values[action.dest] is None:
                continue
            value = values[action.dest]
            if action.choices and value not in action.choices:
                parser.error('Invalid config value for ' + action.dest)
            if action.type is not None:
                try:
                    values[action.dest] = action.type(value)
                except (TypeError, ValueError):
                    parser.error('Invalid config type for ' + action.dest)
        values['seed'] = values['seed'] if values['seed'] >= 0 else int(time.time())
        values['dev_type'] = values['dev_type'] or values['train_type']
        if values['dev_type'] != values['train_type']:
            parser.error('Development candidates must match train_type: IMP to IMP, RandS50 to RandS50')
        values['test_type'] = values['test_type'] or values['eval_type']
        self.attribute_dict = values
        for key, value in values.items():
            setattr(self, key, value)
        dataset_dir = {'small': 'MIND_small', '200k': 'MIND_200k', 'large': 'MIND_large', 'codetest': 'MIND_codetest'}[self.dataset]
        dataset_root = self.data_root or os.path.join('..', 'data', 'MIND', dataset_dir)
        self.train_root = os.path.join(dataset_root, 'train')
        self.dev_root = os.path.join(dataset_root, 'dev')
        self.test_root = os.path.join(dataset_root, 'test')
        behavior_name = lambda kind: 'behaviors.tsv' if kind == 'imp' else 'behaviors_' + kind + '.tsv'
        self.train_behaviors_file = os.path.join(self.train_root, behavior_name(self.train_type))
        self.dev_behaviors_file = os.path.join(self.dev_root, behavior_name(self.dev_type))
        self.test_behaviors_file = os.path.join(self.test_root, behavior_name(self.test_type))
        self.dev_truth_file = os.path.join('dev', 'ref', f'truth-{self.dataset}-{self.dev_type}.txt')
        self.test_truth_file = os.path.join('test', 'ref', f'truth-{self.dataset}-{self.test_type}.txt')
        os.environ['MIND_EVAL_TYPE'] = self.eval_type  # legacy fallback
        os.environ['MIND_DEV_TRUTH_FILE'] = self.dev_truth_file
        os.environ['MIND_TEST_TRUTH_FILE'] = self.test_truth_file
        assert not (self.no_self_connection and not self.no_adjacent_normalization), 'Adjacent normalization of graph only can be set in case of self-connection'
        print('*' * 32 + ' Experiment setting ' + '*' * 32)
        for attribute in self.attribute_dict:
            print(attribute + ' : ' + str(getattr(self, attribute)))
        print('*' * 32 + ' Experiment setting ' + '*' * 32)
        assert self.batch_size % self.world_size == 0, 'For multi-gpu training, batch size must be divisible by world size'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '1024'


    def set_cuda(self):
        gpu_available = torch.cuda.is_available()
        assert gpu_available, 'GPU is not available'
        torch.cuda.set_device(self.device_id)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True # For reproducibility (https://pytorch.org/docs/stable/notes/randomness.html)


    def preliminary_setup(self):
        import importlib.util
        from pathlib import Path
        helper=Path(__file__).resolve().parents[1]/'scripts/context_feature_policy.py'
        spec=importlib.util.spec_from_file_location('_tors_context_feature_config',helper)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        context=module.context_feature_spec(self.data_root,encoder=self.news_encoder,user_encoder=self.user_encoder)
        self.context_feature_mode=context['mode'];self.context_features_required=context['required']
        dataset_files = [
            self.train_root + '/news.tsv', self.train_behaviors_file, self.train_root + '/entity_embedding.vec', self.train_root + '/context_embedding.vec', 
            self.dev_root + '/news.tsv', self.dev_behaviors_file, self.dev_root + '/entity_embedding.vec', self.dev_root + '/context_embedding.vec', 
            self.test_root + '/news.tsv', self.test_behaviors_file, self.test_root + '/entity_embedding.vec', self.test_root + '/context_embedding.vec'
        ]
        if not context['required']:
            dataset_files=[path for path in dataset_files if not path.endswith('/context_embedding.vec')]
        missing = [path for path in dataset_files if not os.path.isfile(path)]
        if missing:
            if self.data_root:
                raise FileNotFoundError('Explicit dataset is incomplete; no automatic split/download: ' + ', '.join(missing))
            exec('prepare_MIND_%s()' % self.dataset)

        model_name = self.news_encoder + '-' + self.user_encoder
        mkdirs = lambda x: os.makedirs(x) if not os.path.exists(x) else None
        self.config_dir = 'configs/' + self.dataset + '/' + model_name
        self.model_dir = 'models/' + self.dataset + '/' + model_name
        self.best_model_dir = 'best_model/' + self.dataset + '/' + model_name
        self.dev_res_dir = 'dev/res/' + self.dataset + '/' + model_name
        self.test_res_dir = 'test/res/' + self.dataset + '/' + model_name
        self.result_dir = 'results/' + self.dataset + '/' + model_name
        mkdirs(self.config_dir)
        mkdirs(self.model_dir)
        mkdirs(self.best_model_dir)
        mkdirs('dev/ref')
        mkdirs(self.dev_res_dir)
        mkdirs('test/ref')
        mkdirs(self.test_res_dir)
        mkdirs(self.result_dir)
        if not os.path.exists(self.dev_truth_file):
            with open(self.dev_behaviors_file, 'r', encoding='utf-8') as dev_f:
                with open(self.dev_truth_file, 'w', encoding='utf-8') as truth_f:
                    for dev_ID, line in enumerate(dev_f):
                        impression_ID, user_ID, time, history, impressions = line.split('\t')
                        labels = [int(impression[-1]) for impression in impressions.strip().split(' ')]
                        truth_f.write(('' if dev_ID == 0 else '\n') + str(dev_ID + 1) + ' ' + str(labels).replace(' ', ''))
        if self.dataset != 'large':
            if not os.path.exists(self.test_truth_file):
                with open(self.test_behaviors_file, 'r', encoding='utf-8') as test_f:
                    with open(self.test_truth_file, 'w', encoding='utf-8') as truth_f:
                        for test_ID, line in enumerate(test_f):
                            impression_ID, user_ID, time, history, impressions = line.split('\t')
                            labels = [int(impression[-1]) for impression in impressions.strip().split(' ')]
                            truth_f.write(('' if test_ID == 0 else '\n') + str(test_ID + 1) + ' ' + str(labels).replace(' ', ''))
        else:
            self.prediction_dir = 'prediction/large/' + model_name
            mkdirs(self.prediction_dir)


    def __init__(self):
        self.parse_argument()
        if self.train_type == 'rs50':
            # Guard direct legacy entry points as well as the public wrappers.
            import importlib.util
            helper = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  'scripts', 'prepare_development_candidates.py')
            spec = importlib.util.spec_from_file_location('_nnr_matched_candidate_profile', helper)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            module.validate_matched_candidate_profile(os.path.dirname(self.train_root))
        self.preliminary_setup()
        self.set_cuda()


if __name__ == '__main__':
    config = Config()
