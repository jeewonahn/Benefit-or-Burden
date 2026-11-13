import os
import json
import shutil
import random
import numpy as np
import shutil
import collections
random.seed(0)
np.random.seed(0)
BASE_DATA_ROOT = '../data/MIND'
MIND_small_dataset_root = os.path.join(BASE_DATA_ROOT, 'MIND_small')
MIND_large_dataset_root = os.path.join(BASE_DATA_ROOT, 'MIND_large')
MIND_200k_dataset_root = os.path.join(BASE_DATA_ROOT, 'MIND_200k')
MIND_codetest_dataset_root = os.path.join(BASE_DATA_ROOT, 'MIND_codetest')


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def download_extract_MIND_small():
    if not os.path.exists(MIND_small_dataset_root):
        os.mkdir(MIND_small_dataset_root)
    if not os.path.exists(MIND_small_dataset_root + '/download'):
        os.mkdir(MIND_small_dataset_root + '/download')
    if not os.path.exists(MIND_small_dataset_root + '/download/train'):
        if not os.path.exists(MIND_small_dataset_root + '/download/MINDsmall_train.zip'):
            os.system('wget https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip -P %s/download' % MIND_small_dataset_root)
        assert os.path.exists(MIND_small_dataset_root + '/download/MINDsmall_train.zip'), 'Train set zip not found'
        os.mkdir(MIND_small_dataset_root + '/download/train')
        os.system('unzip %s/download/MINDsmall_train.zip -d %s/download/train' % (MIND_small_dataset_root, MIND_small_dataset_root))
    if not os.path.exists(MIND_small_dataset_root + '/download/dev'):
        if not os.path.exists(MIND_small_dataset_root + '/download/MINDsmall_dev.zip'):
            os.system('wget https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip -P %s/download' % MIND_small_dataset_root)
        assert os.path.exists(MIND_small_dataset_root + '/download/MINDsmall_dev.zip'), 'Dev set zip not found'
        os.mkdir(MIND_small_dataset_root + '/download/dev')
        os.system('unzip %s/download/MINDsmall_dev.zip -d %s/download/dev' % (MIND_small_dataset_root, MIND_small_dataset_root))
    if not os.path.exists(MIND_small_dataset_root + '/download/wikidata-graph'):
        if not os.path.exists(MIND_small_dataset_root + '/download/wikidata-graph.zip'):
            os.system('wget https://mind201910.blob.core.windows.net/knowledge-graph/wikidata-graph.zip -P %s/download' % MIND_small_dataset_root)
        os.system('unzip %s/download/wikidata-graph.zip -d %s/download' % (MIND_small_dataset_root, MIND_small_dataset_root))


def download_extract_MIND_large(mode):
    assert mode in ['200k', 'large']
    dataset_root = MIND_200k_dataset_root if mode == '200k' else MIND_large_dataset_root
    if not os.path.exists(dataset_root):
        os.mkdir(dataset_root)
    if not os.path.exists(os.path.join(dataset_root, 'train')):
        os.mkdir(os.path.join(dataset_root, 'train'))
    if not os.path.exists(os.path.join(dataset_root, 'dev')):
        os.mkdir(os.path.join(dataset_root, 'dev'))
    if not os.path.exists(os.path.join(dataset_root, 'test')):
        os.mkdir(os.path.join(dataset_root, 'test'))
    if not os.path.exists(dataset_root + '/download'):
        os.mkdir(dataset_root + '/download')
    if (not os.path.exists(dataset_root + '/train/news.tsv') or not os.path.exists(dataset_root + '/train/behaviors.tsv') or not os.path.exists(dataset_root + '/train/entity_embedding.vec')) and \
        not os.path.exists(dataset_root + '/download/train'):
        if not os.path.exists(dataset_root + '/download/MINDlarge_train.zip'):
            os.system('wget https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip -P %s/download' % dataset_root)
        assert os.path.exists(dataset_root + '/download/MINDlarge_train.zip'), 'Train set zip not found'
        os.mkdir(dataset_root + '/download/train')
        os.system('unzip %s/download/MINDlarge_train.zip -d %s/download/train' % (dataset_root, dataset_root))
    if (not os.path.exists(dataset_root + '/dev/news.tsv') or not os.path.exists(dataset_root + '/dev/behaviors.tsv') or not os.path.exists(dataset_root + '/dev/entity_embedding.vec')) and \
        not os.path.exists(dataset_root + '/download/dev'):
        if not os.path.exists(dataset_root + '/download/MINDlarge_dev.zip'):
            os.system('wget https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip -P %s/download' % dataset_root)
        assert os.path.exists(dataset_root + '/download/MINDlarge_dev.zip'), 'Dev set zip not found'
        os.mkdir(dataset_root + '/download/dev')
        os.system('unzip %s/download/MINDlarge_dev.zip -d %s/download/dev' % (dataset_root, dataset_root))
    if (not os.path.exists(dataset_root + '/test/news.tsv') or not os.path.exists(dataset_root + '/test/behaviors.tsv') or not os.path.exists(dataset_root + '/test/entity_embedding.vec')) and \
        not os.path.exists(dataset_root + '/download/test'):
        if not os.path.exists(dataset_root + '/download/MINDlarge_test.zip'):
            os.system('wget https://mind201910small.blob.core.windows.net/release/MINDlarge_test.zip -P %s/download' % dataset_root)
        assert os.path.exists(dataset_root + '/download/MINDlarge_test.zip'), 'Test set zip not found'
        os.mkdir(dataset_root + '/download/test')
        os.system('unzip %s/download/MINDlarge_test.zip -d %s/download/test' % (dataset_root, dataset_root))
    if not os.path.exists(dataset_root + '/download/wikidata-graph'):
        if not os.path.exists(dataset_root + '/download/wikidata-graph.zip'):
            os.system('wget https://mind201910.blob.core.windows.net/knowledge-graph/wikidata-graph.zip -P %s/download' % dataset_root)
        os.system('unzip %s/download/wikidata-graph.zip -d %s/download' % (dataset_root, dataset_root))
    if mode == 'large':
        for data in ['train', 'dev', 'test']:
            if not os.path.exists(os.path.join(MIND_large_dataset_root, data, 'news.tsv')):
                shutil.copyfile(os.path.join(MIND_large_dataset_root, 'download', data, 'news.tsv'),
                                os.path.join(MIND_large_dataset_root, data, 'news.tsv'))
            if not os.path.exists(os.path.join(MIND_large_dataset_root, data, 'behaviors.tsv')):
                shutil.copyfile(os.path.join(MIND_large_dataset_root, 'download', data, 'behaviors.tsv'),
                                os.path.join(MIND_large_dataset_root, data, 'behaviors.tsv'))


def split_training_behaviors():
    MIND_small_train_ratio = 0.95
    train_behavior_lines = []
    dev_behavior_lines = []
    behavior_lines = []
    with open(MIND_small_dataset_root + '/download/train/behaviors.tsv', 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip()) > 0:
                behavior_lines.append(line)
    random.shuffle(behavior_lines)

    behavior_num = len(behavior_lines)
    behavior_id = [i for i in range(behavior_num)]
    random.shuffle(behavior_id)
    train_num = int(behavior_num * MIND_small_train_ratio)
    train_behavior_id = random.sample(behavior_id, train_num)
    train_behavior_id = set(train_behavior_id)
    for i, line in enumerate(behavior_lines):
        if i in train_behavior_id:
            train_behavior_lines.append(line)
        else:
            dev_behavior_lines.append(line)
    return train_behavior_lines, dev_behavior_lines


def preprocess_MIND_small():
    train_behavior_lines, dev_behavior_lines = split_training_behaviors()

    # train set
    train_set_root = MIND_small_dataset_root + '/train'
    if not os.path.exists(train_set_root):
        os.mkdir(train_set_root)
    with open(train_set_root + '/behaviors.tsv', 'w', encoding='utf-8') as f:
        for line in train_behavior_lines:
            f.write(line)
    if not os.path.exists(train_set_root + '/news.tsv'):
        shutil.copyfile(MIND_small_dataset_root + '/download/train/news.tsv', train_set_root + '/news.tsv')

    # dev set
    dev_set_root = MIND_small_dataset_root + '/dev'
    if not os.path.exists(dev_set_root):
        os.mkdir(dev_set_root)
    with open(dev_set_root + '/behaviors.tsv', 'w', encoding='utf-8') as f:
        for line in dev_behavior_lines:
            f.write(line)
    if not os.path.exists(dev_set_root):
        os.mkdir(dev_set_root)
    if not os.path.exists(dev_set_root + '/news.tsv'):
        shutil.copyfile(MIND_small_dataset_root + '/download/train/news.tsv', dev_set_root + '/news.tsv')

    # test set
    test_set_root = MIND_small_dataset_root + '/test'
    if not os.path.exists(test_set_root):
        os.mkdir(test_set_root)
    if not os.path.exists(test_set_root + '/behaviors.tsv'):
        shutil.copyfile(MIND_small_dataset_root + '/download/dev/behaviors.tsv', test_set_root + '/behaviors.tsv')
    if not os.path.exists(test_set_root + '/news.tsv'):
        shutil.copyfile(MIND_small_dataset_root + '/download/dev/news.tsv', test_set_root + '/news.tsv')


def sampling_MIND_dataset():
    sample_num = 200000
    source_root = MIND_large_dataset_root
    ensure_dir(MIND_200k_dataset_root)
    for split in ['train', 'dev', 'test']:
        ensure_dir(os.path.join(MIND_200k_dataset_root, split))

    # 1. randomly sample users from MIND-large
    user_set = set()
    for split in ['train', 'dev']:
        with open(os.path.join(source_root, split, 'behaviors.tsv'), 'r', encoding='utf-8') as f:
            for line in f:
                impression_ID, user_ID, time, history, impressions = line.strip().split('\t')
                user_set.add(user_ID)
    user_list = list(user_set)
    random.shuffle(user_list)
    assert sample_num <= len(user_list), 'sample num must be less than or equal to user count in MIND-large'
    sample_user_list = random.sample(user_list, sample_num)
    with open(os.path.join(MIND_200k_dataset_root, 'sample_users.json'), 'w', encoding='utf-8') as f:
        json.dump(sample_user_list, f)
    sampled_user_set = set(sample_user_list)

    # 2. write sampled behavior files (train/dev/test from MIND-large)
    with open(os.path.join(source_root, 'train', 'behaviors.tsv'), 'r', encoding='utf-8') as f:
        with open(os.path.join(MIND_200k_dataset_root, 'train', 'behaviors.tsv'), 'w', encoding='utf-8') as out_f:
            for line in f:
                impression_ID, user_ID, time, history, impressions = line.strip().split('\t')
                if user_ID in sampled_user_set:
                    out_f.write(line)

    source_test_path = os.path.join(source_root, 'test', 'behaviors.tsv')
    if os.path.exists(source_test_path):
        for split in ['dev', 'test']:
            with open(os.path.join(source_root, split, 'behaviors.tsv'), 'r', encoding='utf-8') as f:
                with open(os.path.join(MIND_200k_dataset_root, split, 'behaviors.tsv'), 'w', encoding='utf-8') as out_f:
                    for line in f:
                        impression_ID, user_ID, time, history, impressions = line.strip().split('\t')
                        if user_ID in sampled_user_set:
                            out_f.write(line)
    else:
        cnt = 0
        with open(os.path.join(source_root, 'dev', 'behaviors.tsv'), 'r', encoding='utf-8') as f:
            with open(os.path.join(MIND_200k_dataset_root, 'dev', 'behaviors.tsv'), 'w', encoding='utf-8') as dev_f:
                with open(os.path.join(MIND_200k_dataset_root, 'test', 'behaviors.tsv'), 'w', encoding='utf-8') as test_f:
                    for line in f:
                        impression_ID, user_ID, time, history, impressions = line.strip().split('\t')
                        if user_ID in sampled_user_set:
                            if cnt % 2 == 0:
                                dev_f.write(line)
                            else:
                                test_f.write(line)
                            cnt += 1

    # 3. copy other files from MIND-large
    for split in ['train', 'dev', 'test']:
        news_src = os.path.join(source_root, split, 'news.tsv')
        news_dst = os.path.join(MIND_200k_dataset_root, split, 'news.tsv')
        if os.path.exists(news_src) and not os.path.exists(news_dst):
            shutil.copyfile(news_src, news_dst)
        for name in ['entity_embedding.vec', 'context_embedding.vec', 'relation_embedding.vec']:
            src = os.path.join(source_root, split, name)
            dst = os.path.join(MIND_200k_dataset_root, split, name)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copyfile(src, dst)
    if not os.path.exists(os.path.join(MIND_200k_dataset_root, 'test', 'news.tsv')):
        dev_news_src = os.path.join(source_root, 'dev', 'news.tsv')
        if os.path.exists(dev_news_src):
            shutil.copyfile(dev_news_src, os.path.join(MIND_200k_dataset_root, 'test', 'news.tsv'))
    for name in ['entity_embedding.vec', 'context_embedding.vec', 'relation_embedding.vec']:
        test_dst = os.path.join(MIND_200k_dataset_root, 'test', name)
        if not os.path.exists(test_dst):
            dev_src = os.path.join(source_root, 'dev', name)
            if os.path.exists(dev_src):
                shutil.copyfile(dev_src, test_dst)


def generate_knowledge_entity_embedding(data_mode):
    assert data_mode in ['200k', 'small', 'large']
    # 1. copy entity embedding file
    if not os.path.exists(os.path.join(BASE_DATA_ROOT, 'MIND_%s' % data_mode, 'train', 'entity_embedding.vec')):
        shutil.copyfile(os.path.join(BASE_DATA_ROOT, 'MIND_%s' % data_mode, 'download', 'train', 'entity_embedding.vec'),
                        os.path.join(BASE_DATA_ROOT, 'MIND_%s' % data_mode, 'train', 'entity_embedding.vec'))
    if not os.path.exists(os.path.join(BASE_DATA_ROOT, 'MIND_%s' % data_mode, 'dev', 'entity_embedding.vec')):
        shutil.copyfile(os.path.join(BASE_DATA_ROOT, 'MIND_%s' % data_mode, 'download', 'dev', 'entity_embedding.vec'),
                        os.path.join(BASE_DATA_ROOT, 'MIND_%s' % data_mode, 'dev', 'entity_embedding.vec'))
    if data_mode in ['200k', 'small']:
        if not os.path.exists(os.path.join(BASE_DATA_ROOT, 'MIND_%s' % data_mode, 'test', 'entity_embedding.vec')):
            shutil.copyfile(os.path.join(BASE_DATA_ROOT, 'MIND_%s' % data_mode, 'download', 'dev', 'entity_embedding.vec'),
                            os.path.join(BASE_DATA_ROOT, 'MIND_%s' % data_mode, 'test', 'entity_embedding.vec'))
    else:
        if not os.path.exists(os.path.join(MIND_large_dataset_root, 'test', 'entity_embedding.vec')):
            shutil.copyfile(os.path.join(MIND_large_dataset_root, 'download', 'test', 'entity_embedding.vec'),
                            os.path.join(MIND_large_dataset_root, 'test', 'entity_embedding.vec'))
    # 2. generate context embedding file
    entity_embeddings = {}
    entity_embedding_files = [os.path.join(BASE_DATA_ROOT, 'MIND_%s' % data_mode, mode, 'entity_embedding.vec') for mode in ['train', 'dev', 'test']]
    for entity_embedding_file in entity_embedding_files:
        with open(entity_embedding_file, 'r', encoding='utf-8') as f:
            for line in f:
                if len(line.strip()) > 0:
                    terms = line.strip().split('\t')
                    assert len(terms) == 101
                    entity_embeddings[terms[0]] = list(map(float, terms[1:]))
    entity_embedding_relation = collections.defaultdict(set)
    wikidata_graph_path = os.path.join(BASE_DATA_ROOT, 'wikidata-graph', 'wikidata-graph.tsv')
    if not os.path.exists(wikidata_graph_path):
        wikidata_graph_path = os.path.join(BASE_DATA_ROOT, 'MIND_%s' % data_mode, 'download', 'wikidata-graph', 'wikidata-graph.tsv')
    with open(wikidata_graph_path, 'r', encoding='utf-8') as wikidata_graph_f:
        for line in wikidata_graph_f:
            if len(line.strip()) > 0:
                terms = line.strip().split('\t')
                entity_embedding_relation[terms[0]].add(terms[2])
                entity_embedding_relation[terms[2]].add(terms[0])
    context_embeddings = {}
    for entity in entity_embeddings:
        entity_embedding = entity_embeddings[entity]
        context_embedding = [entity_embedding[i] for i in range(100)]
        cnt = 1
        for _entity in entity_embedding_relation[entity]:
            if _entity in entity_embeddings:
                embedding = entity_embeddings[_entity]
                for i in range(100):
                    context_embedding[i] += embedding[i]
                cnt += 1
        for i in range(100):
            context_embedding[i] /= cnt
        context_embeddings[entity] = context_embedding
    for mode in ['train', 'dev', 'test']:
        with open(os.path.join(BASE_DATA_ROOT, 'MIND_%s' % data_mode, mode, 'entity_embedding.vec'), 'r', encoding='utf-8') as entity_embedding_f:
            with open(os.path.join(BASE_DATA_ROOT, 'MIND_%s' % data_mode, mode, 'context_embedding.vec'), 'w', encoding='utf-8') as context_embedding_f:
                for line in entity_embedding_f:
                    if len(line.strip()) > 0:
                        entity = line.split('\t')[0]
                        context_embedding_f.write(entity + '\t' + '\t'.join(list(map(str, context_embeddings[entity]))) + '\n')


def prepare_MIND_small():
    # download_extract_MIND_small()
    preprocess_MIND_small()
    generate_knowledge_entity_embedding('small')


def prepare_MIND_large():
    # download_extract_MIND_large('large')
    generate_knowledge_entity_embedding('large')


def prepare_MIND_200k():
    # download_extract_MIND_large('200k')
    sampling_MIND_dataset()
    generate_knowledge_entity_embedding('200k')


def sampling_MIND_codetest():
    sample_num = 5000
    source_root = MIND_small_dataset_root
    ensure_dir(MIND_codetest_dataset_root)
    for split in ['train', 'dev', 'test']:
        ensure_dir(os.path.join(MIND_codetest_dataset_root, split))

    # 1. randomly sample users from MIND-small
    user_set = set()
    for split in ['train', 'dev']:
        with open(os.path.join(source_root, split, 'behaviors.tsv'), 'r', encoding='utf-8') as f:
            for line in f:
                impression_ID, user_ID, time, history, impressions = line.strip().split('\t')
                user_set.add(user_ID)
    user_list = list(user_set)
    random.shuffle(user_list)
    assert sample_num <= len(user_list), 'sample num must be less than or equal to user count in MIND-small'
    sample_user_list = random.sample(user_list, sample_num)
    with open(os.path.join(MIND_codetest_dataset_root, 'sample_users.json'), 'w', encoding='utf-8') as f:
        json.dump(sample_user_list, f)
    sampled_user_set = set(sample_user_list)

    # 2. write sampled behavior files (train/dev/test from MIND-small)
    for split in ['train', 'dev', 'test']:
        with open(os.path.join(source_root, split, 'behaviors.tsv'), 'r', encoding='utf-8') as f:
            with open(os.path.join(MIND_codetest_dataset_root, split, 'behaviors.tsv'), 'w', encoding='utf-8') as out_f:
                for line in f:
                    impression_ID, user_ID, time, history, impressions = line.strip().split('\t')
                    if user_ID in sampled_user_set:
                        out_f.write(line)

    # 3. copy other files from MIND-small
    for split in ['train', 'dev', 'test']:
        news_src = os.path.join(source_root, split, 'news.tsv')
        news_dst = os.path.join(MIND_codetest_dataset_root, split, 'news.tsv')
        if os.path.exists(news_src) and not os.path.exists(news_dst):
            shutil.copyfile(news_src, news_dst)
        for name in ['entity_embedding.vec', 'context_embedding.vec', 'relation_embedding.vec']:
            src = os.path.join(source_root, split, name)
            dst = os.path.join(MIND_codetest_dataset_root, split, name)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copyfile(src, dst)


def prepare_MIND_codetest():
    if not os.path.exists(os.path.join(MIND_small_dataset_root, 'train', 'behaviors.tsv')):
        preprocess_MIND_small()
    sampling_MIND_codetest()


if __name__ == '__main__':
    prepare_MIND_small()
    prepare_MIND_large()
    prepare_MIND_200k()
    prepare_MIND_codetest()
