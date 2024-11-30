import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from django.db import transaction, connection

from .models import Essay
from .models import Edge

_papers_cache = None
_feats_cache = None
_edges_cache = None

DATA_DIR = Path(__file__).resolve().parent.parent / 'dataset'

# Read abstract, category, year of each paper
def load_papers():
    global _papers_cache
    if _papers_cache is None:
        _papers_cache = pd.read_csv(DATA_DIR / 'papers.csv.gz', compression='gzip')
    return _papers_cache

# Read the embedding vector of each paper
def load_features():
    global _feats_cache
    if _feats_cache is None:
        _feats_cache = pd.read_csv(DATA_DIR / 'feats.csv.gz', compression='gzip', header=None).values.astype(np.float32)
    return _feats_cache

# Read the citation relations between papers
def load_edges():
    global _edges_cache
    if _edges_cache is None:
        edges = pd.read_csv(DATA_DIR / 'edges.csv.gz', compression='gzip', header=None).values.T.astype(np.int32)
        _edges_cache = edges[0], edges[1]
    return _edges_cache

@transaction.atomic
def load_essays_into_db():
    """
    将论文和特征向量数据加载到数据库中
    """
    papers_df = load_papers()
    features = load_features()

    # 检查论文和特征向量数量是否一致
    if len(papers_df) != len(features):
        raise ValueError("The number of papers and features do not match!")

    essays = []
    for index, row in tqdm(papers_df.iterrows(), total=len(papers_df), desc="Loading papers"):
        essays.append(
            Essay(
                title=row['title'],
                abstract=row['abstract'],
                year=row['year'],
                category=row.get('category', None),  # 如果没有分类标签，则默认为 None
                feature_vector=features[index].tobytes()  # 将特征向量转换为二进制格式
            )
        )

    # 批量插入数据库
    Essay.objects.bulk_create(essays, batch_size=1000)
    print(f"{len(essays)} papers have been successfully loaded into the database!")

@transaction.atomic
def load_edges_into_db():
    """
    将引用关系加载到数据库中
    """
    edges = load_edges()  # 加载引用数据
    citing, cited = edges

    if len(citing) != len(cited):
        raise ValueError("referring and referred papers do not match!")

    # 检查所有引用的论文是否存在
    # invalid_citing_ids = [essay_id for essay_id in citing if not Essay.objects.filter(id=essay_id).exists()]
    # invalid_cited_ids = [essay_id for essay_id in cited if not Essay.objects.filter(id=essay_id).exists()]

    # if invalid_citing_ids or invalid_cited_ids:
    #     return ValueError(f"Invalid essay IDs found: citing - {invalid_citing_ids}, cited - {invalid_cited_ids}")

    edge_objects = []
    for essay_id, cited_id in tqdm(zip(citing, cited), total=len(citing), desc="Loading edges"):
        # print("essay_id:", essay_id, "cited_id:", cited_id)
        edge_objects.append(
            Edge(essay_id=essay_id, cited_id=cited_id)
        )

    # 批量插入引用关系
    Edge.objects.bulk_create(edge_objects, batch_size=1000)
    print(f"{len(edge_objects)} references have been successfully loaded into the database!")