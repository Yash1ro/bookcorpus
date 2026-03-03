#!/usr/bin/env python3
"""
BookCorpus 数据集生成脚本
论文: "Reveal Hidden Pitfalls and Navigate Next Generation of VSS" (arXiv:2512.12980)

处理流程:
  1. 读取 books_large_p1.txt + books_large_p2.txt
  2. 按每 8 句拼接为一个段落，共约 9,250,529 段
  3. 用 Stella_en_1.5B_v5 编码为 1024 维向量
  4. 随机抽取 10,000 段作为 query，其余为 database
  5. FAISS 精确 L2 搜索生成 top-100 ground truth

输出 (与现有数据集格式一致):
  data/bookcorpus_base.bin          float32, (N_db, 1024)
  data/bookcorpus_query.bin         float32, (10000, 1024)
  data/bookcorpus_groundtruth.bin   int32,   (10000, 100)
  data/bookcorpus_base.txt          每行 1024 个浮点数
  data/bookcorpus_query.txt         每行 1024 个浮点数
  data/bookcorpus_groundtruth.txt   每行 100 个整数 ID
  data/bookcorpus_paragraphs_db.txt    database 段落原文（便于复查）
  data/bookcorpus_paragraphs_query.txt query 段落原文
"""

import os
import sys
import random
import numpy as np
import torch
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────────────────────
CORPUS_FILES = [
    "books_large_p1.txt",
    "books_large_p2.txt",
]
DATA_DIR        = os.path.dirname(os.path.abspath(__file__))
SENTENCES_PER_PARA = 8          # 论文: 每段 8 句
N_QUERY            = 10_000     # 论文: 10,000 query
TOP_K              = 100
EMBED_MODEL        = "NovaSearch/stella_en_1.5B_v5"
EMBED_DIM          = 1024
BATCH_SIZE         = 32         # RTX 3090 + Stella 1.5B fp16 安全值
RANDOM_SEED        = 42
MAX_PARA           = None       # 设 None 使用全量；调试时可设为 50000

OUT_BASE_BIN   = os.path.join(DATA_DIR, "bookcorpus_base.bin")
OUT_QUERY_BIN  = os.path.join(DATA_DIR, "bookcorpus_query.bin")
OUT_GT_BIN     = os.path.join(DATA_DIR, "bookcorpus_groundtruth.bin")
OUT_BASE_TXT   = os.path.join(DATA_DIR, "bookcorpus_base.txt")
OUT_QUERY_TXT  = os.path.join(DATA_DIR, "bookcorpus_query.txt")
OUT_GT_TXT     = os.path.join(DATA_DIR, "bookcorpus_groundtruth.txt")
OUT_PARA_DB    = os.path.join(DATA_DIR, "bookcorpus_paragraphs_db.txt")
OUT_PARA_QUERY = os.path.join(DATA_DIR, "bookcorpus_paragraphs_query.txt")


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: 读取并分段
# ──────────────────────────────────────────────────────────────────────────────
def build_paragraphs(corpus_files: list, sentences_per_para: int,
                     max_para=None) -> list:
    """将 BookCorpus 行合并为固定句数的段落。"""
    print("=== Step 1: 读取语料并分段 ===")
    sentences = []
    for fname in corpus_files:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  [警告] 文件不存在，跳过: {fpath}")
            continue
        print(f"  读取: {fpath}")
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    sentences.append(line)

    print(f"  总句数: {len(sentences):,}")
    paragraphs = []
    for i in range(0, len(sentences) - sentences_per_para + 1, sentences_per_para):
        para = " ".join(sentences[i: i + sentences_per_para])
        paragraphs.append(para)
        if max_para and len(paragraphs) >= max_para:
            break

    print(f"  总段落数: {len(paragraphs):,}  (论文目标: ~9,250,529)")
    return paragraphs


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: 分割 db / query
# ──────────────────────────────────────────────────────────────────────────────
def split_db_query(paragraphs: list, n_query: int, seed: int):
    rng = random.Random(seed)
    indices = list(range(len(paragraphs)))
    query_idx = sorted(rng.sample(indices, n_query))
    query_set = set(query_idx)
    db_idx    = [i for i in indices if i not in query_set]

    db_paras    = [paragraphs[i] for i in db_idx]
    query_paras = [paragraphs[i] for i in query_idx]
    print(f"  Database: {len(db_paras):,} 段，Query: {len(query_paras):,} 段")
    return db_paras, query_paras


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: 编码
# ──────────────────────────────────────────────────────────────────────────────
def load_stella(model_name: str):
    """加载 Stella_en_1.5B_v5，返回 (model, tokenizer, device)。"""
    from transformers import AutoTokenizer, AutoModel

    print(f"  加载模型: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  使用设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model     = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    return model, tokenizer, device


def encode_texts(texts: list, model, tokenizer, device,
                 batch_size: int, desc: str = "编码",
                 out_bin: str = None, embed_dim: int = 1024) -> np.ndarray:
    """批量编码文本，增量写入 memmap 文件，避免内存溢出。"""
    n = len(texts)
    # 用 memmap 直接写盘，避免把全量向量放在内存
    mm = np.memmap(out_bin, dtype=np.float32, mode='w+', shape=(n, embed_dim))

    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        batch = texts[start:end]
        if start % (batch_size * 50) == 0:
            pct = start / n * 100
            print(f"  {desc}: {start:,}/{n:,} ({pct:.1f}%)", flush=True)

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            vecs = outputs.last_hidden_state[:, 0, :]  # (B, hidden)
            # Stella Matryoshka: 截取前 embed_dim 维
            vecs = vecs[:, :embed_dim]

        mm[start:end] = vecs.cpu().float().numpy()
        torch.cuda.empty_cache()

    mm.flush()
    print(f"  {desc}: 完成 {n:,} 条", flush=True)
    # 返回只读 memmap 供后续使用
    return np.memmap(out_bin, dtype=np.float32, mode='r', shape=(n, embed_dim))


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Ground Truth (FAISS 精确 L2)
# ──────────────────────────────────────────────────────────────────────────────
def build_ground_truth(db_vecs: np.ndarray, query_vecs: np.ndarray,
                       top_k: int) -> np.ndarray:
    """使用 FAISS IndexFlatL2 精确搜索，返回 int32 (nq, top_k)。"""
    import faiss

    d = db_vecs.shape[1]
    print(f"  构建 FAISS IndexFlatL2  d={d}, n_db={db_vecs.shape[0]:,}")

    cpu_index = faiss.IndexFlatL2(d)
    try:
        res   = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        print("  使用 GPU 搜索")
    except Exception:
        index = cpu_index
        print("  使用 CPU 搜索")

    index.add(np.ascontiguousarray(db_vecs, dtype=np.float32))
    print(f"  索引共 {index.ntotal:,} 个向量，开始搜索 top-{top_k} ...")
    _, I = index.search(np.ascontiguousarray(query_vecs, dtype=np.float32), top_k)
    return I.astype(np.int32)


# ──────────────────────────────────────────────────────────────────────────────
# Step 5: 保存
# ──────────────────────────────────────────────────────────────────────────────
def save_bin(arr: np.ndarray, path: str):
    arr.tofile(path)
    print(f"  保存: {path}  shape={arr.shape} dtype={arr.dtype}")


def save_txt_float(arr: np.ndarray, path: str):
    """每行空格分隔浮点数（与 glove100_base.txt 格式一致）。"""
    with open(path, "w") as f:
        for row in arr:
            f.write(" ".join(f"{v:.8f}" for v in row) + "\n")
    print(f"  保存: {path}")


def save_txt_int(arr: np.ndarray, path: str):
    """每行空格分隔整数（与 glove100_groundtruth.txt 格式一致）。"""
    with open(path, "w") as f:
        for row in arr:
            f.write(" ".join(str(v) for v in row) + "\n")
    print(f"  保存: {path}")


def save_paragraphs(paras: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(paras))
    print(f"  保存: {path}  ({len(paras):,} 段)")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ── Step 1: 分段 ──────────────────────────────────────────────────────────
    paragraphs = build_paragraphs(CORPUS_FILES, SENTENCES_PER_PARA, MAX_PARA)

    # ── Step 2: 分割 ──────────────────────────────────────────────────────────
    print("\n=== Step 2: 分割 database / query ===")
    db_paras, query_paras = split_db_query(paragraphs, N_QUERY, RANDOM_SEED)
    del paragraphs  # 释放内存

    # ── Step 3: 编码 ──────────────────────────────────────────────────────────
    print("\n=== Step 3: 加载 Stella 模型并编码 ===")
    model, tokenizer, device = load_stella(EMBED_MODEL)

    print(f"\n  编码 database ({len(db_paras):,} 段) ...")
    db_vecs = encode_texts(db_paras, model, tokenizer, device, BATCH_SIZE,
                           "DB", out_bin=OUT_BASE_BIN, embed_dim=EMBED_DIM)

    print(f"\n  编码 query ({len(query_paras):,} 段) ...")
    query_vecs = encode_texts(query_paras, model, tokenizer, device, BATCH_SIZE,
                              "Query", out_bin=OUT_QUERY_BIN, embed_dim=EMBED_DIM)

    # ── Step 4: Ground Truth ─────────────────────────────────────────────────
    print("\n=== Step 4: 生成 Top-100 Ground Truth (L2) ===")
    gt = build_ground_truth(db_vecs, query_vecs, TOP_K)

    # ── Step 5: 保存 ─────────────────────────────────────────────────────────
    print("\n=== Step 5: 保存文件 ===")
    # db_vecs / query_vecs 已在编码阶段直接写入 .bin，无需再次保存
    print(f"  已保存: {OUT_BASE_BIN}  shape={db_vecs.shape}")
    print(f"  已保存: {OUT_QUERY_BIN}  shape={query_vecs.shape}")
    save_bin(gt,         OUT_GT_BIN)

    save_txt_float(db_vecs,    OUT_BASE_TXT)
    save_txt_float(query_vecs, OUT_QUERY_TXT)
    save_txt_int(gt,           OUT_GT_TXT)

    save_paragraphs(db_paras,    OUT_PARA_DB)
    save_paragraphs(query_paras, OUT_PARA_QUERY)

    print("\n=== 完成 ===")
    print(f"  database 向量: {db_vecs.shape},  {db_vecs.nbytes/1024**3:.2f} GB")
    print(f"  query 向量:    {query_vecs.shape}")
    print(f"  ground truth:  {gt.shape}")
    print(f"  示例 GT[0]:    {gt[0, :10]}")


if __name__ == "__main__":
    main()
