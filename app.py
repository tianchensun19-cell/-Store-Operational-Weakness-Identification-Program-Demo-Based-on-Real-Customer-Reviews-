import io
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import jieba
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# ========================
# 0. 页面配置
# ========================
st.set_page_config(page_title="ReviewPulse CN Pro", layout="wide")

st.title("📊 门店运营短板分析")
st.subheader("基于真实顾客评论的AI舆情风控驾驶舱")

TARGET_YEAR = 2026
TOPK_KEYWORDS = 8
MIN_REQUIRED_ROWS = 5

@st.dialog("欢迎使用 📊门店运营短板分析产品Demo")
def show_intro_dialog():
    st.markdown("### 这个产品能做什么")
    st.markdown("""
- 自动识别门店评论中的主要风险问题，如口味、配送、漏送、服务、包装、价格等。
- 将本期评论同时与固定 baseline 和上一期结果对比，帮助判断是长期问题还是新近恶化问题。
- 输出最严重问题、最优先处理问题、恶化最快问题，并提供中文整改建议。
- 对小样本问题进行平滑处理，避免 1 条差评就把风险率抬到 100% 造成误判。
""")
    st.markdown("### 数据集来源说明")
    st.markdown("""
- Baseline 数据：本地固定的 `baseline.xlsx`，作为历史基线结构，不会在运行中被自动更新。
- 当前期数据：由用户手动上传的 CSV / XLSX 评论数据集，需至少包含 `评价内容`，建议包含 `评价类型`。
- 历史期数据：来自当前会话中用户之前已上传并完成分析的第 1 期、第 2 期等数据。
""")
    st.info("系统会在你阅读本说明时预加载 baseline，因此关闭弹窗后通常无需再长时间等待基线分析。")
    if st.button("我已了解，开始使用", use_container_width=True):
        st.session_state.app_stage = "booting"
        st.rerun()


# ========================
# A. Session State 初始化
# ========================
if "current_period" not in st.session_state:
    st.session_state.current_period = 1
if "period_history" not in st.session_state:
    st.session_state.period_history = {}
if "manual_focus_clusters" not in st.session_state:
    st.session_state.manual_focus_clusters = []
if "last_ai_summary" not in st.session_state:
    st.session_state.last_ai_summary = ""
if "last_processed_upload_key" not in st.session_state:
    st.session_state.last_processed_upload_key = None
if "current_analysis_result" not in st.session_state:
    st.session_state.current_analysis_result = None
if "latest_completed_period" not in st.session_state:
    st.session_state.latest_completed_period = 0
if "pending_period" not in st.session_state:
    st.session_state.pending_period = 1
if "app_stage" not in st.session_state:
    st.session_state.app_stage = "intro"
if "baseline_ready" not in st.session_state:
    st.session_state.baseline_ready = False

# ========================
# 1. 模型
# ========================
@st.cache_resource

def load_model():
    return SentenceTransformer("BAAI/bge-small-zh-v1.5")


model = load_model()

# ========================
# 2. 标签
# ========================
ASPECT_LABELS = [
    "送餐速度慢",
    "等待时间过长",
    "配送错误或漏送",
    "服务态度差",
    "食品质量问题",
    "食物不新鲜",
    "食物温度异常",
    "口味不符预期",
    "分量不足",
    "包装问题",
    "价格性价比低",
    "环境卫生差",
    "门店秩序混乱",
    "设施体验差",
    "其他零散问题",
]


@st.cache_data

def get_label_embeddings(labels):
    return model.encode(list(labels), normalize_embeddings=True)


# ========================
# 3. 基础函数
# ========================
def safe_read_table(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    rename_map = {}
    for c in df.columns:
        c_str = str(c).strip()
        if c_str.lower() in ["评价内容", "content", "comment", "评论内容"]:
            rename_map[c] = "评价内容"
        if c_str in ["评价类型", "review_type", "sentiment"]:
            rename_map[c] = "评价类型"
    if rename_map:
        df = df.rename(columns=rename_map)

    if "评价内容" not in df.columns:
        if len(df.columns) == 1:
            df.columns = ["评价内容"]
        else:
            raise ValueError("上传文件必须包含 评价内容 列")

    if "评价类型" not in df.columns:
        st.warning("未检测到“评价类型”列，系统将默认中差评率为 0")

    df["评价内容"] = df["评价内容"].fillna("").astype(str).str.strip()
    df = df[df["评价内容"] != ""].reset_index(drop=True)
    return df


def get_embeddings(texts):
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def get_stopwords():
    return {
        "真的", "就是", "感觉", "有点", "这个", "那个", "我们", "你们", "他们", "然后", "而且",
        "不是", "一个", "一下", "还是", "非常", "比较", "特别", "没有", "什么", "这家", "店里",
        "门店", "东西", "可以", "一般", "但是", "因为", "所以", "一下子", "一直", "已经", "问题",
        "评论", "觉得", "还有", "时候", "今天", "昨天", "这次", "服务", "态度", "速度", "配送",
        "包装", "价格", "环境", "卫生", "口味", "店员", "外卖", "吃的", "食物", "一下下", "一个个",
        "一下子", "特别特别", "还是会", "有一些", "不是很", "有时候", "我们家", "他们家", "一下吧"
    }


def get_cluster_keywords(df, cluster_id, topk=TOPK_KEYWORDS):
    texts = df[df["cluster"] == cluster_id]["评价内容"].fillna("").astype(str)
    words = []
    for t in texts:
        words.extend(jieba.lcut(t))
    stopwords = get_stopwords()
    words = [w for w in words if len(w) >= 2 and w.strip() and w not in stopwords and not re.fullmatch(r"\d+", w)]
    return [w for w, _ in Counter(words).most_common(topk)]


def classify_single_review_aspect(text):
    text = str(text)

    delivery_error_words = [
        "送错", "漏送", "少送", "没送", "没送来", "没收到", "未送达", "没给", "给错", "漏发", "少给",
        "发票没有", "没有发票", "发票没送", "赠品没有", "赠品没送", "订单已完成", "显示订单已完成",
        "订单完成", "送错餐", "送错了", "补上送来", "根本没有送", "没拿到饭"
    ]
    stale_words = ["不新鲜", "变质", "拉肚子", "馊", "坏了", "异味", "腥味"]
    temp_words = ["凉了", "冷了", "不热", "温的", "热乎", "温度", "面了"]
    quality_words = [
        "难吃", "不好吃", "苦", "夹生", "腻", "口感", "荤", "干", "硬", "奇葩", "没味道", "没味", "太甜",
        "太咸", "太淡", "不喜欢", "一般", "嚼不烂", "极差", "后悔", "失望", "不香", "太油", "太辣", "没熟",
        "味道", "不合口味", "不符合预期"
    ]
    speed_words = ["太慢", "很慢", "送餐慢", "送的太久", "迟到", "等了", "很久", "半小时", "一小时", "一个半小时", "两个小时", "三个小时", "送餐时间过长", "一个多小时", "两个多小时"]
    service_words = ["态度差", "服务差", "不接电话", "骂人", "叫唤", "凶"]
    portion_words = ["分量", "太少", "量少", "不够吃", "少了点", "没几片"]
    packaging_words = ["包装", "洒了", "漏了", "撒了", "破了", "盒子"]
    strong_price_words = ["太贵", "太坑", "坑爹", "不值", "不划算", "性价比低", "性价比差", "贵得离谱"]
    weak_price_words = ["贵", "便宜", "价格", "性价比"]

    if any(w in text for w in delivery_error_words):
        return "配送错误或漏送"
    if any(w in text for w in stale_words):
        return "食物不新鲜"
    if any(w in text for w in temp_words):
        return "食物温度异常"
    if any(w in text for w in quality_words):
        return "口味不符预期"
    if any(w in text for w in speed_words):
        return "送餐速度慢"
    if any(w in text for w in service_words):
        return "服务态度差"
    if any(w in text for w in portion_words):
        return "分量不足"
    if any(w in text for w in packaging_words):
        return "包装问题"

    if any(w in text for w in strong_price_words):
        return "价格性价比低"
    if any(w in text for w in weak_price_words):
        non_price_signals = delivery_error_words + stale_words + temp_words + quality_words + speed_words + service_words + portion_words + packaging_words
        if not any(w in text for w in non_price_signals):
            return "价格性价比低"

    return "其他零散问题"

def get_representative_texts(df, cluster_id, topn=3, preferred_aspect=None):
    sub = df[df["cluster"] == cluster_id].copy()
    if sub.empty or "评价内容" not in sub.columns:
        return []

    texts = sub["评价内容"].fillna("").astype(str).tolist()

    if preferred_aspect:
        filtered = [t for t in texts if classify_single_review_aspect(t) == preferred_aspect]
        if len(filtered) >= min(2, topn):
            texts = filtered

    if len(texts) <= topn:
        return texts

    try:
        embs = model.encode(texts, normalize_embeddings=True)
        center = np.mean(embs, axis=0, keepdims=True)
        sims = cosine_similarity(embs, center).reshape(-1)
        ranked = np.argsort(-sims)[:topn]
        return [texts[i] for i in ranked]
    except Exception:
        return texts[:topn]


def compress_text_for_label(text):
    text = str(text).strip()
    if len(text) <= 30:
        return text
    return text[:30]


def assign_aspect_name_ai(keywords_tuple: tuple, representative_texts=None) -> str:
    keywords = [str(k).strip() for k in keywords_tuple if str(k).strip()]
    representative_texts = [str(t).strip() for t in (representative_texts or []) if str(t).strip()]
    if not keywords and not representative_texts:
        return "其他零散问题"

    rep_text = " ".join(representative_texts[:2])
    kw_text = " ".join(keywords) + " " + rep_text

    strong_positive_words = {"不错", "好吃", "很快", "满意", "赞", "可以", "新鲜"}
    quality_negative_words = {"难吃", "不好吃", "变质", "拉肚子", "苦", "夹生", "腻", "异味", "馊", "坏了"}
    taste_negative_words = {"太咸", "太淡", "没味", "口味", "味道", "甜", "辣", "酸"}
    delay_words = {"太慢", "很慢", "送餐慢", "送的太久", "迟到", "等了", "很久", "半小时", "一小时", "一个半小时", "两个小时", "三个小时", "小时", "分钟", "送到", "时间"}
    wait_words = {"排队", "等待", "等餐", "催单", "海枯石烂"}
    delivery_error_words = {"送错", "漏送", "少送", "没给", "给错", "漏发", "少给", "餐具", "发票"}
    service_words = {"态度差", "服务差", "不接电话", "骂人", "叫唤", "凶"}
    packaging_words = {"包装", "洒了", "漏了", "撒了", "破了", "盒子"}
    portion_words = {"分量", "太少", "量少", "不够吃", "少了点"}
    temp_words = {"凉了", "冷了", "不热", "温的", "热乎", "温度"}
    price_words = {"太贵", "不值", "性价比", "高价"}

    keyword_set = set(keywords)

    def count_hits(word_set):
        pool = keywords + representative_texts
        return sum(1 for w in pool if w in word_set or any(token in w for token in word_set) or any(w in token for token in word_set))

    score = {
        "配送错误或漏送": count_hits(delivery_error_words),
        "食物不新鲜": count_hits({"不新鲜", "变质", "拉肚子", "馊", "坏了", "异味"}),
        "食物温度异常": count_hits(temp_words),
        "食品质量问题": count_hits(quality_negative_words | {"质量", "口感", "油", "菜品"}),
        "口味不符预期": count_hits(taste_negative_words),
        "分量不足": count_hits(portion_words),
        "送餐速度慢": count_hits(delay_words),
        "等待时间过长": count_hits(wait_words),
        "服务态度差": count_hits(service_words),
        "包装问题": count_hits(packaging_words),
        "价格性价比低": count_hits(price_words),
    }

    # 关键修正：遇到“太慢/小时/时间/送到”等延迟词时，优先判成速度问题；
    # “不错/好吃/很快”这类正向词不应把簇拉去口味标签。
    if score["配送错误或漏送"] >= 1:
        return "配送错误或漏送"
    if score["食物不新鲜"] >= 1:
        return "食物不新鲜"
    if score["食物温度异常"] >= 1 and score["送餐速度慢"] == 0:
        return "食物温度异常"
    if score["送餐速度慢"] >= 2 or (score["送餐速度慢"] >= 1 and any(w in kw_text for w in ["太慢", "小时", "送到", "时间", "等了"])):
        return "送餐速度慢"
    if score["等待时间过长"] >= 1:
        return "等待时间过长"
    if score["服务态度差"] >= 1:
        return "服务态度差"
    if score["包装问题"] >= 1:
        return "包装问题"
    if score["分量不足"] >= 1:
        return "分量不足"
    if score["价格性价比低"] >= 1:
        return "价格性价比低"
    if score["食品质量问题"] >= 1:
        return "食品质量问题"
    if score["口味不符预期"] >= 2 and score["送餐速度慢"] == 0 and score["食品质量问题"] == 0:
        return "口味不符预期"

    # 剔除明显正向词再做语义兜底，减少“不错/好吃/很快”误导标签。
    filtered_keywords = [w for w in keywords if w not in strong_positive_words]
    if not filtered_keywords:
        filtered_keywords = keywords
    rep_for_label = [compress_text_for_label(t) for t in representative_texts[:2]]
    label_source = " ".join(filtered_keywords[:6] + rep_for_label)
    if not label_source.strip():
        label_source = " ".join(filtered_keywords[:6])

    kw_emb = model.encode([label_source], normalize_embeddings=True)
    labels = tuple(ASPECT_LABELS)
    label_embs = get_label_embeddings(labels)
    sims = cosine_similarity(kw_emb, label_embs)[0]

    if len(sims) != len(labels):
        return " / ".join(filtered_keywords[:2]) if len(filtered_keywords) >= 2 else "其他零散问题"

    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])

    if best_idx >= len(labels):
        return " / ".join(filtered_keywords[:2]) if len(filtered_keywords) >= 2 else "其他零散问题"

    if best_sim < 0.68:
        fallback = filtered_keywords[:2] if len(filtered_keywords) >= 2 else [compress_text_for_label(t) for t in representative_texts[:1]]
        return " / ".join(fallback) if fallback else "其他零散问题"

    return labels[best_idx]


def deduplicate_aspect_names(df):
    df = df.copy()
    seen = {}
    for i in df.index:
        aspect = str(df.at[i, "aspect"])
        keywords = [k.strip() for k in str(df.at[i, "keywords"]).split(",") if k.strip()]
        current_set = set(keywords)
        if aspect not in seen:
            seen[aspect] = [current_set]
            continue
        overlap_max = max(len(current_set & prev_set) / max(1, len(current_set | prev_set)) for prev_set in seen[aspect])
        if overlap_max < 0.65 and len(keywords) >= 2 and aspect != "其他零散问题":
            df.at[i, "aspect"] = f"{aspect}（{keywords[0]}/{keywords[1]}）"
        seen.setdefault(aspect, []).append(current_set)
    return df


def calc_cluster_neg_ratio(df, cluster_id):
    sub = df[df["cluster"] == cluster_id]
    if len(sub) == 0 or "评价类型" not in sub.columns:
        return 0.0
    neg = sub["评价类型"].isin(["中评", "差评"]).sum()
    return float(neg / len(sub)) * 100


def calc_cluster_sample_count(df, cluster_id):
    return int((df["cluster"] == cluster_id).sum())


def remap_cluster_ids(df):
    df = df.copy()
    unique_ids = sorted(df["cluster"].unique())
    mapping = {old: new for new, old in enumerate(unique_ids)}
    df["cluster"] = df["cluster"].map(mapping)
    return df


def bayes_smoothed_rate_pct(neg_count, total_count, prior_rate=0.18, prior_strength=12):
    total_count = int(total_count)
    neg_count = int(neg_count)
    if total_count <= 0:
        return 0.0
    alpha = prior_rate * prior_strength
    beta = (1 - prior_rate) * prior_strength
    return float((neg_count + alpha) / (total_count + alpha + beta) * 100.0)


def sample_size_weight(total_count, full_weight_at=20):
    total_count = max(int(total_count), 0)
    return min(total_count / float(full_weight_at), 1.0)


def calc_cluster_stats(df, cluster_id):
    sub = df[df["cluster"] == cluster_id]
    total = len(sub)
    if total == 0:
        return {"total": 0, "neg": 0, "neg_rate": 0.0, "smoothed_neg_rate": 0.0}
    if "评价类型" not in sub.columns:
        return {"total": total, "neg": 0, "neg_rate": 0.0, "smoothed_neg_rate": 0.0}
    neg = int(sub["评价类型"].isin(["中评", "差评"]).sum())
    raw_rate = float(neg / total) * 100.0
    smoothed = bayes_smoothed_rate_pct(neg, total)
    return {"total": total, "neg": neg, "neg_rate": raw_rate, "smoothed_neg_rate": smoothed}


def classify_priority(smoothed_neg_rate, sample_count, change_vs_baseline, change_vs_prev, is_new):
    prev_change = 0 if pd.isna(change_vs_prev) else change_vs_prev
    weight = sample_size_weight(sample_count)
    if is_new and sample_count >= 5 and smoothed_neg_rate >= 20:
        return "高优先级"
    if (smoothed_neg_rate >= 45 and sample_count >= 8) or (change_vs_baseline >= 20 and sample_count >= 5) or (prev_change >= 15 and sample_count >= 5):
        return "高优先级"
    if (smoothed_neg_rate >= 25 and sample_count >= 5) or (change_vs_baseline >= 10 and weight >= 0.35) or (prev_change >= 8 and weight >= 0.35):
        return "中优先级"
    return "低优先级"


def action_advice(aspect):
    mapping = {
        "服务态度差": "建议抽检服务录音/监控、复盘服务话术，并对高频被点名班次做针对培训。",
        "送餐速度慢": "建议排查骑手调度、接单分发与高峰履约链路，识别超时集中时段。",
        "等待时间过长": "建议重点监控下单到送达的等待时长，并拆分高峰期瓶颈环节。",
        "配送错误或漏送": "建议复盘拣货、打包和交接流程，重点检查漏餐、错餐和餐具遗漏。",
        "食品质量问题": "建议优先排查出品标准、原料状态和保温时长，并回看同批次投诉样本。",
        "环境卫生差": "建议增加高峰时段巡检频次，重点检查桌面、地面、洗手区与垃圾点位。",
        "价格性价比低": "建议复盘价格感知来源，结合份量、套餐结构与促销展示一起优化。",
        "配送速度慢": "建议排查履约链路、接单节奏与高峰备餐能力，识别拥堵时段。",
        "食物不新鲜": "建议立即排查原料保质、库存周转和当日废弃机制，优先关注肉类与凉菜。",
        "食物温度异常": "建议检查保温链路、出餐等待时长和骑手取餐衔接。",
        "包装问题": "建议检查打包规范、封签稳定性与汤水类包装适配性。",
        "口味不符预期": "建议核查标准配方、门店执行偏差与顾客预期管理文案。",
        "分量不足": "建议复核标准克重、加料执行和门店份量一致性。",
        "噪音嘈杂": "建议识别高峰拥堵区域，优化排队动线、座位布局和广播音量。",
        "等待时间过长": "建议重点关注点单到取餐时长，并拆分高峰时段瓶颈环节。",
        "门店秩序混乱": "建议优化排队分流、取餐指引与现场人员站位。",
        "设施体验差": "建议检查座椅、空调、灯光、叫号屏及自助设备稳定性。",
        "其他零散问题": "建议继续观察样本增长情况，并结合原文人工复核是否形成稳定问题类。",
    }
    return mapping.get(aspect, "建议结合该类评论原文抽样复核，确认是流程、服务还是环境导致。")


def build_data_quality_report(df):
    total = len(df)
    empty_reviews = int(df["评价内容"].isna().sum()) if "评价内容" in df.columns else total
    duplicate_reviews = int(df["评价内容"].duplicated().sum()) if "评价内容" in df.columns else 0
    short_reviews = int((df["评价内容"].astype(str).str.len() < 6).sum()) if "评价内容" in df.columns else 0
    has_type = "评价类型" in df.columns
    neutral_neg = int(df["评价类型"].isin(["中评", "差评"]).sum()) if has_type else 0
    quality_flags = []
    if total < MIN_REQUIRED_ROWS:
        quality_flags.append("样本量偏少")
    if duplicate_reviews / max(1, total) > 0.15:
        quality_flags.append("重复评论偏多")
    if short_reviews / max(1, total) > 0.25:
        quality_flags.append("短文本偏多")
    if not has_type:
        quality_flags.append("缺少评价类型")
    return {
        "total": total,
        "empty_reviews": empty_reviews,
        "duplicate_reviews": duplicate_reviews,
        "short_reviews": short_reviews,
        "neutral_neg": neutral_neg,
        "has_type": has_type,
        "quality_flags": quality_flags,
    }


def build_export_bytes(summary_text, comparison_df, period_no):
    buffer = io.StringIO()
    buffer.write(f"ReviewPulse CN Pro 管理摘要\n")
    buffer.write(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    buffer.write(f"分析期数：{TARGET_YEAR}年第{period_no}期\n\n")
    buffer.write("一、AI摘要\n")
    buffer.write(summary_text + "\n\n")
    buffer.write("二、风险明细\n")
    if comparison_df is not None and not comparison_df.empty:
        buffer.write(comparison_df.to_csv(index=False))
    return buffer.getvalue().encode("utf-8-sig")


# ========================
# 4. DBSCAN 聚类
# ========================
def dbscan_cluster_embeddings(embeddings, eps=0.18, min_samples=5):
    n = len(embeddings)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([0])
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    return clustering.fit_predict(embeddings)


def attach_noise_to_major_clusters(df, embeddings, min_major_size=5, attach_threshold=0.72):
    df = df.copy()
    counts = df["cluster"].value_counts().to_dict()
    major_clusters = [cid for cid, cnt in counts.items() if cid != -1 and cnt >= min_major_size]
    if not major_clusters:
        return df
    centers = {}
    for cid in major_clusters:
        idxs = df.index[df["cluster"] == cid].tolist()
        centers[cid] = embeddings[idxs].mean(axis=0)
    noise_idxs = df.index[df["cluster"] == -1].tolist()
    for idx in noise_idxs:
        emb = embeddings[idx]
        best_target = None
        best_sim = -1.0
        for cid in major_clusters:
            sim = cosine_similarity([emb], [centers[cid]])[0][0]
            if sim > best_sim:
                best_sim = sim
                best_target = cid
        if best_target is not None and best_sim >= attach_threshold:
            df.at[idx, "cluster"] = best_target
    return df


def merge_tiny_clusters_to_major(df, embeddings, tiny_size=3, attach_threshold=0.74):
    df = df.copy()
    counts = df["cluster"].value_counts().to_dict()
    valid_clusters = [cid for cid in counts if cid != -1]
    if not valid_clusters:
        return df
    centers = {}
    for cid in valid_clusters:
        idxs = df.index[df["cluster"] == cid].tolist()
        centers[cid] = embeddings[idxs].mean(axis=0)
    major_clusters = [cid for cid, cnt in counts.items() if cid != -1 and cnt > tiny_size]
    tiny_clusters = [cid for cid, cnt in counts.items() if cid != -1 and cnt <= tiny_size]
    if not major_clusters:
        return df
    for cid in tiny_clusters:
        emb = centers[cid]
        best_target = None
        best_sim = -1.0
        for target in major_clusters:
            if target == cid:
                continue
            sim = cosine_similarity([emb], [centers[target]])[0][0]
            if sim > best_sim:
                best_sim = sim
                best_target = target
        if best_target is not None and best_sim >= attach_threshold:
            df.loc[df["cluster"] == cid, "cluster"] = best_target
    return df


def collect_remaining_sparse_to_other(df, min_size=3):
    df = df.copy()
    counts = df["cluster"].value_counts().to_dict()
    sparse_clusters = [cid for cid, cnt in counts.items() if cnt < min_size or cid == -1]
    if not sparse_clusters:
        return remap_cluster_ids(df)
    other_cluster_id = -999
    for cid in sparse_clusters:
        df.loc[df["cluster"] == cid, "cluster"] = other_cluster_id
    return remap_cluster_ids(df)


def reduce_duplicate_clusters(df, embeddings, center_threshold=0.84, keyword_overlap_threshold=0.68):
    df = df.copy()
    cluster_ids = sorted(df["cluster"].unique())
    if len(cluster_ids) <= 1:
        return df
    centers = {}
    keyword_sets = {}
    for cid in cluster_ids:
        idxs = df.index[df["cluster"] == cid].tolist()
        centers[cid] = embeddings[idxs].mean(axis=0)
        keyword_sets[cid] = set(get_cluster_keywords(df, cid, topk=TOPK_KEYWORDS))
    parent = {cid: cid for cid in cluster_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            c1, c2 = cluster_ids[i], cluster_ids[j]
            sim = cosine_similarity([centers[c1]], [centers[c2]])[0][0]
            kws1, kws2 = keyword_sets[c1], keyword_sets[c2]
            overlap = len(kws1 & kws2) / max(1, len(kws1 | kws2))
            if sim >= center_threshold or overlap >= keyword_overlap_threshold:
                union(c1, c2)
    mapping = {cid: find(cid) for cid in cluster_ids}
    df["cluster"] = df["cluster"].map(mapping)
    return remap_cluster_ids(df)


def cluster_reviews(embeddings):
    return dbscan_cluster_embeddings(embeddings, eps=0.18, min_samples=5)


# ========================
# 5. baseline 匹配
# ========================
def build_aspect_centers(df, embeddings):
    if df.empty:
        return {}
    centers = {}
    for aspect in sorted(df["aspect_rule"].dropna().unique()):
        idx = df.index[df["aspect_rule"] == aspect].tolist()
        if not idx:
            continue
        centers[aspect] = np.mean(embeddings[idx], axis=0)
    return centers


def assign_other_aspects_by_similarity(df, embeddings, centers):
    if not centers:
        return df
    df = df.copy()
    other_idx = df.index[df["aspect_rule"] == "其他零散问题"].tolist()
    if not other_idx:
        return df
    labels = list(centers.keys())
    center_mat = np.vstack([centers[k] for k in labels])
    sims = cosine_similarity(embeddings[other_idx], center_mat)
    for pos, idx in enumerate(other_idx):
        best_j = int(np.argmax(sims[pos]))
        best_sim = float(sims[pos][best_j])
        if best_sim >= 0.48:
            df.at[idx, "aspect_rule"] = labels[best_j]
    return df


def cluster_within_aspects(df, embeddings):
    df = df.copy()
    if len(df) == 0:
        df["cluster"] = []
        return df

    df["aspect_rule"] = df["评价内容"].fillna("").astype(str).apply(classify_single_review_aspect)
    centers = build_aspect_centers(df[df["aspect_rule"] != "其他零散问题"], embeddings)
    df = assign_other_aspects_by_similarity(df, embeddings, centers)

    cluster_values = np.full(len(df), -1, dtype=int)
    next_cluster = 0
    preferred_order = [
        "口味不符预期", "食品质量问题", "食物不新鲜", "食物温度异常",
        "送餐速度慢", "等待时间过长", "配送错误或漏送", "服务态度差",
        "分量不足", "包装问题", "价格性价比低", "其他零散问题"
    ]
    aspect_order = [a for a in preferred_order if a in set(df["aspect_rule"])] + [a for a in sorted(set(df["aspect_rule"])) if a not in preferred_order]

    for aspect in aspect_order:
        idx = df.index[df["aspect_rule"] == aspect].tolist()
        if not idx:
            continue
        sub_emb = embeddings[idx]
        if len(idx) < 6:
            cluster_values[idx] = next_cluster
            next_cluster += 1
            continue
        local_clusters = cluster_reviews(sub_emb)
        temp = df.loc[idx].copy().reset_index(drop=True)
        temp["cluster"] = local_clusters
        temp = attach_noise_to_major_clusters(temp, sub_emb, min_major_size=4, attach_threshold=0.70)
        temp = merge_tiny_clusters_to_major(temp, sub_emb, tiny_size=2, attach_threshold=0.72)
        temp = reduce_duplicate_clusters(temp, sub_emb, center_threshold=0.82, keyword_overlap_threshold=0.65)
        temp = collect_remaining_sparse_to_other(temp, min_size=2)
        local_unique = sorted(temp["cluster"].unique())
        mapping = {c: i + next_cluster for i, c in enumerate(local_unique)}
        remapped = temp["cluster"].map(mapping).values
        cluster_values[idx] = remapped
        next_cluster += len(local_unique)

    df["cluster"] = cluster_values
    return df


def match_to_baseline(baseline_df, current_df, similarity_threshold=0.76):
    baseline_clusters = sorted(baseline_df["cluster"].unique())
    current_clusters = sorted(current_df["cluster"].unique())

    def cluster_text(df, cid):
        kws = get_cluster_keywords(df, cid, topk=TOPK_KEYWORDS)
        return " ".join(kws) if kws else "其他零散问题"

    def cluster_aspect(df, cid):
        sub = df[df["cluster"] == cid]
        if sub.empty:
            return "其他零散问题"
        texts = sub["评价内容"].fillna("").astype(str).tolist()
        labels = [classify_single_review_aspect(t) for t in texts]
        if not labels:
            return "其他零散问题"
        counts = pd.Series(labels).value_counts()
        return str(counts.index[0])

    base_texts = [cluster_text(baseline_df, c) for c in baseline_clusters]
    curr_texts = [cluster_text(current_df, c) for c in current_clusters]
    base_aspects = {c: cluster_aspect(baseline_df, c) for c in baseline_clusters}
    curr_aspects = {c: cluster_aspect(current_df, c) for c in current_clusters}
    base_embs = model.encode(base_texts, normalize_embeddings=True)
    curr_embs = model.encode(curr_texts, normalize_embeddings=True)
    sim_matrix = cosine_similarity(curr_embs, base_embs)

    compatible_aspects = {
        "口味不符预期": {"口味不符预期", "食品质量问题", "食物不新鲜", "食物温度异常", "其他零散问题"},
        "食品质量问题": {"口味不符预期", "食品质量问题", "食物不新鲜", "食物温度异常", "其他零散问题"},
        "食物不新鲜": {"口味不符预期", "食品质量问题", "食物不新鲜", "食物温度异常", "其他零散问题"},
        "食物温度异常": {"口味不符预期", "食品质量问题", "食物不新鲜", "食物温度异常", "其他零散问题"},
        "送餐速度慢": {"送餐速度慢", "等待时间过长", "其他零散问题"},
        "等待时间过长": {"送餐速度慢", "等待时间过长", "其他零散问题"},
        "配送错误或漏送": {"配送错误或漏送", "其他零散问题"},
        "服务态度差": {"服务态度差", "其他零散问题"},
        "分量不足": {"分量不足", "其他零散问题"},
        "包装问题": {"包装问题", "其他零散问题"},
        "价格性价比低": {"价格性价比低", "其他零散问题"},
        "其他零散问题": set(base_aspects.values()) | {"其他零散问题"},
    }

    mapping = {}
    new_id = -1
    for i, curr_c in enumerate(current_clusters):
        curr_aspect = curr_aspects.get(curr_c, "其他零散问题")
        allowed = compatible_aspects.get(curr_aspect, {curr_aspect, "其他零散问题"})
        candidate_idxs = [j for j, b in enumerate(baseline_clusters) if base_aspects.get(b, "其他零散问题") in allowed]

        if candidate_idxs:
            local_sims = sim_matrix[i][candidate_idxs]
            best_local_pos = int(np.argmax(local_sims))
            best_sim = float(local_sims[best_local_pos])
            best_base = baseline_clusters[candidate_idxs[best_local_pos]]
        else:
            best_sim = -1.0
            best_base = None

        if best_base is not None and best_sim >= similarity_threshold:
            mapping[curr_c] = best_base
        else:
            mapping[curr_c] = new_id
            new_id -= 1

    current_df = current_df.copy()
    current_df["cluster"] = current_df["cluster"].map(mapping)
    return current_df, mapping


def build_baseline_summary(baseline_df, baseline_clusters):
    rows = []
    for c in baseline_clusters:
        keywords = get_cluster_keywords(baseline_df, c)
        rep_texts = get_representative_texts(baseline_df, c, topn=3)
        aspect = assign_aspect_name_ai(tuple(keywords), rep_texts)
        rep_texts = get_representative_texts(baseline_df, c, topn=3, preferred_aspect=aspect)
        aspect = assign_aspect_name_ai(tuple(keywords), rep_texts)
        neg_ratio = calc_cluster_neg_ratio(baseline_df, c)
        rows.append({
            "cluster": c,
            "aspect": aspect,
            "keywords": ", ".join(keywords),
            "baseline_neg": round(neg_ratio, 1),
            "样本量": calc_cluster_sample_count(baseline_df, c),
        })
    result = pd.DataFrame(rows)
    return deduplicate_aspect_names(result)


# ========================
# 6. 单期分析
# ========================
def analyze_period(df, baseline_df, baseline_clusters):
    step_box = st.container(border=True)
    with step_box:
        st.markdown("#### 🤖 AI 分析进度")
        ph1 = st.empty()
        ph2 = st.empty()
        ph3 = st.empty()
        ph4 = st.empty()
        ph1.info("1/4 正在编码当前期评论")
        current_embeddings = get_embeddings(df["评价内容"].fillna("").astype(str).tolist())
        ph1.success("1/4 评论编码完成")

        ph2.info("2/4 正在按问题方面分组后聚类")
        df = df.copy()
        df = cluster_within_aspects(df, current_embeddings)
        ph2.success("2/4 分方面聚类完成")

        cluster_count = df["cluster"].nunique()
        ph3.info("3/4 正在将当前期与 baseline 进行语义匹配")
        df, _ = match_to_baseline(baseline_df, df, similarity_threshold=0.76)
        ph3.success("3/4 baseline 匹配完成")

        ph4.info("4/4 正在生成风控结论与建议")
        new_clusters = [cid for cid in df["cluster"].unique() if cid < 0]
        all_clusters = set(baseline_clusters).union(set(df["cluster"].unique()))
        rows = []
        for c in sorted(all_clusters):
            is_new = c < 0
            baseline_stats = calc_cluster_stats(baseline_df, c) if not is_new else {"total": 0, "neg": 0, "neg_rate": 0.0, "smoothed_neg_rate": 0.0}
            current_stats = calc_cluster_stats(df, c)
            baseline_neg = baseline_stats["neg_rate"]
            baseline_smoothed_neg = baseline_stats["smoothed_neg_rate"]
            current_neg = current_stats["neg_rate"]
            current_smoothed_neg = current_stats["smoothed_neg_rate"]
            change = current_smoothed_neg - baseline_smoothed_neg
            current_count = calc_cluster_sample_count(df, c)
            baseline_count = calc_cluster_sample_count(baseline_df, c) if not is_new else 0
            if is_new:
                combined_kws = get_cluster_keywords(df, c)
            else:
                combined_kws = list(dict.fromkeys(get_cluster_keywords(baseline_df, c) + get_cluster_keywords(df, c)))
            aspect = "其他零散问题" if not combined_kws else assign_aspect_name_ai(tuple(combined_kws))
            if is_new and aspect != "其他零散问题":
                aspect = "🆕 " + aspect
            rows.append({
                "cluster": c,
                "aspect": aspect,
                "keywords": ", ".join(combined_kws),
                "baseline_neg": round(baseline_neg, 1),
                "baseline_smoothed_neg": round(baseline_smoothed_neg, 1),
                "current_neg": round(current_neg, 1),
                "current_smoothed_neg": round(current_smoothed_neg, 1),
                "change_vs_baseline": round(change, 1),
                "baseline_count": baseline_count,
                "current_count": current_count,
                "sample_count": current_count,
                "is_new": is_new,
            })
        comparison = pd.DataFrame(rows)
        comparison = deduplicate_aspect_names(comparison)
        ph4.success("4/4 风控结论生成完成")

    return {
        "clustered_df": df,
        "comparison": comparison,
        "new_clusters": new_clusters,
        "cluster_count": cluster_count,
    }


def enrich_comparison_with_prev(comparison, prev_comparison):
    comparison = comparison.copy()
    if prev_comparison is not None and not prev_comparison.empty:
        prev_cols = [c for c in ["cluster", "current_neg", "current_smoothed_neg"] if c in prev_comparison.columns]
        prev_df = prev_comparison[prev_cols].copy()
        rename_map = {}
        if "current_neg" in prev_df.columns:
            rename_map["current_neg"] = "prev_neg"
        if "current_smoothed_neg" in prev_df.columns:
            rename_map["current_smoothed_neg"] = "prev_smoothed_neg"
        prev_df = prev_df.rename(columns=rename_map)
        comparison = comparison.merge(prev_df, on="cluster", how="left")
        comparison["prev_neg"] = comparison.get("prev_neg", 0.0)
        comparison["prev_smoothed_neg"] = comparison.get("prev_smoothed_neg", comparison.get("prev_neg", 0.0))
        comparison["prev_neg"] = comparison["prev_neg"].fillna(0.0)
        comparison["prev_smoothed_neg"] = comparison["prev_smoothed_neg"].fillna(comparison["prev_neg"]).fillna(0.0)
        comparison["change_vs_prev"] = (comparison["current_smoothed_neg"] - comparison["prev_smoothed_neg"]).round(1)
    else:
        comparison["prev_neg"] = np.nan
        comparison["prev_smoothed_neg"] = np.nan
        comparison["change_vs_prev"] = np.nan

    comparison["priority"] = comparison.apply(
        lambda x: classify_priority(x["current_smoothed_neg"], x["sample_count"], x["change_vs_baseline"], x["change_vs_prev"], x["cluster"] < 0), axis=1
    )
    comparison["action_advice"] = comparison["aspect"].str.replace("🆕 ", "", regex=False).apply(action_advice)
    comparison["risk_score"] = (
        comparison["current_neg"].fillna(0) * 0.55
        + comparison["change_vs_baseline"].clip(lower=0).fillna(0) * 0.30
        + comparison["change_vs_prev"].clip(lower=0).fillna(0) * 0.15
    ).round(1)
    return comparison


def build_ai_summary(comparison, period_no, new_clusters):
    top_absolute = comparison.sort_values(["current_smoothed_neg", "sample_count"], ascending=[False, False]).iloc[0]
    top_growth_baseline = comparison.sort_values("change_vs_baseline", ascending=False).iloc[0]
    valid_prev = comparison[comparison["change_vs_prev"].notna()]
    top_growth_prev = valid_prev.sort_values("change_vs_prev", ascending=False).iloc[0] if not valid_prev.empty else None
    focus_df = comparison.sort_values(["priority", "risk_score"], ascending=[True, False]).copy()
    high_priority = focus_df[focus_df["priority"] == "高优先级"].head(3)
    focus_lines = []
    for _, row in high_priority.iterrows():
        focus_lines.append(f"- {row['aspect']}：平滑后中差评率 {row['current_smoothed_neg']:.1f}%，样本量 {int(row['sample_count'])}，建议 {row['action_advice']}")
    if not focus_lines:
        focus_lines.append("- 当前暂无高优先级问题，建议继续观察中优先级问题趋势。")

    lines = [
        f"系统判断：{TARGET_YEAR} 年第 {period_no} 期最值得优先处理的是“{top_absolute['aspect']}”，平滑后中差评率为 {top_absolute['current_smoothed_neg']:.1f}%，样本量为 {int(top_absolute['sample_count'])}。",
        f"相对 Baseline 恶化最快的是“{top_growth_baseline['aspect']}”，变化幅度为 {top_growth_baseline['change_vs_baseline']:+.1f}%。",
    ]
    if top_growth_prev is not None:
        lines.append(f"相对上一期恶化最快的是“{top_growth_prev['aspect']}”，变化幅度为 {top_growth_prev['change_vs_prev']:+.1f}%。")
    lines.append(f"本期识别到 {len(new_clusters)} 个新问题类别。")
    lines.append("建议本周优先动作：")
    lines.extend(focus_lines)
    return "\n".join(lines)


# ========================
# 7. baseline 固定处理
# ========================
@st.cache_data(show_spinner=False)
def load_and_prepare_baseline():
    baseline_df = pd.read_excel("baseline.xlsx")
    if "评价内容" not in baseline_df.columns:
        if len(baseline_df.columns) == 1:
            baseline_df.columns = ["评价内容"]
        else:
            raise ValueError("baseline.xlsx 必须包含 评价内容 列")
    baseline_df["评价内容"] = baseline_df["评价内容"].fillna("").astype(str).str.strip()
    baseline_df = baseline_df[baseline_df["评价内容"] != ""].reset_index(drop=True)
    baseline_embeddings = get_embeddings(baseline_df["评价内容"].fillna("").astype(str).tolist())
    baseline_df["cluster"] = cluster_reviews(baseline_embeddings)
    baseline_df = attach_noise_to_major_clusters(baseline_df, baseline_embeddings, min_major_size=5, attach_threshold=0.72)
    baseline_df = merge_tiny_clusters_to_major(baseline_df, baseline_embeddings, tiny_size=3, attach_threshold=0.74)
    baseline_df = reduce_duplicate_clusters(baseline_df, baseline_embeddings, center_threshold=0.84, keyword_overlap_threshold=0.68)
    baseline_df = collect_remaining_sparse_to_other(baseline_df, min_size=3)
    baseline_clusters = sorted(baseline_df["cluster"].unique())
    baseline_summary = build_baseline_summary(baseline_df, baseline_clusters)
    return baseline_df, baseline_clusters, baseline_summary

if st.session_state.app_stage == "intro":
    show_intro_dialog()
    st.stop()

if st.session_state.app_stage == "booting":
    st.info("系统正在初始化 baseline 基线结构，请稍候……")
    with st.spinner("首次加载中，大约需要几秒钟"):
        try:
            baseline_df, baseline_clusters, baseline_summary = load_and_prepare_baseline()
            st.session_state.baseline_ready = True
            st.session_state.app_stage = "ready"
            st.rerun()
        except Exception as e:
            st.error(f"baseline.xlsx 读取失败：{e}")
            st.stop()

try:
    with st.spinner("正在加载 baseline 基线结构..."):
        baseline_df, baseline_clusters, baseline_summary = load_and_prepare_baseline()
        st.session_state.baseline_ready = True
        if st.session_state.app_stage != "ready":
            st.session_state.app_stage = "ready"
except Exception as e:
    st.error(f"baseline.xlsx 读取失败：{e}")
    st.stop()

# ========================
# 8. 侧边栏工作台
# ========================
with st.sidebar:
    st.markdown("## 🧭 工作台")
    st.metric("当前期数", f"第 {st.session_state.current_period} 期")
    st.metric("Baseline 问题类数", len(baseline_clusters))
    st.metric("已完成分析期数", len(st.session_state.period_history))
    if st.session_state.period_history:
        st.markdown("### 历史期数")
        for p in sorted(st.session_state.period_history.keys()):
            cmp_df = st.session_state.period_history[p]["comparison"]
            top_name = cmp_df.sort_values(["current_smoothed_neg", "sample_count"], ascending=[False, False]).iloc[0]["aspect"] if not cmp_df.empty else "-"
            st.caption(f"第 {p} 期：最高风险 {top_name}")
    st.markdown("---")
    show_only_worse = st.checkbox("只看恶化问题", value=False)
    show_only_new = st.checkbox("只看新问题", value=False)
    category_filter = st.multiselect(
        "按问题类别筛选",
        options=["服务", "食品", "环境", "价格", "配送", "包装", "口味", "秩序", "其他"],
        default=[]
    )

# ========================
# 9. 上传区 + baseline 展示
# ========================
if st.session_state.latest_completed_period >= st.session_state.current_period:
    st.session_state.current_period = st.session_state.latest_completed_period + 1
if st.session_state.pending_period != st.session_state.current_period:
    st.session_state.pending_period = st.session_state.current_period
period_no = st.session_state.pending_period
st.markdown("### 📌 Baseline 中差评结构")
st.dataframe(baseline_summary, use_container_width=True)
st.markdown("---")

st.markdown("### 📤 上传本期评论数据")
st.info(f"请上传 {TARGET_YEAR} 年第 {period_no} 期评论数据（需包含 评价内容 + 评价类型）。系统将在点击分析按钮后再执行数据体检与 AI 风控，减少重复刷新。")

st.markdown("#### 🎯 快速体验 Demo")
st.caption("每一期都支持：下载样例、一键载入、或手动上传自己的同结构数据。")

DEMO_FILES = {
    1: "sample_data/2026_period_1_demo.xlsx",
    2: "sample_data/2026_period_2_demo.xlsx",
    3: "sample_data/2026_period_3_demo.xlsx",
}

def read_demo_bytes(path):
    with open(path, "rb") as f:
        return f.read()

demo_cols = st.columns(3)
for idx, demo_period in enumerate([1, 2, 3]):
    demo_path = DEMO_FILES[demo_period]
    with demo_cols[idx]:
        st.markdown(f"##### 第 {demo_period} 期")
        if Path(demo_path).exists():
            st.download_button(
                label=f"下载第{demo_period}期样例",
                data=read_demo_bytes(demo_path),
                file_name=f"2026_period_{demo_period}_demo.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_demo_{demo_period}",
                use_container_width=True,
            )
            if st.button(f"一键载入第{demo_period}期", key=f"load_demo_{demo_period}", use_container_width=True):
                demo_df = pd.read_excel(demo_path)
                st.session_state.demo_uploaded_df = demo_df.copy()
                st.session_state.demo_run_analysis = True
                st.session_state.demo_period_no = demo_period
                st.session_state.pending_period = demo_period
                st.rerun()
        else:
            st.warning(f"未找到第 {demo_period} 期样例文件")

        uploaded_demo_file = st.file_uploader(
            f"手动上传第{demo_period}期数据",
            type=["csv", "xlsx"],
            key=f"manual_upload_period_{demo_period}",
        )
        if uploaded_demo_file is not None:
            st.session_state.manual_uploaded_file = uploaded_demo_file
            st.session_state.manual_period_no = demo_period
            st.session_state.pending_period = demo_period
            st.caption(f"已选择手动上传：第 {demo_period} 期")
with st.form(key=f"upload_form_{period_no}"):
    uploaded_file = st.file_uploader(
        label=f"上传 {TARGET_YEAR} 年第 {period_no} 期评论数据集",
        type=["csv", "xlsx"],
        key=f"uploader_period_{period_no}",
    )
    run_analysis = st.form_submit_button("开始分析本期数据", use_container_width=True)

# ========================
# 10. 当前期分析
# ========================
demo_df = st.session_state.get("demo_uploaded_df")
demo_run_analysis = st.session_state.get("demo_run_analysis", False)
demo_period_no = st.session_state.get("demo_period_no", None)
manual_uploaded_file = st.session_state.get("manual_uploaded_file")
manual_period_no = st.session_state.get("manual_period_no", None)

active_manual_file = None
if manual_uploaded_file is not None and manual_period_no == period_no:
    active_manual_file = manual_uploaded_file
elif uploaded_file is not None and run_analysis:
    active_manual_file = uploaded_file

if (active_manual_file is not None and (run_analysis or manual_period_no == period_no)) or (demo_df is not None and demo_run_analysis and demo_period_no == period_no):
    if active_manual_file is not None:
        upload_key = f"{period_no}::{active_manual_file.name}::{getattr(active_manual_file, 'size', 0)}"
    else:
        upload_key = f"demo::{period_no}::2026_period_{period_no}_demo.xlsx"

    if st.session_state.last_processed_upload_key == upload_key and st.session_state.current_analysis_result is not None:
        df = st.session_state.current_analysis_result["raw_df"].copy()
        quality = st.session_state.current_analysis_result["quality"]
        result = st.session_state.current_analysis_result["result"]
        comparison = st.session_state.current_analysis_result["comparison"].copy()
        new_clusters = st.session_state.current_analysis_result["new_clusters"]
    else:
        try:
            if active_manual_file is not None:
                df = safe_read_table(active_manual_file)
            else:
                df = st.session_state.demo_uploaded_df.copy()
                rename_map = {}
                for c in df.columns:
                    c_str = str(c).strip()
                    if c_str.lower() in ["评价内容", "content", "comment", "评论内容"]:
                        rename_map[c] = "评价内容"
                    if c_str in ["评价类型", "review_type", "sentiment"]:
                        rename_map[c] = "评价类型"
                if rename_map:
                    df = df.rename(columns=rename_map)
                if "评价内容" not in df.columns:
                    if len(df.columns) == 1:
                        df.columns = ["评价内容"]
                    else:
                        raise ValueError("样例数据缺少 评价内容 列")
                df["评价内容"] = df["评价内容"].fillna("").astype(str).str.strip()
                df = df[df["评价内容"] != ""].reset_index(drop=True)
        except Exception as e:
            st.error(f"文件读取失败：{e}")
            st.stop()

        quality = build_data_quality_report(df)
        if len(df) < MIN_REQUIRED_ROWS:
            st.warning(f"评论条数过少，建议至少上传 {MIN_REQUIRED_ROWS} 条以上。")
            st.stop()

        st.success("样例数据已载入，系统开始分析。" if upload_key.startswith("demo::") else f"第 {period_no} 期手动上传成功，系统开始分析。")
        result = analyze_period(df, baseline_df, baseline_clusters)
        prev_period = period_no - 1
        prev_cmp = st.session_state.period_history[prev_period]["comparison"] if prev_period in st.session_state.period_history else None
        comparison = enrich_comparison_with_prev(result["comparison"], prev_cmp)
        new_clusters = result["new_clusters"]
    st.markdown("### 🩺 数据体检")
    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("评论条数", quality["total"])
    q2.metric("重复评论", quality["duplicate_reviews"])
    q3.metric("短文本数", quality["short_reviews"])
    q4.metric("中差评条数", quality["neutral_neg"] if quality["has_type"] else 0)
    q5.metric("评价类型字段", "已检测" if quality["has_type"] else "缺失")
    if quality["quality_flags"]:
        st.warning("数据体检提示：" + "、".join(quality["quality_flags"]))
    else:
        st.success("数据体检通过：未发现明显结构性问题。")

    st.session_state.period_history[period_no] = {
        "raw_df": df.copy(),
        "clustered_df": result["clustered_df"].copy(),
        "comparison": comparison.copy(),
    }
    st.session_state.latest_completed_period = max(st.session_state.latest_completed_period, period_no)
    st.session_state.last_processed_upload_key = upload_key
    st.session_state.current_analysis_result = {
        "raw_df": df.copy(),
        "quality": quality,
        "result": result,
        "comparison": comparison.copy(),
        "new_clusters": new_clusters,
    }

    # 过滤视图
    filtered_df = comparison.copy()
    if show_only_worse:
        filtered_df = filtered_df[(filtered_df["change_vs_baseline"] > 0) | (filtered_df["change_vs_prev"].fillna(0) > 0)]
    if show_only_new:
        filtered_df = filtered_df[filtered_df["cluster"] < 0]
    if category_filter:
        def hit_category(x):
            x = str(x)
            mapping = {
                "服务": ["服务", "态度"],
                "食品": ["食品", "出餐"],
                "环境": ["环境", "卫生", "噪音", "设施"],
                "价格": ["价格", "性价比"],
                "配送": ["配送", "等待"],
                "包装": ["包装"],
                "口味": ["口味"],
                "秩序": ["秩序"],
                "其他": ["其他"]
            }
            for cat in category_filter:
                if any(k in x for k in mapping.get(cat, [])):
                    return True
            return False
        filtered_df = filtered_df[filtered_df["aspect"].apply(hit_category)]
    if filtered_df.empty:
        filtered_df = comparison.copy()

    top_absolute = comparison.sort_values(["current_smoothed_neg", "sample_count"], ascending=[False, False]).iloc[0]
    top_growth_baseline = comparison.sort_values("change_vs_baseline", ascending=False).iloc[0]
    top_growth_prev = comparison[comparison["change_vs_prev"].notna()].sort_values("change_vs_prev", ascending=False).iloc[0] if comparison["change_vs_prev"].notna().any() else None
    ai_summary = build_ai_summary(comparison, period_no, new_clusters)
    st.session_state.last_ai_summary = ai_summary

    st.markdown(f"### 📊 {TARGET_YEAR} 年第 {period_no} 期 KPI 风控总览")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("当前最严重问题", top_absolute["aspect"], f"平滑后 {top_absolute['current_smoothed_neg']:.1f}% / 样本{int(top_absolute['sample_count'])}")
    top_priority = comparison.sort_values(["risk_score", "sample_count"], ascending=[False, False]).iloc[0]
    col2.metric("当前最优先处理", top_priority["aspect"], f"风险分 {top_priority['risk_score']:.1f}")
    col3.metric("较 Baseline 恶化最快", top_growth_baseline["aspect"], f"{top_growth_baseline['change_vs_baseline']:+.1f}%")
    col4.metric("较上期恶化最快", top_growth_prev["aspect"] if top_growth_prev is not None else "暂无上期", f"{top_growth_prev['change_vs_prev']:+.1f}%" if top_growth_prev is not None else "-")

    # 标签页
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["总览", "图表", "风险明细", "AI建议", "历史追踪"])

    with tab1:
        st.markdown("#### 🤖 AI风控摘要")
        st.info(ai_summary)
        st.markdown("#### 优先处理清单")
        overview_cols = ["aspect", "priority", "sample_count", "current_neg", "current_smoothed_neg", "change_vs_baseline", "change_vs_prev", "risk_score"]
        st.dataframe(filtered_df[overview_cols].sort_values(["risk_score", "current_neg"], ascending=False), use_container_width=True)

    with tab2:
        st.markdown("#### 风险对比图表")
        chart_df = filtered_df.copy()
        chart_df["label"] = chart_df["aspect"] + " (C" + chart_df["cluster"].astype(str) + ")"
        chart_df = chart_df.sort_values(["current_smoothed_neg", "sample_count"], ascending=[False, False])
        left, right = st.columns(2)
        with left:
            st.markdown(f"##### 第 {period_no} 期 vs Baseline")
            baseline_compare_df = chart_df.set_index("label")[["baseline_neg", "current_neg"]].rename(columns={"baseline_neg": "Baseline", "current_neg": f"第{period_no}期"})
            st.bar_chart(baseline_compare_df, use_container_width=True)
        with right:
            st.markdown(f"##### 第 {period_no} 期 vs 第 {period_no - 1} 期")
            if chart_df["prev_neg"].notna().any():
                prev_compare_df = chart_df.set_index("label")[["prev_neg", "current_neg"]].rename(columns={"prev_neg": f"第{period_no - 1}期", "current_neg": f"第{period_no}期"})
                st.bar_chart(prev_compare_df, use_container_width=True)
            else:
                st.info("当前为第 1 期，暂无上期对比。")
        st.markdown("#### 风险矩阵（表格版）")
        risk_matrix = chart_df[["aspect", "sample_count", "current_neg", "current_smoothed_neg", "change_vs_baseline", "change_vs_prev", "priority"]].copy()
        risk_matrix = risk_matrix.rename(columns={
            "aspect": "问题类别",
            "sample_count": "样本量",
"current_neg": "原始中差评率",
"current_smoothed_neg": "平滑后风险",
            "change_vs_baseline": "相对Baseline变化",
            "change_vs_prev": "相对上期变化",
            "priority": "优先级"
        })
        st.dataframe(risk_matrix, use_container_width=True)

    with tab3:
        st.markdown("#### 风险明细")
        detail_cols = [
            "cluster", "aspect", "keywords", "baseline_count", "current_count",
            "baseline_neg", "baseline_smoothed_neg", "prev_neg", "prev_smoothed_neg", "current_neg", "current_smoothed_neg", "sample_count", "change_vs_baseline", "change_vs_prev", "priority", "risk_score"
        ]
        st.dataframe(filtered_df[detail_cols].sort_values("risk_score", ascending=False, na_position="last"), use_container_width=True)
        cluster_options = filtered_df["cluster"].tolist()
        cluster_map = {f"C{row['cluster']} - {row['aspect']}": row['cluster'] for _, row in filtered_df.iterrows()}
        selected_focus = st.multiselect("手动标记重点问题", options=list(cluster_map.keys()))
        st.session_state.manual_focus_clusters = [cluster_map[x] for x in selected_focus]

    with tab4:
        st.markdown("#### AI处置建议")
        advice_df = filtered_df[["aspect", "priority", "action_advice", "keywords"]].copy().sort_values(["priority", "aspect"])
        for _, row in advice_df.iterrows():
            with st.expander(f"{row['aspect']}｜{row['priority']}"):
                st.write(f"关键词：{row['keywords']}")
                st.write(f"建议动作：{row['action_advice']}")
        export_bytes = build_export_bytes(ai_summary, filtered_df, period_no)
        st.download_button(
            "⬇ 导出管理摘要（CSV文本版）",
            data=export_bytes,
            file_name=f"reviewpulse_summary_{TARGET_YEAR}_period_{period_no}.csv",
            mime="text/csv"
        )

    with tab5:
        st.markdown("#### 历史追踪")
        if st.session_state.period_history:
            history_rows = []
            for p, payload in sorted(st.session_state.period_history.items()):
                cmp_df = payload["comparison"]
                if cmp_df.empty:
                    continue
                top_row = cmp_df.sort_values(["current_smoothed_neg", "sample_count"], ascending=[False, False]).iloc[0]
                history_rows.append({
                    "期数": f"第{p}期",
                    "最高风险问题": top_row["aspect"],
                    "最高原始中差评率": round(float(top_row["current_neg"]), 1),
"最高平滑后风险": round(float(top_row["current_smoothed_neg"]), 1),
                    "问题类数": int(cmp_df['cluster'].nunique()),
                })
            if history_rows:
                history_df = pd.DataFrame(history_rows)
                st.dataframe(history_df, use_container_width=True)
        else:
            st.info("暂无历史期数记录。")

    st.markdown("### ⚠ 风险提示")
    st.error(f"当前最严重问题：{top_absolute['aspect']}（平滑后 {top_absolute['current_smoothed_neg']:.1f}%，样本量 {int(top_absolute['sample_count'])}）")
    st.warning(f"相对 Baseline 恶化最快问题：{top_growth_baseline['aspect']}（{top_growth_baseline['change_vs_baseline']:+.1f}%）")
    if top_growth_prev is not None:
        st.warning(f"相对上一期恶化最快问题：{top_growth_prev['aspect']}（{top_growth_prev['change_vs_prev']:+.1f}%）")
    if new_clusters:
        new_aspects = comparison[comparison["cluster"] < 0]["aspect"].tolist()
        st.info("新发现问题：" + "、".join(new_aspects))

    st.markdown("---")
    if st.button(f"➡ 开始上传 {TARGET_YEAR} 年第 {period_no + 1} 期评论数据", key=f"next_period_{period_no}"):
        next_period = max(st.session_state.current_period, st.session_state.latest_completed_period + 1)
        st.session_state.current_period = next_period
        st.session_state.pending_period = next_period
        st.session_state.current_analysis_result = None
        st.session_state.last_processed_upload_key = None
        st.rerun()
