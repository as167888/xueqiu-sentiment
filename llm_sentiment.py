#!/usr/bin/env python3
"""
雪球帖子 LLM 情绪分析
======================
使用 DeepSeek API 对每条帖子进行语义级情绪评分。
Python 代码只负责编排（分批、调 API、汇总），情绪判断完全由大模型完成。

模型会从以下维度判断每条帖子的情绪：
- 对公司/业务的态度是乐观还是悲观
- 对股价/估值的态度是看涨还是看跌
- 语气强度（淡定 → 强烈）
- 是否有数据/逻辑支撑（理性 vs 情绪化）
- 是否在传播乐观/悲观预期

输出: 每帖 0-100 情绪分 + 日度聚合时间序列
"""

import json
import os
import sys
import time
import math
import sqlite3
from datetime import datetime
from collections import defaultdict

from openai import OpenAI

# ============================================================
# 配置
# ============================================================
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

# 数据库配置
DB_PATH = os.environ.get("XUEQIU_DB_PATH", r"E:\project\xueqiu\xueqiu_stock.db")
CACHE_DB_PATH = os.environ.get("SENTIMENT_CACHE_DB", r"E:\project\kaye1\sentiment_cache.db")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ============================================================
# 从 SQLite 数据库加载个股贴文
# ============================================================
def load_posts_from_db(db_path, stock_code='02400'):
    """从 xueqiu_stock.db 的 stock_posts 表读取个股贴文。
    不读取 user_posts（逸修的博主贴文）和 column_articles（专栏文章）。
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "SELECT id, created_at, text, like_count, reply_count, retweet_count, "
        "user_id, user_name, source, target, retweeted_id, type "
        "FROM stock_posts WHERE stock = ? ORDER BY created_at",
        (stock_code,)
    )
    posts = [dict(row) for row in cur.fetchall()]
    conn.close()
    return posts


# ============================================================
# 情绪分析结果缓存（避免重复调用 LLM API）
# ============================================================
def init_cache_db(db_path):
    """初始化缓存数据库，创建 sentiment_cache 表"""
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS sentiment_cache ("
        "post_id INTEGER PRIMARY KEY,"
        "score INTEGER NOT NULL,"
        "direction TEXT NOT NULL,"
        "confidence REAL NOT NULL,"
        "rationale TEXT,"
        "analyzed_at TEXT NOT NULL"
        ")"
    )
    conn.commit()
    conn.close()


def get_cached_results(db_path):
    """从缓存中读取所有已分析帖子的结果，返回 {post_id: result_dict}"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "SELECT post_id, score, direction, confidence, rationale FROM sentiment_cache"
    )
    cached = {}
    for row in cur.fetchall():
        d = dict(row)
        cached[d.pop('post_id')] = d
    conn.close()
    return cached


def save_batch_to_cache(db_path, results):
    """将一批分析结果保存到缓存数据库"""
    conn = sqlite3.connect(db_path)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for r in results:
        conn.execute(
            "INSERT OR REPLACE INTO sentiment_cache (post_id, score, direction, confidence, rationale, analyzed_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (r['id'], r['score'], r['direction'], r['confidence'], r.get('rationale', ''), now)
        )
    conn.commit()
    conn.close()

# ============================================================
# 系统提示词
# ============================================================
SYSTEM_PROMPT = """你是一位专注于中国互联网/游戏行业的买方分析师，擅长从投资社区讨论中提取市场情绪信号。

## 你的任务
对给定的雪球论坛帖子逐一进行情绪评分。每条帖子你需要判断：

1. **情绪方向**：发帖人对 心动公司(02400.HK) 的态度是乐观(看多)还是悲观(看空)
2. **情绪强度**：从极度悲观到极度乐观的程度
3. **信号质量**：该帖子的论据是否有数据/逻辑支撑，还是纯情绪宣泄

## 评分标准 (0-100)
```
0-10:   极度悲观 — 认为公司没救了、会退市、管理层是骗子
10-25:  非常悲观 — 认为会继续大跌、基本面严重恶化
25-40:  偏悲观   — 担心短期承压、对某些方面不满、失望
40-50:  轻微偏空 — 有疑虑但不算悲观，提出质疑或担忧
50-55:  中性/信息 — 纯信息分享、提问、客观陈述
55-65:  轻微偏多 — 认为有价值/机会，但保持谨慎
65-80:  偏乐观   — 看好业务前景、认为当前低估、积极持有
80-90:  非常乐观 — 强烈看好、认为会大涨、公司非常优秀
90-100: 极度乐观 — 认为公司是最佳投资标的、必然成功
```

## 判断维度
- **业务判断**：对公司产品(TapTap/游戏)、管理层、战略的看法
- **估值判断**：对当前股价、未来走势、合理估值的看法
- **语气特征**：坚定/犹豫、理性/情绪化、建设性批评/纯粹抱怨
- **预期引导**：是否在传播某种乐观/悲观预期

## 输出格式
对每条帖子，返回以下 JSON 字段：
- `id`: 帖子 ID (int)
- `score`: 情绪分数 (0-100 的整数)
- `direction`: "bullish" / "bearish" / "neutral"
- `confidence`: 你对这个判断的置信度 (0.0-1.0)
- `rationale`: 一句话说明判断依据 (中文, 15字以内)

请按顺序返回一个 JSON 数组。只输出 JSON，不要输出其他内容。"""


# ============================================================
# 文本清洗
# ============================================================
def clean_post_text(text):
    """清洗帖子文本，去除 HTML 标签但保留语义"""
    import re
    # 移除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 移除图片链接
    text = re.sub(r'https?://\S+', '', text)
    # 清理多余空白
    text = re.sub(r'\s+', ' ', text)
    return text.strip()[:800]  # 截断过长文本


# ============================================================
# 批量分析
# ============================================================
def analyze_batch(posts_batch, batch_num, total_batches):
    """调用 LLM 分析一批帖子"""
    # 构建用户消息：每条帖子格式化
    post_lines = []
    for i, post in enumerate(posts_batch):
        text = clean_post_text(post.get('text', ''))
        # 提取关键互动数据
        likes = post.get('like_count', 0)
        replies = post.get('reply_count', 0)
        retweets = post.get('retweet_count', 0)
        engagement = f"赞{likes}/回{replies}/转{retweets}" if (likes + replies) > 0 else "无互动"

        post_lines.append(
            f"--- Post #{i+1} ---\n"
            f"id: {post['id']}\n"
            f"用户: {post.get('user_name', '?')}\n"
            f"时间: {post.get('created_at', '?')}\n"
            f"互动: {engagement}\n"
            f"正文: {text}\n"
        )

    user_message = "\n".join(post_lines)
    user_message += f"\n---\n请分析以上 {len(posts_batch)} 条帖子，返回 JSON 数组。"

    print(f"  Batch {batch_num}/{total_batches}: 发送 {len(posts_batch)} 条帖子到 {MODEL} ... ", end='', flush=True)

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                max_tokens=4096,
                temperature=0.1,  # 低温度保证一致性
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )

            raw = resp.choices[0].message.content.strip()

            # 提取 JSON（处理可能的 markdown 包裹）
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                if raw.endswith("```"):
                    raw = raw[:-3]
            raw = raw.strip()

            results = json.loads(raw)
            if isinstance(results, dict) and 'posts' in results:
                results = results['posts']
            if isinstance(results, dict):
                results = [results]

            print(f"OK ({len(results)} 条)")
            return results

        except Exception as e:
            print(f"失败 (attempt {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(2 * (attempt + 1))

    print("批量分析彻底失败，返回默认分数")
    return [{"id": p['id'], "score": 50, "direction": "neutral", "confidence": 0.0, "rationale": "API调用失败"} for p in posts_batch]


# ============================================================
# 主程序
# ============================================================
def main():
    stock_code = sys.argv[1] if len(sys.argv) > 1 else '02400'
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    # 从数据库加载个股贴文（仅 stock_posts 表，不含逸修博主贴文和专栏文章）
    posts = load_posts_from_db(DB_PATH, stock_code)

    # 初始化缓存并读取已有结果
    init_cache_db(CACHE_DB_PATH)
    cached = get_cached_results(CACHE_DB_PATH)

    # 分离新帖（未缓存）与已缓存帖
    new_posts = [p for p in posts if p['id'] not in cached]
    cached_results = [{'id': pid, **info} for pid, info in cached.items()]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"=" * 65)
    print(f"  雪球帖子 LLM 情绪分析 (模型: {MODEL})")
    print(f"  数据库: {DB_PATH} | 股票: {stock_code}")
    print(f"  帖子: {len(posts)} 条 | 已缓存: {len(cached)} 条 | 待分析: {len(new_posts)} 条")
    print(f"  每批: {batch_size} 条 | 预计 {math.ceil(len(new_posts)/batch_size) if new_posts else 0} 批")
    print(f"=" * 65)

    start_time = time.time()
    new_results = []

    if new_posts:
        batches = [new_posts[i:i+batch_size] for i in range(0, len(new_posts), batch_size)]
        total_batches = len(batches)

        for i, batch in enumerate(batches, 1):
            results = analyze_batch(batch, i, total_batches)
            new_results.extend(results)
            if i < total_batches:
                time.sleep(0.5)

        # 将新结果写入缓存
        save_batch_to_cache(CACHE_DB_PATH, new_results)
        print(f"  已保存 {len(new_results)} 条新结果到缓存")

    # 合并缓存与新结果
    all_results = cached_results + new_results
    # 确保按帖子时间排序（用于后续聚合）
    id_order = {p['id']: i for i, p in enumerate(posts)}
    all_results.sort(key=lambda r: id_order.get(r['id'], 0))

    elapsed = time.time() - start_time
    analyzed_count = len(new_posts)
    if analyzed_count > 0:
        print(f"\n  分析完成! 耗时 {elapsed:.1f}s, 平均 {elapsed/analyzed_count:.1f}s/帖")
    else:
        print(f"\n  全部 {len(posts)} 条帖子已缓存，无需调用 API")

    # ============================================================
    # 按日聚合
    # ============================================================
    # 先建 id->post 映射
    post_map = {p['id']: p for p in posts}

    daily_data = defaultdict(list)
    for r in all_results:
        pid = r['id']
        post = post_map.get(pid, {})
        date = post.get('created_at', '')[:10]
        if date:
            daily_data[date].append({
                'score': r['score'],
                'direction': r['direction'],
                'confidence': r['confidence'],
                'rationale': r.get('rationale', ''),
                'post': post,
            })

    time_series = []
    for date in sorted(daily_data.keys()):
        items = daily_data[date]
        scores = [it['score'] for it in items]

        # 互动加权均分
        weighted = 0.0
        total_w = 0.0
        for it in items:
            p = it['post']
            eng = p.get('like_count', 0) + p.get('reply_count', 0) * 1.5
            w = max(1.0, math.log(1 + eng))
            weighted += it['score'] * w
            total_w += w
        wm = weighted / total_w if total_w > 0 else sum(scores) / len(scores)

        mean_s = sum(scores) / len(scores)
        median_s = sorted(scores)[len(scores) // 2]
        std_s = (sum((s - mean_s)**2 for s in scores) / len(scores)) ** 0.5

        bullish = sum(1 for s in scores if s > 55)
        neutral = sum(1 for s in scores if 45 <= s <= 55)
        bearish = sum(1 for s in scores if s < 45)

        label = 'bullish' if wm >= 60 else ('bearish' if wm < 40 else ('slightly_bullish' if wm >= 52 else ('slightly_bearish' if wm <= 48 else 'neutral')))

        time_series.append({
            'date': date,
            'post_count': len(scores),
            'weighted_mean': round(wm, 1),
            'mean_score': round(mean_s, 1),
            'median_score': round(median_s, 1),
            'std_dev': round(std_s, 1),
            'bullish_pct': round(bullish / len(scores) * 100, 1),
            'neutral_pct': round(neutral / len(scores) * 100, 1),
            'bearish_pct': round(bearish / len(scores) * 100, 1),
            'avg_confidence': round(sum(it['confidence'] for it in items) / len(items), 2),
            'sentiment_label': label,
        })

    # ============================================================
    # 输出
    # ============================================================
    print(f"\n{'Date':<12} {'N':>3}  {'Score':>6}  {'Bull%':>6}  {'Neut%':>6}  {'Bear%':>6}  {'Label':<16}  {'Confidence'}")
    print("-" * 75)
    for r in time_series:
        bar = '+' * max(0, int((r['weighted_mean'] - 30) / 5)) + '-' * max(0, 8 - int((r['weighted_mean'] - 30) / 5))
        print(f"{r['date']:<12} {r['post_count']:>3}  {r['weighted_mean']:>6.1f}  {r['bullish_pct']:>5.1f}%  {r['neutral_pct']:>5.1f}%  {r['bearish_pct']:>5.1f}%  {r['sentiment_label']:<16} [{bar}]  {r['avg_confidence']:.2f}")
    print("-" * 75)

    scores_list = [r['weighted_mean'] for r in time_series]
    print(f"\n  Mean: {sum(scores_list)/len(scores_list):.1f} | High: {max(scores_list):.1f} | Low: {min(scores_list):.1f} | Range: {max(scores_list)-min(scores_list):.1f}")

    # 保存（以股票代码 + 时间戳命名）
    out_json = f'{stock_code}_{timestamp}_llm_sentiment.json'
    out_csv = f'{stock_code}_{timestamp}_llm_sentiment_timeseries.csv'

    output = {
        'metadata': {
            'source': f'sqlite://{DB_PATH}::stock_posts',
            'stock_code': stock_code,
            'total_posts': len(posts),
            'cached_posts': len(cached),
            'new_posts': len(new_posts),
            'model': MODEL,
            'analysis_method': 'LLM-based semantic sentiment analysis',
            'scoring': '0-100 scale: 0=extreme bearish, 50=neutral, 100=extreme bullish',
            'batch_size': batch_size,
            'processing_time_seconds': round(elapsed, 1),
        },
        'per_post_results': [
            {
                'id': r['id'],
                'score': r['score'],
                'direction': r['direction'],
                'confidence': r['confidence'],
                'rationale': r.get('rationale', ''),
                'user_name': post_map.get(r['id'], {}).get('user_name', ''),
                'created_at': post_map.get(r['id'], {}).get('created_at', ''),
                'text_preview': clean_post_text(post_map.get(r['id'], {}).get('text', ''))[:120],
            }
            for r in all_results
        ],
        'time_series': time_series,
    }

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    with open(out_csv, 'w', encoding='utf-8') as f:
        f.write("date,weighted_mean,mean_score,post_count,bullish_pct,neutral_pct,bearish_pct,sentiment_label,avg_confidence\n")
        for r in time_series:
            f.write(f"{r['date']},{r['weighted_mean']},{r['mean_score']},{r['post_count']},{r['bullish_pct']},{r['neutral_pct']},{r['bearish_pct']},{r['sentiment_label']},{r['avg_confidence']}\n")

    print(f"\n  详细结果: {out_json}")
    print(f"  时间序列: {out_csv}")

    return time_series


if __name__ == '__main__':
    main()
