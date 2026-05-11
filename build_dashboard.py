#!/usr/bin/env python3
"""Generate sentiment dashboard HTML from LLM analysis results, with price overlay."""

import json
import sys
import os
import glob
import sqlite3
import math
import requests
from datetime import datetime
from collections import defaultdict

# 数据库配置
DB_PATH = os.environ.get("XUEQIU_DB_PATH", r"E:\project\xueqiu\xueqiu_stock.db")
CACHE_DB_PATH = os.environ.get("SENTIMENT_CACHE_DB", r"E:\project\kaye1\sentiment_cache.db")

# 命令行: python build_dashboard.py [stock_code]

# ============================================================
# 从腾讯财经获取港股日K线收盘价
# ============================================================
def fetch_hk_daily_kline(code, days=640):
    """从腾讯财经获取港股日K线数据，返回 [{'date': 'YYYY-MM-DD', 'close': float}, ...]"""
    url = f"http://web.ifzq.gtimg.cn/appstock/app/kline/kline?_var=kline_day&param=hk{code},day,,,{days}"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.encoding = "gbk"

    text = resp.text
    start = text.find("{")
    end = text.rfind("}") + 1
    data = json.loads(text[start:end])

    stock_data = data["data"][f"hk{code}"]
    kline_key = "day" if "day" in stock_data else "qfqday"
    klines = stock_data[kline_key]

    prices = []
    for k in klines:
        prices.append({
            "date": k[0],
            "close": round(float(k[2]), 2),
        })
    return prices


# ============================================================
# 从本地数据库加载情绪分析结果
# ============================================================
def clean_post_text(text):
    """清洗帖子文本，去除 HTML 标签"""
    import re
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()[:800]


def load_sentiment_from_db(stock_code='02400'):
    """直接从 sentiment_cache.db + xueqiu_stock.db 加载情绪数据和帖子元数据。
    返回 (time_series, per_post_results, metadata) 或 None。
    """
    if not os.path.exists(CACHE_DB_PATH):
        return None

    cache_conn = sqlite3.connect(CACHE_DB_PATH)
    cache_conn.row_factory = sqlite3.Row
    cache_cur = cache_conn.execute("SELECT * FROM sentiment_cache")
    cached = {row['post_id']: dict(row) for row in cache_cur.fetchall()}
    cache_conn.close()

    if not cached:
        return None

    posts_conn = sqlite3.connect(DB_PATH)
    posts_conn.row_factory = sqlite3.Row
    placeholders = ','.join('?' * len(cached))
    posts_cur = posts_conn.execute(
        f"SELECT id, created_at, text, like_count, reply_count, retweet_count, "
        f"user_name, user_id FROM stock_posts "
        f"WHERE stock = ? AND id IN ({placeholders}) ORDER BY created_at",
        (stock_code, *cached.keys())
    )
    posts_rows = {row['id']: dict(row) for row in posts_cur.fetchall()}
    posts_conn.close()

    # 构建 per_post_results
    per_post_results = []
    for pid, c in cached.items():
        post = posts_rows.get(pid)
        if not post:
            continue
        per_post_results.append({
            'id': pid,
            'score': c['score'],
            'direction': c['direction'],
            'confidence': c['confidence'],
            'rationale': c.get('rationale', ''),
            'user_name': post.get('user_name', ''),
            'created_at': post.get('created_at', ''),
            'text_preview': clean_post_text(post.get('text', ''))[:120],
        })

    if not per_post_results:
        return None

    per_post_results.sort(key=lambda p: p['created_at'])

    # 按日聚合
    post_map = posts_rows
    daily_data = defaultdict(list)
    for r in per_post_results:
        date = r['created_at'][:10]
        if date:
            daily_data[date].append({
                'score': r['score'],
                'direction': r['direction'],
                'confidence': r['confidence'],
                'rationale': r.get('rationale', ''),
                'post': post_map.get(r['id'], {}),
            })

    time_series = []
    for date in sorted(daily_data.keys()):
        items = daily_data[date]
        scores = [it['score'] for it in items]

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

        label = ('bullish' if wm >= 60 else
                 'bearish' if wm < 40 else
                 'slightly_bullish' if wm >= 52 else
                 'slightly_bearish' if wm <= 48 else 'neutral')

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

    metadata = {
        'source': f'sqlite://{CACHE_DB_PATH}::sentiment_cache + sqlite://{DB_PATH}::stock_posts',
        'stock_code': stock_code,
        'total_posts': len(cached),
        'model': 'deepseek-chat (cached)',
        'analysis_method': 'LLM-based semantic sentiment analysis (从本地缓存数据库加载)',
    }

    return time_series, per_post_results, metadata



def main():
    # ============================================================
    # Load sentiment data — 优先从本地数据库直接读取
    # ============================================================
    stock_code = sys.argv[1] if len(sys.argv) > 1 else '02400'

    db_result = load_sentiment_from_db(stock_code)
    if db_result:
        ts, posts, meta = db_result
        print(f'  情绪数据来源: 本地数据库 (sentiment_cache.db + xueqiu_stock.db)')
    else:
        # 数据库无缓存时回退到 JSON 文件
        if len(sys.argv) > 1 and sys.argv[1].endswith('.json'):
            SENTIMENT_FILE = sys.argv[1]
        else:
            candidates = sorted(glob.glob('*_llm_sentiment.json'))
            if not candidates:
                candidates = sorted(glob.glob('*_sentiment.json'))
            SENTIMENT_FILE = candidates[-1] if candidates else None

        if SENTIMENT_FILE and os.path.exists(SENTIMENT_FILE):
            print(f'  情绪数据来源: JSON 文件 ({SENTIMENT_FILE})')
            with open(SENTIMENT_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            ts = data['time_series']
            posts = data['per_post_results']
            posts.sort(key=lambda p: p['created_at'])
            meta = data.get('metadata', {})
            stock_code = meta.get('stock_code', '02400')
        else:
            print('  错误: 无缓存数据且无 JSON 文件。请先运行 llm_sentiment.py')
            sys.exit(1)

    # ============================================================
    # Load price data from Tencent Finance API & align with sentiment date range
    # ============================================================
    all_prices = fetch_hk_daily_kline(stock_code)
    sentiment_dates = set(r['date'] for r in ts)
    price_data = [p for p in all_prices if p['date'] in sentiment_dates]
    # Also include last few days before/after for context
    min_date = min(sentiment_dates)
    max_date = max(sentiment_dates)
    price_data_full = [p for p in all_prices if min_date <= p['date'] <= max_date]
    if len(price_data_full) > len(price_data):
        price_data = price_data_full  # use full range if available

    print(f'  Sentiment days: {len(ts)}, Price days matched: {len(price_data)} (Tencent Finance API)')

    # Embed as JS
    ts_json = json.dumps(ts, ensure_ascii=False)
    posts_json = json.dumps(posts, ensure_ascii=False)
    price_json = json.dumps(price_data, ensure_ascii=False)
    meta_json = json.dumps(meta, ensure_ascii=False)

    date_range_str = f"{ts[0]['date']} ~ {ts[-1]['date']}"

    html = f'''<!DOCTYPE html>
    <html lang="zh-CN">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{stock_code}.HK 雪球社区情绪仪表盘 · DeepSeek LLM</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
      :root {{
        --bg: #0f1118;
        --card: #1a1d2a;
        --border: #2a2d3a;
        --text: #c9cdd4;
        --text-dim: #6b7084;
        --accent: #6366f1;
        --bull: #22c55e;
        --bear: #ef4444;
        --neutral: #f59e0b;
      }}
      * {{ margin: 0; padding: 0; box-sizing: border-box; }}
      body {{
        background: var(--bg);
        color: var(--text);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
        line-height: 1.5;
        min-height: 100vh;
      }}
      .container {{ max-width: 1400px; margin: 0 auto; padding: 20px 24px; }}

      .header {{
        display: flex; align-items: center; justify-content: space-between;
        padding: 20px 0; border-bottom: 1px solid var(--border); margin-bottom: 24px;
        flex-wrap: wrap; gap: 12px;
      }}
      .header h1 {{ font-size: 1.5rem; font-weight: 700; }}
      .header .stock-tag {{ background: var(--accent); color: #fff; padding: 3px 12px; border-radius: 4px; font-size: 0.85rem; font-weight: 600; }}
      .header .meta {{ color: var(--text-dim); font-size: 0.85rem; }}
      .model-badge {{ background: rgba(99,102,241,0.2); color: #a5b4fc; padding: 3px 10px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }}

      .summary-grid {{
        display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
        gap: 16px; margin-bottom: 24px;
      }}
      .summary-card {{
        background: var(--card); border: 1px solid var(--border); border-radius: 10px;
        padding: 18px 20px; position: relative; overflow: hidden;
      }}
      .summary-card .label {{ font-size: 0.78rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }}
      .summary-card .value {{ font-size: 2rem; font-weight: 700; }}
      .summary-card .sub {{ font-size: 0.78rem; color: var(--text-dim); margin-top: 4px; }}

      .charts-grid {{ display: grid; grid-template-columns: 2fr 1fr; gap: 16px; margin-bottom: 24px; }}
      .charts-grid.full {{ grid-template-columns: 1fr; }}
      .charts-grid.triple {{ grid-template-columns: 1fr 1fr 1fr; }}
      .chart-card {{
        background: var(--card); border: 1px solid var(--border); border-radius: 10px;
        padding: 20px; position: relative;
      }}
      .chart-card h3 {{ font-size: 0.95rem; font-weight: 600; margin-bottom: 16px; }}
      .chart-wrap {{ position: relative; width: 100%; }}

      .table-card {{
        background: var(--card); border: 1px solid var(--border); border-radius: 10px;
        padding: 20px; margin-bottom: 24px; overflow-x: auto;
      }}
      .table-card h3 {{ font-size: 0.95rem; font-weight: 600; margin-bottom: 14px; }}
      table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
      th {{ text-align: left; padding: 10px 12px; border-bottom: 2px solid var(--border); color: var(--text-dim); font-weight: 600; font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap; }}
      td {{ padding: 9px 12px; border-bottom: 1px solid rgba(42,45,58,0.5); }}
      tr:hover td {{ background: rgba(255,255,255,0.02); }}
      .badge {{
        display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.76rem;
        font-weight: 600;
      }}
      .badge-bearish {{ background: rgba(239,68,68,0.18); color: var(--bear); }}
      .badge-s_bearish {{ background: rgba(249,115,22,0.18); color: #f97316; }}
      .badge-neutral {{ background: rgba(245,158,11,0.18); color: var(--neutral); }}
      .badge-s_bullish {{ background: rgba(132,204,22,0.18); color: #84cc16; }}
      .badge-bullish {{ background: rgba(34,197,94,0.18); color: var(--bull); }}

      .score-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }}

      .tab-bar {{ display: flex; gap: 4px; margin-bottom: 20px; }}
      .tab-btn {{
        padding: 8px 20px; border: 1px solid var(--border); border-radius: 6px;
        background: transparent; color: var(--text-dim); cursor: pointer;
        font-size: 0.88rem; transition: all 0.2s;
      }}
      .tab-btn.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
      .tab-btn:hover {{ color: #fff; }}

      .framework-grid {{
        display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 14px; margin-bottom: 20px;
      }}
      .framework-item {{
        background: rgba(99,102,241,0.06); border: 1px solid rgba(99,102,241,0.2); border-radius: 8px;
        padding: 14px 16px;
      }}
      .framework-item .dim {{ font-size: 0.82rem; color: #a5b4fc; font-weight: 700; margin-bottom: 4px; }}
      .framework-item .desc {{ font-size: 0.82rem; color: var(--text-dim); }}

      @media (max-width: 768px) {{
        .charts-grid {{ grid-template-columns: 1fr; }}
        .charts-grid.triple {{ grid-template-columns: 1fr; }}
        .summary-card .value {{ font-size: 1.5rem; }}
      }}

      .sentiment-legend {{
        display: flex; gap: 16px; flex-wrap: wrap; font-size: 0.82rem; color: var(--text-dim);
        margin-top: 12px;
      }}
      .sentiment-legend span {{ display: flex; align-items: center; gap: 4px; }}
    </style>
    </head>
    <body>
    <div class="container">

      <div class="header">
        <div>
          <h1><span class="stock-tag">{stock_code}.HK</span> 社区情绪仪表盘</h1>
          <div class="meta">数据来源: 雪球 · 分析区间: {date_range_str} · 总帖数: {len(posts)} · <span class="model-badge">DeepSeek LLM</span></div>
        </div>
        <div style="text-align:right">
          <div style="font-size:0.78rem;color:var(--text-dim)">分析引擎</div>
          <div style="font-weight:700;font-size:1rem;color:#a5b4fc">deepseek-chat · 语义级情绪评分</div>
        </div>
      </div>

      <div class="summary-grid" id="summaryCards"></div>

      <div class="charts-grid">
        <div class="chart-card">
          <h3>日度情绪得分走势 (LLM语义分析)</h3>
          <div class="chart-wrap"><canvas id="sentimentChart"></canvas></div>
          <div class="sentiment-legend">
            <span><span style="color:#ef4444">■</span> 0-30 极度悲观</span>
            <span><span style="color:#f97316">■</span> 30-45 偏悲观</span>
            <span><span style="color:#f59e0b">■</span> 45-55 中性</span>
            <span><span style="color:#84cc16">■</span> 55-70 偏乐观</span>
            <span><span style="color:#22c55e">■</span> 70-100 极度乐观</span>
          </div>
        </div>
        <div class="chart-card">
          <h3>情绪分布 (看多/中性/看空 %)</h3>
          <div class="chart-wrap"><canvas id="distributionChart"></canvas></div>
        </div>
      </div>

      <div class="charts-grid">
        <div class="chart-card">
          <h3>每日发帖量与情绪对比</h3>
          <div class="chart-wrap"><canvas id="volumeChart"></canvas></div>
        </div>
        <div class="chart-card">
          <h3>情绪离散度与分歧指数</h3>
          <div class="chart-wrap"><canvas id="volatilityChart"></canvas></div>
        </div>
      </div>

      <div class="charts-grid full">
        <div class="chart-card">
          <h3>逐帖情绪评分散点图 (按时间排列, 大小=置信度)</h3>
          <div class="chart-wrap"><canvas id="scatterChart"></canvas></div>
        </div>
      </div>

      <div class="chart-card" style="margin-bottom:20px">
        <h3>分析框架 · DeepSeek LLM 语义情绪评分</h3>
        <div class="framework-grid">
          <div class="framework-item">
            <div class="dim">语义理解</div>
            <div class="desc">大模型理解投资社区语境，区分理性分析/情绪宣泄、讽刺/真诚、短期/长期判断</div>
          </div>
          <div class="framework-item">
            <div class="dim">多维评分</div>
            <div class="desc">综合评估业务态度、估值判断、语气强度、论据质量、预期传播等维度</div>
          </div>
          <div class="framework-item">
            <div class="dim">方向 + 强度 + 置信度</div>
            <div class="desc">每帖输出 bull/bear/neutral 方向 + 0-100 强度分 + 判断置信度</div>
          </div>
          <div class="framework-item">
            <div class="dim">日度聚合</div>
            <div class="desc">按日聚合为加权均分(互动加权)、分布比例、标准差等统计量</div>
          </div>
          <div class="framework-item">
            <div class="dim">批次处理</div>
            <div class="desc">179帖分12批发送，低温度(0.1)保证评分一致性，3次重试保障鲁棒性</div>
          </div>
        </div>
      </div>

      <div class="table-card">
        <h3>日度情绪时间序列表</h3>
        <table id="dailyTable">
          <thead><tr><th>日期</th><th>帖数</th><th>加权均分</th><th>算术均分</th><th>中位数</th><th>标准差</th><th>看多%</th><th>中性%</th><th>看空%</th><th>情绪标签</th><th>置信度</th><th>得分柱</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>

      <div class="table-card">
        <h3>逐帖情绪评分明细 (按时间排序)</h3>
        <table id="postTable" style="font-size:0.82rem">
          <thead><tr><th style="width:80px">时间</th><th style="width:70px">用户</th><th>帖子内容</th><th style="width:60px">得分</th><th style="width:55px">方向</th><th style="width:55px">置信度</th><th>判断依据</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>

    </div>

    <script>
    const TS = {ts_json};
    const POSTS = {posts_json};
    const PRICE = {price_json};

    function scoreColor(s) {{
      if (s >= 70) return '#22c55e';
      if (s >= 55) return '#84cc16';
      if (s >= 45) return '#f59e0b';
      if (s >= 30) return '#f97316';
      return '#ef4444';
    }}

    function scoreBg(s, alpha) {{
      alpha = alpha || 0.15;
      if (s >= 70) return `rgba(34,197,94,${{alpha}})`;
      if (s >= 55) return `rgba(132,204,22,${{alpha}})`;
      if (s >= 45) return `rgba(245,158,11,${{alpha}})`;
      if (s >= 30) return `rgba(249,115,22,${{alpha}})`;
      return `rgba(239,68,68,${{alpha}})`;
    }}

    function badgeClass(label) {{
      if (label === 'bearish') return 'badge-bearish';
      if (label === 'slightly_bearish') return 'badge-s_bearish';
      if (label === 'neutral') return 'badge-neutral';
      if (label === 'slightly_bullish') return 'badge-s_bullish';
      return 'badge-bullish';
    }}

    function labelText(label) {{
      const map = {{ bearish:'看空', slightly_bearish:'偏空', neutral:'中性', slightly_bullish:'偏多', bullish:'看多' }};
      return map[label] || label;
    }}

    function dirText(d) {{ return d === 'bullish' ? '看多' : d === 'bearish' ? '看空' : '中性'; }}

    Chart.defaults.color = '#6b7084';
    Chart.defaults.borderColor = 'rgba(42,45,58,0.6)';
    Chart.defaults.font.family = "-apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei',sans-serif";
    Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(26,29,42,0.95)';
    Chart.defaults.plugins.tooltip.borderColor = '#2a2d3a';
    Chart.defaults.plugins.tooltip.borderWidth = 1;
    Chart.defaults.plugins.tooltip.padding = 12;

    // === Summary Cards ===
    const scores = TS.map(d => d.weighted_mean);
    const avgScore = scores.reduce((a,b)=>a+b,0) / scores.length;
    const maxScore = Math.max(...scores);
    const minScore = Math.min(...scores);
    const maxDay = TS[scores.indexOf(maxScore)].date;
    const minDay = TS[scores.indexOf(minScore)].date;
    const totalPosts = TS.reduce((a,b)=>a+b.post_count,0);
    const trendDir = scores[scores.length-1] > scores[0] ? 'warming' : 'cooling';
    const latestScore = scores[scores.length-1];

    // Count bullish vs bearish posts
    const allBullish = POSTS.filter(p => p.direction === 'bullish').length;
    const allBearish = POSTS.filter(p => p.direction === 'bearish').length;
    const allNeutral = POSTS.filter(p => p.direction === 'neutral').length;

    document.getElementById('summaryCards').innerHTML = `
      <div class="summary-card">
        <div class="label">周期平均情绪</div>
        <div class="value" style="color:${{scoreColor(avgScore)}}">${{avgScore.toFixed(1)}}</div>
        <div class="sub">区间: ${{minScore}} ~ ${{maxScore}} · 振幅 ${{(maxScore-minScore).toFixed(1)}}</div>
      </div>
      <div class="summary-card">
        <div class="label">最新情绪 (${{TS[TS.length-1].date.slice(5)}})</div>
        <div class="value" style="color:${{scoreColor(latestScore)}}">${{latestScore.toFixed(1)}}</div>
        <div class="sub">${{labelText(TS[TS.length-1].sentiment_label)}} · 趋势: ${{trendDir==='cooling'?'偏冷 ↓':'偏暖 ↑'}}</div>
      </div>
      <div class="summary-card">
        <div class="label">全周期帖情绪方向</div>
        <div class="value" style="font-size:1.5rem"><span style="color:#22c55e">${{allBullish}}</span> / <span style="color:#f59e0b">${{allNeutral}}</span> / <span style="color:#ef4444">${{allBearish}}</span></div>
        <div class="sub">看多/中性/看空 帖数 · 共${{totalPosts}}帖</div>
      </div>
      <div class="summary-card">
        <div class="label">最高情绪日</div>
        <div class="value" style="color:${{scoreColor(maxScore)}}">${{maxScore.toFixed(1)}}</div>
        <div class="sub">${{maxDay}} · ${{TS[scores.indexOf(maxScore)].post_count}}帖 · ${{labelText(TS[scores.indexOf(maxScore)].sentiment_label)}}</div>
      </div>
      <div class="summary-card">
        <div class="label">最低情绪日</div>
        <div class="value" style="color:${{scoreColor(minScore)}}">${{minScore.toFixed(1)}}</div>
        <div class="sub">${{minDay}} · ${{TS[scores.indexOf(minScore)].post_count}}帖 · ${{labelText(TS[scores.indexOf(minScore)].sentiment_label)}}</div>
      </div>
    `;

    // === Chart 1: Sentiment + Close Price Dual-Axis ===
    const priceMap = {{}};
    PRICE.forEach(p => {{ priceMap[p.date] = p.close; }});
    const priceDataAligned = TS.map(d => priceMap[d.date] || null);

    new Chart(document.getElementById('sentimentChart'), {{
      type: 'line',
      data: {{
        labels: TS.map(d => d.date),
        datasets: [
          {{
            label: '加权均分 (互动加权)',
            data: TS.map(d => d.weighted_mean),
            borderColor: '#6366f1',
            backgroundColor: 'rgba(99,102,241,0.08)',
            borderWidth: 2.5, tension: 0.4, fill: true,
            pointRadius: 5, pointBackgroundColor: '#6366f1', pointBorderColor: '#1a1d2a', pointBorderWidth: 2,
            yAxisID: 'y',
            order: 1,
          }},
          {{
            label: '收盘价 (港元)',
            data: priceDataAligned,
            borderColor: '#f59e0b',
            backgroundColor: 'rgba(245,158,11,0.05)',
            borderWidth: 2.5, tension: 0.4, fill: false,
            pointRadius: 5, pointBackgroundColor: '#f59e0b', pointBorderColor: '#1a1d2a', pointBorderWidth: 2,
            borderDash: [4, 2],
            yAxisID: 'y1',
            order: 0,
            spanGaps: true,
          }}
        ]
      }},
      options: {{
        responsive: true,
        interaction: {{ intersect: false, mode: 'index' }},
        plugins: {{
          legend: {{ labels: {{ usePointStyle: true, padding: 20, font: {{ size: 11 }} }} }},
          tooltip: {{
            callbacks: {{
              label: ctx => {{
                if (ctx.datasetIndex === 0) return `情绪均分: ${{ctx.raw.toFixed(1)}}`;
                return ctx.raw ? `收盘价: HK$${{ctx.raw.toFixed(2)}}` : '休市';
              }},
              afterBody: items => {{
                const d = TS[items[0].dataIndex];
                const px = priceMap[d.date];
                return [`帖数: ${{d.post_count}}  看多: ${{d.bullish_pct}}%  中性: ${{d.neutral_pct}}%  看空: ${{d.bearish_pct}}%`, px ? `收盘价: HK$${{px}}` : ''];
              }}
            }}
          }}
        }},
        scales: {{
          y: {{
            type: 'linear', position: 'left',
            title: {{ display: true, text: '情绪得分 (0-100)', color: '#6366f1' }},
            min: 30, max: 80, ticks: {{ stepSize: 5, color: '#6366f1' }},
            grid: {{ color: 'rgba(42,45,58,0.4)' }}
          }},
          y1: {{
            type: 'linear', position: 'right',
            title: {{ display: true, text: '收盘价 (港元)', color: '#f59e0b' }},
            ticks: {{ color: '#f59e0b' }},
            grid: {{ drawOnChartArea: false }}
          }},
          x: {{ grid: {{ display: false }} }}
        }}
      }}
    }});

    // === Chart 2: Distribution ===
    new Chart(document.getElementById('distributionChart'), {{
      type: 'bar',
      data: {{
        labels: TS.map(d => d.date),
        datasets: [
          {{ label: '看多', data: TS.map(d => d.bullish_pct), backgroundColor: '#22c55e', borderRadius: 2 }},
          {{ label: '中性', data: TS.map(d => d.neutral_pct), backgroundColor: '#f59e0b', borderRadius: 2 }},
          {{ label: '看空', data: TS.map(d => d.bearish_pct), backgroundColor: '#ef4444', borderRadius: 2 }},
        ]
      }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ labels: {{ usePointStyle: true, padding: 16, font: {{ size: 11 }} }} }} }},
        scales: {{
          x: {{ stacked: true, grid: {{ display: false }} }},
          y: {{ stacked: true, max: 100, ticks: {{ callback: v => v+'%' }}, grid: {{ color: 'rgba(42,45,58,0.4)' }} }}
        }}
      }}
    }});

    // === Chart 3: Volume vs Sentiment ===
    new Chart(document.getElementById('volumeChart'), {{
      type: 'bar',
      data: {{
        labels: TS.map(d => d.date),
        datasets: [
          {{
            label: '发帖量', data: TS.map(d => d.post_count),
            backgroundColor: TS.map(d => scoreBg(d.weighted_mean, 0.25)),
            borderColor: TS.map(d => scoreColor(d.weighted_mean)),
            borderWidth: 1.5, borderRadius: 4,
            order: 0, yAxisID: 'y',
          }},
          {{
            label: '加权均分', data: TS.map(d => d.weighted_mean),
            type: 'line',
            borderColor: '#fff', borderWidth: 2.5, tension: 0.4, fill: false,
            pointRadius: 5, pointBackgroundColor: TS.map(d => scoreColor(d.weighted_mean)),
            pointBorderColor: '#1a1d2a', pointBorderWidth: 2,
            order: 1, yAxisID: 'y1',
          }}
        ]
      }},
      options: {{
        responsive: true,
        interaction: {{ intersect: false, mode: 'index' }},
        plugins: {{ legend: {{ labels: {{ usePointStyle: true, padding: 20, font: {{ size: 11 }} }} }} }},
        scales: {{
          x: {{ grid: {{ display: false }} }},
          y: {{ type: 'linear', position: 'left', title: {{ display: true, text: '帖子数', color: '#6b7084' }}, grid: {{ color: 'rgba(42,45,58,0.4)' }} }},
          y1: {{ type: 'linear', position: 'right', title: {{ display: true, text: '情绪分', color: '#6b7084' }}, min: 30, max: 70, grid: {{ drawOnChartArea: false }} }}
        }}
      }}
    }});

    // === Chart 4: Volatility (Std Dev) ===
    new Chart(document.getElementById('volatilityChart'), {{
      type: 'bar',
      data: {{
        labels: TS.map(d => d.date),
        datasets: [{{
          label: '标准差', data: TS.map(d => d.std_dev),
          backgroundColor: TS.map(d => {{
            const v = d.std_dev;
            if (v > 20) return 'rgba(239,68,68,0.6)';
            if (v > 15) return 'rgba(249,115,22,0.5)';
            if (v > 10) return 'rgba(245,158,11,0.4)';
            return 'rgba(34,197,94,0.35)';
          }}),
          borderColor: TS.map(d => {{
            const v = d.std_dev;
            if (v > 20) return '#ef4444'; if (v > 15) return '#f97316';
            if (v > 10) return '#f59e0b'; return '#22c55e';
          }}),
          borderWidth: 1.5, borderRadius: 4,
        }}]
      }},
      options: {{
        responsive: true,
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{ callbacks: {{
            label: ctx => `标准差: ${{ctx.raw.toFixed(1)}}`,
            afterBody: items => {{
              const s = TS[items[0].dataIndex].std_dev;
              return [s < 8 ? '情绪高度一致' : s < 14 ? '有一定分歧' : s < 20 ? '分歧明显' : '高度分歧'];
            }}
          }} }}
        }},
        scales: {{
          x: {{ grid: {{ display: false }} }},
          y: {{ min: 0, max: 25, grid: {{ color: 'rgba(42,45,58,0.4)' }} }}
        }}
      }}
    }});

    // === Chart 5: Post-level Scatter ===
    const scatterData = POSTS.map((p, i) => ({{
      x: p.created_at,
      y: p.score,
      r: Math.max(3, p.confidence * 8),
      direction: p.direction,
      rationale: p.rationale,
      user: p.user_name,
      text: p.text_preview,
    }}));

    new Chart(document.getElementById('scatterChart'), {{
      type: 'bubble',
      data: {{
        datasets: [
          {{
            label: '看多', data: scatterData.filter(d => d.direction === 'bullish'),
            backgroundColor: 'rgba(34,197,94,0.5)', borderColor: '#22c55e', borderWidth: 1,
          }},
          {{
            label: '中性', data: scatterData.filter(d => d.direction === 'neutral'),
            backgroundColor: 'rgba(245,158,11,0.5)', borderColor: '#f59e0b', borderWidth: 1,
          }},
          {{
            label: '看空', data: scatterData.filter(d => d.direction === 'bearish'),
            backgroundColor: 'rgba(239,68,68,0.5)', borderColor: '#ef4444', borderWidth: 1,
          }},
        ]
      }},
      options: {{
        responsive: true,
        plugins: {{
          legend: {{ labels: {{ usePointStyle: true, padding: 16, font: {{ size: 11 }} }} }},
          tooltip: {{ callbacks: {{
            label: ctx => `${{ctx.raw.user}}: ${{ctx.raw.y}}分 - ${{ctx.raw.rationale}}`,
            afterLabel: ctx => `${{ctx.raw.text}}`
          }} }}
        }},
        scales: {{
          x: {{
            type: 'category', grid: {{ display: false }},
            ticks: {{ font: {{ size: 9 }}, maxRotation: 45, autoSkip: true, maxTicksLimit: 20 }}
          }},
          y: {{ min: 0, max: 100, ticks: {{ stepSize: 10 }}, grid: {{ color: 'rgba(42,45,58,0.4)' }} }}
        }}
      }}
    }});

    // === Daily Table ===
    let tbody = '';
    [...TS].reverse().forEach(d => {{
      const sc = scoreColor(d.weighted_mean);
      const barW = ((d.weighted_mean - 20) / 60 * 100).toFixed(0);
      tbody += `<tr>
        <td style="font-weight:600">${{d.date}}</td>
        <td>${{d.post_count}}</td>
        <td style="color:${{sc}};font-weight:700">${{d.weighted_mean.toFixed(1)}}</td>
        <td>${{d.mean_score.toFixed(1)}}</td>
        <td>${{d.median_score}}</td>
        <td>${{d.std_dev.toFixed(1)}}</td>
        <td style="color:#22c55e">${{d.bullish_pct}}%</td>
        <td style="color:#f59e0b">${{d.neutral_pct}}%</td>
        <td style="color:#ef4444">${{d.bearish_pct}}%</td>
        <td><span class="badge ${{badgeClass(d.sentiment_label)}}">${{labelText(d.sentiment_label)}}</span></td>
        <td>${{d.avg_confidence}}</td>
        <td>
          <div style="display:flex;align-items:center;gap:6px">
            <div style="flex:1;height:5px;border-radius:3px;background:rgba(255,255,255,0.06);overflow:hidden">
              <div style="width:${{barW}}%;height:100%;border-radius:3px;background:${{sc}}"></div>
            </div>
            <span style="font-size:0.78rem;color:${{sc}};font-weight:600">${{d.weighted_mean.toFixed(0)}}</span>
          </div>
        </td>
      </tr>`;
    }});
    document.querySelector('#dailyTable tbody').innerHTML = tbody;

    // === Post-level Table ===
    let ptbody = '';
    [...POSTS].reverse().forEach(p => {{
      const sc = scoreColor(p.score);
      ptbody += `<tr>
        <td style="white-space:nowrap;font-size:0.78rem;color:var(--text-dim)">${{p.created_at.slice(5)}}</td>
        <td style="font-size:0.8rem">${{p.user_name}}</td>
        <td style="max-width:400px">${{p.text_preview}}</td>
        <td style="color:${{sc}};font-weight:700">${{p.score}}</td>
        <td><span class="badge ${{p.direction==='bullish'?'badge-bullish':p.direction==='bearish'?'badge-bearish':'badge-neutral'}}">${{dirText(p.direction)}}</span></td>
        <td>${{p.confidence}}</td>
        <td style="color:var(--text-dim);font-size:0.8rem">${{p.rationale}}</td>
      </tr>`;
    }});
    document.querySelector('#postTable tbody').innerHTML = ptbody;
    </script>
    </body>
    </html>'''

    with open('sentiment_dashboard_llm.html', 'w', encoding='utf-8') as f:
        f.write(html)

    print(f'Dashboard written: sentiment_dashboard_llm.html ({len(html)} bytes)')
    print(f'  Time series: {len(ts)} days')
    print(f'  Per-post results: {len(posts)} posts')


if __name__ == "__main__":
    main()
