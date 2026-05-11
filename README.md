# 心动公司(02400.HK) 雪球社区情绪分析

基于 DeepSeek LLM 的雪球社区情绪分析系统。对心动公司相关帖子进行语义级情绪评分，生成交互式可视化仪表盘。

## 依赖项目

> **数据源**：本项目依赖 [xueqiu-scraper](https://github.com/as167888/xueqiu-scraper) 抓取的雪球帖子数据库 (`xueqiu_stock.db`)。请先运行 xueqiu-scraper 获取帖子数据。

## 数据管线

```
xueqiu_stock.db        sentiment_cache.db
(雪球帖子原始数据)      (LLM 分析结果缓存)
       │                      │
       ▼                      ▼
  llm_sentiment.py ──→ DeepSeek API ──→ sentiment_cache.db
       │                                      │
       │                                      ▼
       │                             build_dashboard.py
       │                                      │
       │                   腾讯财经 API ◄─────┤  (港股日K线收盘价)
       │                                      │
       ▼                                      ▼
  main.py (主控) ────────────────────→ sentiment_dashboard_llm.html
```

1. **`llm_sentiment.py`** — 从 `xueqiu_stock.db` 读取帖子，调用 DeepSeek API 进行语义情绪评分（0-100 分），结果缓存至 `sentiment_cache.db`
2. **`build_dashboard.py`** — 从本地数据库加载情绪数据，通过腾讯财经 API 获取港股收盘价，生成自包含 HTML 仪表盘
3. **`main.py`** — 交互式主控菜单，统筹分析 + 仪表盘生成流程

## 快速开始

### 环境要求

```bash
pip install openai requests
```

### 配置

```bash
# DeepSeek API Key
export DEEPSEEK_API_KEY="your-api-key"

# 雪球数据库路径（可选，默认为 xueqiu-scraper 默认路径）
export XUEQIU_DB_PATH="E:\\project\\xueqiu\\xueqiu_stock.db"
```

### 运行

```bash
# 交互式菜单
python main.py

# 或直接生成仪表盘
python build_dashboard.py 02400
```

## 仪表盘功能

- 日度情绪得分走势 + 收盘价双轴图
- 看多 / 中性 / 看空 情绪分布堆叠图
- 每日发帖量与情绪对比
- 情绪离散度（标准差）分析
- 逐帖气泡散点图（时间 × 得分 × 置信度）
- 日度汇总表 + 逐帖明细表（按时间倒序）

## 情绪评分维度

DeepSeek LLM 从以下维度对每条帖子进行语义分析：

- 对公司/业务的态度（乐观 ↔ 悲观）
- 对股价/估值的判断（看涨 ↔ 看跌）
- 语气强度（淡定 ↔ 强烈）
- 论据质量（数据逻辑支撑 ↔ 纯情绪宣泄）
- 预期传播方向

输出：每帖 0-100 情绪分 + bull/bear/neutral 方向 + 置信度。

## 技术栈

- **LLM**: DeepSeek Chat API（低温度 0.1，批量分析，3 次重试）
- **数据**: SQLite（xueqiu-scraper 输出 + sentiment_cache 本地缓存）
- **股价**: 腾讯财经港股日K线 API（`web.ifzq.gtimg.cn`）
- **可视化**: Chart.js 4.x（暗色主题，自包含 HTML）
- **缓存策略**: 增量分析，仅对新帖子调用 API，已分析帖子直接复用缓存
