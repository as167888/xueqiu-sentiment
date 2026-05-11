#!/usr/bin/env python3
"""
心动公司(02400.HK) 雪球社区情绪分析 · 主控程序
==============================================
统筹 llm_sentiment.py（情绪分析）与 build_dashboard.py（仪表盘生成）。

数据流:
  xueqiu_stock.db::stock_posts ──→ llm_sentiment.py ──→ sentiment_cache.db
                                       (DeepSeek API)       (本地缓存)

  sentiment_cache.db + xueqiu_stock.db ──→ build_dashboard.py ──→ HTML 仪表盘
                                             (直接从 DB 读取)

核心设计:
  - 数据源: 仅读取 stock_posts 表（个股贴文），不读取 user_posts / column_articles
  - 缓存: 已分析的帖子结果保存在 sentiment_cache.db，避免重复调用 API
  - 仪表盘: 直接从本地数据库加载情绪数据，不依赖中间 JSON 文件
"""

import os
import sys
import sqlite3
import subprocess
import unicodedata
import textwrap
from datetime import datetime

# 确保 Windows 控制台使用 UTF-8 编码
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("XUEQIU_DB_PATH", r"E:\project\xueqiu\xueqiu_stock.db")
CACHE_DB_PATH = os.environ.get("SENTIMENT_CACHE_DB", os.path.join(BASE_DIR, "sentiment_cache.db"))
LLM_SCRIPT = os.path.join(BASE_DIR, "llm_sentiment.py")
DASHBOARD_SCRIPT = os.path.join(BASE_DIR, "build_dashboard.py")

# ============================================================
# 工具函数
# ============================================================
def dwidth(s):
    """字符串显示宽度。CJK 字符及全角符号占 2 列，其余占 1 列。"""
    return sum(2 if unicodedata.east_asian_width(c) in ('F', 'W') else 1 for c in str(s))


def pad(s, width):
    """按显示宽度右填充空格至 target width。"""
    cur = dwidth(s)
    return s + ' ' * (width - cur) if cur < width else s


def sep(char='=', width=62):
    """打印分隔线。"""
    print(f"  {char * width}")


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """打印标题头。"""
    sep('=')
    print(f"  {pad('心动公司(02400.HK) 雪球社区情绪分析系统', 60)}")
    print(f"  {'LLM-Based Sentiment Analysis  -  DeepSeek API'}")
    sep('=')


def print_section(title):
    """打印小标题。"""
    print(f"\n  -- {title} --")


# ============================================================
# 状态收集与展示
# ============================================================
def get_status():
    """收集当前流水线状态。"""
    status = {'db_ok': False, 'stock_posts': 0, 'cached': 0, 'new_posts': 0,
              'latest_json': None, 'latest_html': None}

    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.execute("SELECT COUNT(*) FROM stock_posts WHERE stock='02400'")
            status['stock_posts'] = cur.fetchone()[0]
            conn.close()
            status['db_ok'] = True
        except Exception:
            pass

    if os.path.exists(CACHE_DB_PATH):
        try:
            conn = sqlite3.connect(CACHE_DB_PATH)
            conn.row_factory = sqlite3.Row
            cur = conn.execute("SELECT post_id FROM sentiment_cache")
            cached_ids = {row['post_id'] for row in cur.fetchall()}
            status['cached'] = len(cached_ids)

            if status['db_ok']:
                src_conn = sqlite3.connect(DB_PATH)
                src_cur = src_conn.execute("SELECT id FROM stock_posts WHERE stock='02400'")
                all_ids = {row[0] for row in src_cur.fetchall()}
                src_conn.close()
                status['new_posts'] = len(all_ids - cached_ids)
            conn.close()
        except Exception:
            pass

    import glob
    jsons = sorted(glob.glob(os.path.join(BASE_DIR, '*_llm_sentiment.json')))
    if jsons:
        status['latest_json'] = os.path.basename(jsons[-1])
    htmls = sorted(glob.glob(os.path.join(BASE_DIR, 'sentiment_dashboard*.html')))
    if htmls:
        status['latest_html'] = os.path.basename(htmls[-1])

    return status


def print_status(status):
    """打印状态面板。"""
    print_section("系统状态")
    print()

    # 使用统一标签宽度，按显示宽度对齐
    LABEL_W = 16  # 标签最大显示宽度

    db_icon = "OK" if status['db_ok'] else "FAIL"
    print(f"  {pad('源数据库', LABEL_W)}{db_icon:<6}{DB_PATH}")

    print(f"  {pad('个股贴文', LABEL_W)}{status['stock_posts']:<6}条 (stock_posts 表)")

    print(f"  {pad('已缓存分析', LABEL_W)}{status['cached']:<6}条 (sentiment_cache.db)")

    if status['new_posts'] > 0:
        print(f"  {pad('待分析新帖', LABEL_W)}{status['new_posts']:<6}条")
    else:
        print(f"  {pad('待分析新帖', LABEL_W)}0     条 (缓存已完整)")

    if status['latest_json']:
        print(f"  {pad('最新 JSON', LABEL_W)}{status['latest_json']}")
    if status['latest_html']:
        print(f"  {pad('最新仪表盘', LABEL_W)}{status['latest_html']}")

    # API Key 状态
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if api_key:
        print(f"  {pad('DeepSeek API', LABEL_W)}OK    已配置 (环境变量)")
    else:
        print(f"  {pad('DeepSeek API', LABEL_W)}WARN  使用默认 Key")


# ============================================================
# 执行步骤
# ============================================================
def run_analysis(stock_code='02400', batch_size=15):
    """运行情绪分析。"""
    print_section("步骤 1/2: LLM 情绪分析")
    print(f"  数据来源   xueqiu_stock.db  →  stock_posts 表")
    print(f"  缓存位置   sentiment_cache.db")
    print(f"  策略       仅对新增帖子调用 DeepSeek API，已缓存帖直接复用")
    print()

    cmd = [sys.executable, LLM_SCRIPT, stock_code, str(batch_size)]
    result = subprocess.run(cmd, cwd=BASE_DIR)
    return result.returncode == 0


def run_dashboard(stock_code='02400'):
    """生成仪表盘。"""
    print_section("步骤 2/2: 生成可视化仪表盘")
    print(f"  数据来源   sentiment_cache.db + xueqiu_stock.db (直接从本地数据库加载)")
    print(f"  价格数据   腾讯财经 API (港股日K线)")
    print()

    cmd = [sys.executable, DASHBOARD_SCRIPT, stock_code]
    result = subprocess.run(cmd, cwd=BASE_DIR)
    return result.returncode == 0


def open_dashboard():
    """在浏览器中打开最新仪表盘。"""
    import glob as g
    htmls = sorted(g.glob(os.path.join(BASE_DIR, 'sentiment_dashboard*.html')))
    if not htmls:
        print("\n  未找到仪表盘 HTML 文件，请先生成。")
        input("  按 Enter 继续...")
        return
    latest = htmls[-1]
    print(f"\n  正在打开: {os.path.basename(latest)}")
    os.startfile(latest)


# ============================================================
# 流水线原理说明
# ============================================================
def show_documentation():
    """显示流水线原理。"""
    clear_screen()
    print_header()
    print_section("流水线原理")
    print()

    doc = """
  数据流水线架构
  ──────────────

  ① 数据源
     xueqiu_stock.db::stock_posts   — 个股贴文 (1,014 条, 多用户)
     xueqiu_stock.db::user_posts    — 逸修博主贴文 (不使用)
     xueqiu_stock.db::column_articles — 逸修专栏文章 (不使用)

  ② 情绪分析 (llm_sentiment.py)
     1. 读取 stock_posts 表
     2. 查询 sentiment_cache.db — 哪些帖子已分析过?
     3. 仅新帖发送至 DeepSeek API (15 条/批)
     4. 分析结果写入 sentiment_cache.db (INSERT OR REPLACE)
     5. 日度聚合 → JSON + CSV 输出 (可选)

  ③ 本地缓存 (sentiment_cache.db)
     表: sentiment_cache
     字段: post_id | score | direction | confidence | rationale | analyzed_at
     作用: 避免重复调用 API，仅对新增帖子产生费用

  ④ 仪表盘生成 (build_dashboard.py)
     1. 从 sentiment_cache.db 加载所有情绪评分
     2. 从 xueqiu_stock.db 获取帖子元数据 (用户名/时间/正文)
     3. 日度聚合计算 (互动加权均分/分布/标准差)
     4. 读取 2400.HK.xlsx 股价数据
     5. 生成自包含 HTML 仪表盘 (数据内嵌, 无需服务器)

  ⑤ HTML 仪表盘
     · 情绪得分 + 股价双轴走势图
     · 看多 / 中性 / 看空 分布堆叠图
     · 每日发帖量 + 情绪对比图
     · 情绪离散度 (标准差) 分析
     · 逐帖气泡散点图 (时间 × 得分 × 置信度)
     · 日度汇总表 + 逐帖明细表

  核心设计
  ────────
  · 数据源单一  — 仅 stock_posts 表，不含逸修个人内容
  · 增量分析    — 缓存 LLM 结果，每次仅分析新增帖子
  · 数据库直读  — 仪表盘生成时直接从两个本地 DB 加载数据
  · 自包含输出  — HTML 内嵌全部 JSON 数据，单文件即可分享
"""
    print(textwrap.dedent(doc))
    print("  " + "─" * 60)
    input("\n  按 Enter 返回主菜单...")


# ============================================================
# 交互主菜单
# ============================================================
def main():
    stock_code = '02400'

    while True:
        clear_screen()
        print_header()
        status = get_status()
        print_status(status)

        print(f"""
  {'─' * 42}
  操作菜单
  {'─' * 42}

    [1]  运行情绪分析 (仅分析新增帖子)
    [2]  生成仪表盘 (从本地 DB 加载)
    [3]  一键完整流水线 (分析 + 仪表盘)
    [4]  在浏览器打开最新仪表盘
    [5]  流水线原理说明
    [0]  退出
""")
        choice = input("  请输入选项 [0-5]: ").strip()

        if choice == '1':
            run_analysis(stock_code)
            input("\n  按 Enter 返回主菜单...")

        elif choice == '2':
            run_dashboard(stock_code)
            input("\n  按 Enter 返回主菜单...")

        elif choice == '3':
            print()
            if run_analysis(stock_code):
                print()
                run_dashboard(stock_code)
            input("\n  按 Enter 返回主菜单...")

        elif choice == '4':
            open_dashboard()
            input("\n  按 Enter 返回主菜单...")

        elif choice == '5':
            show_documentation()

        elif choice == '0':
            print("\n  再见!\n")
            break

        else:
            print("  无效选项，请重新输入")
            input("  按 Enter 继续...")


if __name__ == '__main__':
    main()
