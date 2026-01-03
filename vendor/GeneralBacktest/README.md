## GeneralBacktest - 通用量化策略回测框架

一个以“权重→交易→净值”为核心、专注于多资产组合回测的轻量级工具库，适合在本地研究环境中快速验证 ETF / 股票 / 自定义资产组合策略。

> 本仓库主要作为你自己研究使用的代码库，而不是对外发布的 Python 包，因此 README 更偏向“怎么用、注意什么”，而不是完整产品文档。

---

## 1. 功能概览

- 任意调仓频率：直接喂给框架一张“调仓权重表”，不用关心具体频率。
- 真实交易细节：
    - 调仓阈值（`rebalance_threshold`）：小于阈值的仓位变化不触发真实交易；
    - 买卖分开计费（`transaction_cost=[buy, sell]`）；
    - 滑点（`slippage`）：买入抬价、卖出压价；
    - 调仓日拆分为 Kept / Sold / Bought 三部分，区分盘中和隔夜收益。
- 大部分计算向量化实现，支持几千标的、数年日频数据。
- 内置 10+ 常用指标：累计收益、年化收益、波动率、最大回撤、夏普、索提诺、卡玛、VaR/CVaR、信息比率、换手率等。
- 提供一套“够用”的可视化：净值 + 回撤、策略 vs 基准、超额收益、换手率、持仓热力图、综合 Dashboard 等。
- 直接对接 ClickHouse（通过 `quantchdb`）获取 ETF / 股票日线数据，一行代码跑全市场组合回测。

---

## 2. 环境与依赖

最小依赖：

```bash
pip install numpy pandas matplotlib
```

若使用数据库接口（`run_backtest_ETF` / `run_backtest_stock`）：

```bash
pip install quantchdb
```

如需导出 Excel：

```bash
pip install openpyxl
```

> 代码默认工作在日频回测场景；分钟级回测暂未适配。

---

## 3. 目录结构（当前版本）

```text
GeneralBacktest/
├── __init__.py      # 导出 GeneralBacktest
├── backtest.py      # 回测主类及全部绘图接口
├── utils.py         # 各类指标计算与辅助函数
├── db_config.py     # ClickHouse 连接配置（ETF / 股票库）
├── test_plotting_redesign.py  # 绘图相关的简单测试 / 示例
└── README.md
```

数据库连接配置在 `db_config.py` 中，请根据自己环境修改：

```python
Agent_db_config = {"host": "...", "port": 20108, ...}
Stock_db_config = {"host": "...", "port": 20108, ...}
```

> 该文件通常不会提交到公共仓库，如需开源请自行脱敏处理。

---

## 4. 最常用的用法

### 4.1 本地 DataFrame 回测

你已经有了两张表：

- `weights_df`：组合在各个调仓日的目标权重；
- `price_df`：资产的日线行情数据（含复权因子）。

```python
from backtest import GeneralBacktest

bt = GeneralBacktest(start_date="2023-01-01", end_date="2023-12-31")

results = bt.run_backtest(
        weights_data=weights_df,
        price_data=price_df,
        buy_price="open",          # 买入价字段
        sell_price="close",        # 卖出价字段
        adj_factor_col="adj_factor",   # 复权因子字段
        close_price_col="close",       # 收盘价字段（用于计算日收益）
        rebalance_threshold=0.005,      # 绝对调仓阈值，5bps
        transaction_cost=[0.001, 0.001],# 买卖各 10bp
        initial_capital=1.0,
        slippage=0.0005                # 5bp 滑点
)

bt.print_metrics()      # 终端打印指标
bt.plot_all()           # 生成综合报告（Dashboard）
```

### 4.2 直接从数据库读取 ETF / 股票

如果已配置好 `db_config.py` 且安装了 `quantchdb`，可以只准备权重表，让框架自己查行情：

```python
from backtest import GeneralBacktest

bt = GeneralBacktest("2023-01-01", "2023-12-31")

results = bt.run_backtest_ETF(
        weights_data=etf_weights_df,      # 列: ['date', 'code', 'weight']
        buy_price="OpenPrice",
        sell_price="ClosePrice",
        transaction_cost=[0.001, 0.001],
        rebalance_threshold=0.01,
        slippage=0.0,
        benchmark_weights=None            # 可选：另一张权重表
)

bt.print_metrics()
bt.plot_all()
```

`run_backtest_stock` 的用法类似，只是行情表字段名换成了 `open / close / adj_factor` 等。

---

## 5. 数据格式要求（框架假定）

### 5.1 权重表 `weights_data`

长表形式，至少三列：

| date       | code      | weight |
|------------|-----------|--------|
| 2023-01-03 | 000001.SZ | 0.25   |
| 2023-01-03 | 600000.SH | 0.30   |
| 2023-01-03 | 600519.SH | 0.45   |

- `date`：调仓日期，支持任意频率（月度、周度、信号驱动等）。
- `code`：资产代码。
- `weight`：目标权重（不必严格归一化，框架会在每个调仓日按行归一）。

### 5.2 价格表 `price_data`

典型 OHLC + 复权因子：

| date       | code      | open  | high  | low   | close | adj_factor |
|------------|-----------|-------|-------|-------|-------|------------|
| 2023-01-03 | 000001.SZ | 10.50 | 10.80 | 10.40 | 10.70 | 1.0        |

- `adj_factor` 为累计复权因子，框架内部会生成：
    - `adj_buy_price = buy_price * adj_factor`
    - `adj_sell_price = sell_price * adj_factor`
    - `adj_close_price = close_price * adj_factor`

如果你在数据库侧已经是复权价，也可以把 `adj_factor` 固定为 1.0。

---

## 6. 结果获取与后处理

回测完成后，`GeneralBacktest` 实例上会挂载几类数据：

- `bt.daily_nav`：`pd.Series`，净值曲线；
- `bt.metrics`：`dict`，全部指标；也可用 `bt.get_metrics()` 转成 DataFrame；
- `bt.daily_positions`：长表，列为 `['date', 'asset', 'weight']`；
- `bt.trade_records`：每个调仓日的交易拆分信息（kept / sold / bought 的收益贡献、佣金等）；
- `bt.turnover_records`：每个调仓日的换手率。

常用便捷方法：

```python
metrics_df = bt.get_metrics()
pos_long = bt.get_daily_positions()
pos_matrix = bt.get_position_matrix()      # index: date, columns: asset
pos_change = bt.get_position_changes()     # 每日权重变化，正为加仓、负为减仓
```

导出到 Excel 的简单范式：

```python
import pandas as pd

with pd.ExcelWriter("backtest_results.xlsx") as writer:
        bt.get_metrics().to_excel(writer, sheet_name="metrics")
        bt.get_position_matrix().to_excel(writer, sheet_name="weights")
        bt.get_position_changes().to_excel(writer, sheet_name="delta_weights")
        bt.trade_records.to_excel(writer, sheet_name="trades", index=False)
        bt.turnover_records.to_excel(writer, sheet_name="turnover", index=False)
```

---

## 7. 一些使用上的经验提示

1. **数据质量第一**
     - 价格缺失、错误复权会直接反映在净值上；
     - 尽量保证同一资产的行情是连续的，停牌日建议前值填充。
2. **合理使用调仓阈值与交易成本**
     - 高频策略请务必设置真实水平的手续费 & 滑点；
     - `rebalance_threshold` 可以有效抑制权重抖动带来的无意义换手。
3. **基准回测**
     - 若传入 `benchmark_weights`，所有相对指标（超额收益、信息比率、跟踪误差）会自动计算。
4. **绘图只是辅助**
     - `plot_all()` 是综合报告，日常调试可只看 `plot_nav_curve()` / `plot_drawdown()` 提高效率。

---

## 8. 协议与开放性

- 当前仓库默认 MIT License（如有变更，请在 `LICENSE` 中更新）；
- 主要面向个人/团队内部研究使用，如需对外开源建议：
    - 清理 / 抽象数据库连接配置；
    - 根据实际 API 调整公开接口和文档粒度。

如果后续你在仓库里增加了新的模块（例如信号生成、组合优化、风控等），可以在本 README 的“目录结构”和“最常用用法”中补一段简短说明即可。
