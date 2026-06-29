"""
================================================================================
  大宗商品期货趋势跟踪策略 — 实盘级系统
================================================================================
设计原则：
  1. 零前视偏差（Zero Look-ahead Bias）：所有指标严格使用已收盘数据
  2. 互斥信号（Exclusive Signals）：做多/做空条件不可同时触发
  3. 参数约束（Constrained Search）：贝叶斯优化内置 ma_fast < ma_slow 约束
  4. 正确止损（ATR-based Stop）：实际使用 ATR 乘数控制回撤
  5. Block Bootstrap 显著性检验：保留时间序列自相关结构
  6. 完整日志（Structured Logging）：每步操作可追溯
  7. 配置驱动（Config-driven）：所有超参在顶部集中管理，禁止魔法数字散落
  8. 类型完整（Full Type Hints）：所有公共接口均有类型标注

依赖安装：
  pip install yfinance bayesian-optimization pandas numpy scipy

使用方法：
  python commodity_trend_pro.py              # 完整流程（优化 + 回测 + 建议）
  python commodity_trend_pro.py --no-optim  # 使用默认参数快速回测
================================================================================
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from bayes_opt import BayesianOptimization
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. 日志配置
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("strategy.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("CommodityTrend")

# ==============================================================================
# 1. 全局配置（唯一真值来源，禁止在代码其他地方硬编码数字）
# ==============================================================================

@dataclass
class Config:
    # ── 数据 ──────────────────────────────────────────────────────────────────
    train_start: str = "2000-01-01"
    train_end: str   = "2020-01-01"
    # test_end 在运行时动态设为今天

    cache_dir: str   = "data_cache"

    # 商品期货代码 → 中文名
    commodities: dict[str, str] = field(default_factory=lambda: {
        "GC=F": "黄金",
        "SI=F": "白银",
        "CL=F": "原油",
        "HG=F": "铜",
        "ZC=F": "玉米",
        "ZW=F": "小麦",
        "ZS=F": "大豆",
        "NG=F": "天然气",
        "SB=F": "糖",
        "KC=F": "咖啡",
    })

    # ── 指标默认值（贝叶斯优化未启用时使用）────────────────────────────────
    ma_fast:      int   = 20
    ma_slow:      int   = 60
    mom_lookback: int   = 60
    rsi_period:   int   = 14
    atr_period:   int   = 20
    atr_stop_mul: float = 2.5   # ATR 止损乘数
    entry_mom_th: float = 0.02  # 动量入场阈值（2%）
    use_ma_filter:  bool = True
    use_mom_filter: bool = True
    use_rsi_filter: bool = True
    use_atr_stop:   bool = True

    # ── 贝叶斯优化 ────────────────────────────────────────────────────────────
    bayes_init_points: int = 15
    bayes_n_iter:      int = 60
    cv_n_splits:       int = 5   # 时间序列交叉验证折数

    # ── 回测 ──────────────────────────────────────────────────────────────────
    initial_capital:  float = 1_000_000.0
    risk_free_annual: float = 0.04   # 无风险年化利率
    trading_days:     int   = 252

    # ── 蒙特卡洛（Block Bootstrap）────────────────────────────────────────────
    mc_n_simulations: int = 2000
    mc_block_size:    int = 20    # 块大小（交易日）

    # ── 显著性阈值 ────────────────────────────────────────────────────────────
    significance_level: float = 0.05


CFG = Config()

# ==============================================================================
# 2. 数据获取与缓存
# ==============================================================================

Path(CFG.cache_dir).mkdir(parents=True, exist_ok=True)


def _cache_path(symbol: str) -> Path:
    safe = symbol.replace("=", "_").replace("/", "_")
    return Path(CFG.cache_dir) / f"{safe}.pkl"


def fetch_ohlcv(
    symbol: str,
    start: str,
    end: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    下载并缓存 OHLCV 日线数据。
    返回列：open, high, low, close, volume（小写）。
    索引：tz-naive DatetimeIndex。

    缓存策略：
      - 若缓存存在且最新日期 >= end，直接返回
      - 否则重新下载全量数据并更新缓存
    """
    cache = _cache_path(symbol)

    if not force_refresh and cache.exists():
        with open(cache, "rb") as f:
            df: pd.DataFrame = pickle.load(f)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        end_dt = pd.to_datetime(end).tz_localize(None)
        if df.index[-1] >= end_dt - timedelta(days=3):  # 允许3天宽限（节假日）
            log.debug("缓存命中: %s (%d 行)", symbol, len(df))
            return df

    log.info("下载数据: %s  %s ~ %s", symbol, start, end)
    raw = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)

    if raw.empty:
        raise ValueError(f"yfinance 未返回任何数据: {symbol}")

    # 展平多级列（yfinance >= 0.2 有时返回 MultiIndex）
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
    df = raw[needed].copy()
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.sort_index(inplace=True)
    df.dropna(subset=["close"], inplace=True)

    with open(cache, "wb") as f:
        pickle.dump(df, f)

    log.info("  → 已缓存 %d 行，最新: %s", len(df), df.index[-1].date())
    return df


def load_all(start: str, end: str) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for symbol, name in CFG.commodities.items():
        try:
            df = fetch_ohlcv(symbol, start, end)
            if len(df) >= CFG.ma_slow + CFG.mom_lookback + 10:
                data[symbol] = df
            else:
                log.warning("跳过 %s(%s)：数据行数不足 (%d)", name, symbol, len(df))
        except Exception as exc:
            log.error("获取 %s(%s) 失败: %s", name, symbol, exc)
    log.info("成功加载 %d / %d 个品种", len(data), len(CFG.commodities))
    return data


# ==============================================================================
# 3. 技术指标（严格无前视偏差）
# ==============================================================================

def sma(series: pd.Series, period: int) -> pd.Series:
    """简单移动平均，预热期内为 NaN（min_periods=period）"""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """指数移动平均，adjust=False 与实盘保持一致"""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder RSI（使用 ewm 实现 Wilder 平滑，min_periods=period 避免前视）。
    与主流交易平台（TradingView / MT5）行为一致。
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder 平滑 = ewm(alpha=1/period)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Average True Range（Wilder 平滑），需要 high/low/close 列。
    回退策略：若缺少 high/low，以 close 的绝对变化作为 TR。
    """
    if "high" in df.columns and "low" in df.columns:
        h = df["high"]
        lo = df["low"]
    else:
        h = df["close"]
        lo = df["close"]
    c = df["close"]
    prev_c = c.shift(1)
    tr = pd.concat(
        [h - lo, (h - prev_c).abs(), (lo - prev_c).abs()], axis=1
    ).max(axis=1)
    alpha = 1.0 / period
    return tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal_period: int = 9
         ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """返回 (macd_line, signal_line, histogram)"""
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    sig_line = macd_line.ewm(span=signal_period, adjust=False,
                              min_periods=signal_period).mean()
    return macd_line, sig_line, macd_line - sig_line


# ==============================================================================
# 4. 策略参数（数据类）
# ==============================================================================

@dataclass
class StrategyParams:
    ma_fast:      int   = CFG.ma_fast
    ma_slow:      int   = CFG.ma_slow
    mom_lookback: int   = CFG.mom_lookback
    rsi_period:   int   = CFG.rsi_period
    atr_period:   int   = CFG.atr_period
    atr_stop_mul: float = CFG.atr_stop_mul
    entry_mom_th: float = CFG.entry_mom_th
    use_ma_filter:  bool = CFG.use_ma_filter
    use_mom_filter: bool = CFG.use_mom_filter
    use_rsi_filter: bool = CFG.use_rsi_filter
    use_atr_stop:   bool = CFG.use_atr_stop

    def validate(self) -> None:
        """参数合法性校验，防止贝叶斯优化传入非法组合"""
        if self.ma_fast >= self.ma_slow:
            raise ValueError(
                f"ma_fast({self.ma_fast}) 必须 < ma_slow({self.ma_slow})"
            )
        if self.ma_fast < 2 or self.ma_slow < 3:
            raise ValueError("MA 周期过短")
        if not (0 < self.entry_mom_th < 1):
            raise ValueError(f"entry_mom_th={self.entry_mom_th} 超出合理范围")
        if self.atr_stop_mul <= 0:
            raise ValueError("atr_stop_mul 必须 > 0")
        # 至少启用一个过滤器，否则策略退化为无信号/满仓
        if not (self.use_ma_filter or self.use_mom_filter or self.use_rsi_filter):
            raise ValueError("至少需要启用一个过滤器（MA / MOM / RSI）")

    def to_dict(self) -> dict:
        return asdict(self)


# ==============================================================================
# 5. 信号生成（互斥、含 ATR 止损）
# ==============================================================================

def generate_signals(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    """
    生成交易信号并应用 ATR 动态止损。

    信号规则（互斥，优先多头）：
      做多条件（AND）：
        - [use_ma_filter]  close > sma_slow  AND  sma_fast > sma_slow
        - [use_mom_filter] 过去 mom_lookback 日收益率 > entry_mom_th
        - [use_rsi_filter] RSI > 50（趋势动量确认）
      做空条件（AND，在不满足做多时判断）：
        - [use_ma_filter]  close < sma_slow  AND  sma_fast < sma_slow
        - [use_mom_filter] 过去 mom_lookback 日收益率 < -entry_mom_th
        - [use_rsi_filter] RSI < 50

    ATR 止损：
      持多时，若 close 跌破 entry_price - atr_stop_mul * ATR，平仓（→ 0）。
      持空时，若 close 涨破 entry_price + atr_stop_mul * ATR，平仓（→ 0）。

    返回含以下新列的 DataFrame：
      signal_raw  — 未经止损的原始信号（1 / -1 / 0）
      signal      — 经过止损后的最终信号（1 / -1 / 0）
      atr_val     — ATR 数值（供调试）
    """
    p.validate()
    d = df.copy()

    # ── 指标 ──────────────────────────────────────────────────────────────────
    d["sma_fast"] = sma(d["close"], p.ma_fast)
    d["sma_slow"] = sma(d["close"], p.ma_slow)
    d["mom"]      = d["close"].pct_change(periods=p.mom_lookback)
    d["rsi_val"]  = rsi(d["close"], p.rsi_period)
    d["atr_val"]  = atr(d, p.atr_period)

    # ── 必须至少启用一个过滤器，否则策略退化为无脑满仓 ──────────────────────
    # 三个过滤器全关 = 无任何信号依据，直接返回全零信号（"不操作"）
    if not (p.use_ma_filter or p.use_mom_filter or p.use_rsi_filter):
        d["signal_raw"] = 0
        d["signal"]     = 0
        return d

    # ── 原始做多 / 做空条件（互斥：先判多，不满足再判空）────────────────────
    # 初始化为 False，只有开启的过滤器才能激活信号（AND 累积）
    long_cond  = pd.Series(False, index=d.index)
    short_cond = pd.Series(False, index=d.index)

    # 开启的过滤器之间是 AND 关系；先用第一个开启的过滤器初始化，后续追加
    first = True
    if p.use_ma_filter:
        ma_long  = (d["close"] > d["sma_slow"]) & (d["sma_fast"] > d["sma_slow"])
        ma_short = (d["close"] < d["sma_slow"]) & (d["sma_fast"] < d["sma_slow"])
        if first:
            long_cond, short_cond, first = ma_long, ma_short, False
        else:
            long_cond &= ma_long
            short_cond &= ma_short

    if p.use_mom_filter:
        mom_long  = d["mom"] > p.entry_mom_th
        mom_short = d["mom"] < -p.entry_mom_th
        if first:
            long_cond, short_cond, first = mom_long, mom_short, False
        else:
            long_cond &= mom_long
            short_cond &= mom_short

    if p.use_rsi_filter:
        rsi_long  = d["rsi_val"] > 50.0
        rsi_short = d["rsi_val"] < 50.0
        if first:
            long_cond, short_cond = rsi_long, rsi_short
        else:
            long_cond &= rsi_long
            short_cond &= rsi_short

    # 互斥赋值：先多后空
    d["signal_raw"] = 0
    d.loc[long_cond,               "signal_raw"] = 1
    d.loc[short_cond & ~long_cond, "signal_raw"] = -1

    # ── ATR 动态止损（逐行状态机）────────────────────────────────────────────
    if p.use_atr_stop:
        d["signal"] = _apply_atr_stop(
            closes=d["close"].values,
            raw_signals=d["signal_raw"].values,
            atr_vals=d["atr_val"].values,
            multiplier=p.atr_stop_mul,
        )
    else:
        d["signal"] = d["signal_raw"]

    return d


def _apply_atr_stop(
    closes: np.ndarray,
    raw_signals: np.ndarray,
    atr_vals: np.ndarray,
    multiplier: float,
) -> np.ndarray:
    """
    ATR 止损状态机（纯 numpy，O(n) 时间复杂度）。

    状态：
      position    — 当前持仓方向（1 / -1 / 0）
      entry_price — 建仓价格
      stop_price  — 止损价格

    转换规则：
      1. 若 raw_signal 与 position 方向相同或 position == 0：
           若满足止损条件 → position = 0（平仓）
           否则维持
      2. 若 raw_signal 反向（翻转）：直接翻转持仓（含止损重置）
      3. 新建仓时更新 entry_price 和 stop_price
    """
    n = len(closes)
    signals = np.zeros(n, dtype=np.int8)
    position = 0
    entry_price = 0.0
    stop_price  = 0.0

    for i in range(n):
        c   = closes[i]
        raw = int(raw_signals[i])
        av  = atr_vals[i]

        if np.isnan(av) or av <= 0:
            signals[i] = position
            continue

        # ── 检查止损 ──────────────────────────────────────────────────────────
        stopped_out = False
        if position == 1 and c < stop_price:
            position = 0
            stopped_out = True
        elif position == -1 and c > stop_price:
            position = 0
            stopped_out = True

        # ── 根据原始信号更新持仓 ───────────────────────────────────────────
        if raw != 0 and raw != position:
            # 方向变化：建仓或翻仓
            position    = raw
            entry_price = c
            stop_price  = c - multiplier * av if raw == 1 else c + multiplier * av
        elif raw == 0 and not stopped_out:
            # 原始信号转中性：平仓
            position = 0

        signals[i] = position

    return signals


# ==============================================================================
# 6. 回测引擎
# ==============================================================================

@dataclass
class BacktestResult:
    returns:      pd.Series
    cum_returns:  pd.Series
    sharpe:       float
    sortino:      float
    calmar:       float
    max_drawdown: float
    total_return: float
    win_rate:     float
    avg_win:      float
    avg_loss:     float
    n_trades:     int
    symbol:       str = ""

    def summary(self) -> str:
        lines = [
            f"  Sharpe:       {self.sharpe:>8.3f}",
            f"  Sortino:      {self.sortino:>8.3f}",
            f"  Calmar:       {self.calmar:>8.3f}",
            f"  最大回撤:     {self.max_drawdown:>8.2%}",
            f"  总收益率:     {self.total_return:>8.2%}",
            f"  胜率:         {self.win_rate:>8.2%}",
            f"  平均盈利:     {self.avg_win:>8.4f}",
            f"  平均亏损:     {self.avg_loss:>8.4f}",
            f"  交易次数:     {self.n_trades:>8d}",
        ]
        return "\n".join(lines)


class BacktestEngine:
    """
    向量化回测引擎。

    信号在当日收盘确认，次日开盘（用收盘价近似）执行，
    对应 position = signal.shift(1)，消除当日前视偏差。
    """

    def __init__(self, rf_annual: float = CFG.risk_free_annual,
                 trading_days: int = CFG.trading_days) -> None:
        self.rf_daily    = rf_annual / trading_days
        self.trading_days = trading_days

    def run(self, df: pd.DataFrame, symbol: str = "") -> BacktestResult:
        """df 必须含 close 和 signal 列"""
        d = df.copy()
        # 信号延迟一天执行（收盘信号，次日买入）
        d["position"] = d["signal"].shift(1).fillna(0)
        d["ret_asset"] = d["close"].pct_change().fillna(0)
        d["ret_strat"] = d["position"] * d["ret_asset"]

        returns = d["ret_strat"].dropna()
        cum     = (1.0 + returns).cumprod()

        # ── 绩效指标 ──────────────────────────────────────────────────────────
        mean_r = returns.mean()
        std_r  = returns.std(ddof=1)
        ann    = self.trading_days

        excess = mean_r - self.rf_daily
        sharpe = (excess / std_r * np.sqrt(ann)) if std_r > 1e-12 else 0.0

        downside_r = returns[returns < self.rf_daily]
        downside_std = downside_r.std(ddof=1) if len(downside_r) > 1 else 1e-12
        sortino = (excess / downside_std * np.sqrt(ann)) if downside_std > 1e-12 else 0.0

        roll_max  = cum.cummax()
        drawdown  = (cum - roll_max) / roll_max
        max_dd    = float(drawdown.min())
        ann_ret   = float((1 + mean_r) ** ann - 1)
        calmar    = ann_ret / abs(max_dd) if abs(max_dd) > 1e-12 else 0.0

        # ── 交易统计 ──────────────────────────────────────────────────────────
        pos = d["position"]
        trades = pos.diff().abs() > 0
        n_trades = int(trades.sum())

        daily_pnl = d.loc[returns.index, "ret_strat"]
        wins  = daily_pnl[daily_pnl > 0]
        losses = daily_pnl[daily_pnl < 0]
        win_rate = len(wins) / (len(wins) + len(losses)) if (len(wins) + len(losses)) > 0 else 0.0
        avg_win  = float(wins.mean())  if len(wins)   > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

        return BacktestResult(
            returns=returns,
            cum_returns=cum,
            sharpe=float(sharpe),
            sortino=float(sortino),
            calmar=float(calmar),
            max_drawdown=max_dd,
            total_return=float(cum.iloc[-1] - 1) if len(cum) > 0 else 0.0,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            n_trades=n_trades,
            symbol=symbol,
        )

    def score(self, result: BacktestResult) -> float:
        """
        综合评分（用于优化目标）。

        惩罚项：
          - 过大回撤：超过 20% 开始扣分
          - 交易次数过少：低于 MIN_TRADES 时大幅惩罚，防止"全程持多"退化解
        """
        MIN_TRADES    = 10   # 单折内最少要有 10 次交易
        base          = 0.5 * result.sharpe + 0.5 * result.sortino
        dd_penalty    = max(0.0, abs(result.max_drawdown) - 0.20) * 2.0
        trade_penalty = max(0.0, MIN_TRADES - result.n_trades) * 0.5  # 每少一次扣 0.5 分
        return base - dd_penalty - trade_penalty


# ==============================================================================
# 7. 时间序列交叉验证（Walk-Forward）
# ==============================================================================

def walk_forward_cv(
    df: pd.DataFrame,
    params: StrategyParams,
    engine: BacktestEngine,
    n_splits: int = CFG.cv_n_splits,
) -> float:
    """
    Walk-Forward 交叉验证：在多个不重叠的测试窗口上求平均得分。
    这是防止参数过拟合最关键的一步。
    """
    n = len(df)
    fold_size = n // (n_splits + 1)

    scores: list[float] = []
    for k in range(n_splits):
        test_start = (k + 1) * fold_size
        test_end   = test_start + fold_size
        if test_end > n:
            break
        fold_df = df.iloc[:test_end].copy()  # 只用到当前折结束位置的数据
        try:
            fold_sig = generate_signals(fold_df, params)
            fold_test = fold_sig.iloc[test_start:test_end]
            if len(fold_test) < 20:
                continue
            result = engine.run(fold_test)
            scores.append(engine.score(result))
        except Exception as exc:
            log.debug("CV fold %d 失败: %s", k, exc)
            continue

    return float(np.mean(scores)) if scores else -999.0


# ==============================================================================
# 8. 贝叶斯优化（含参数约束）
# ==============================================================================

class BayesianOptimizer:
    """
    贝叶斯优化寻找最优策略参数。

    关键改进：
    - 搜索空间内置 ma_fast < ma_slow 约束（违反时返回惩罚分 -999）
    - 目标函数为 Walk-Forward CV 平均得分（非全样本 Sharpe）
    - bool 参数通过 > 0.5 阈值离散化
    """

    def __init__(
        self,
        train_data: dict[str, pd.DataFrame],
        engine: BacktestEngine,
        init_points: int = CFG.bayes_init_points,
        n_iter: int = CFG.bayes_n_iter,
    ) -> None:
        self.train_data  = train_data
        self.engine      = engine
        self.init_points = init_points
        self.n_iter      = n_iter

    def _objective(self, **raw: float) -> float:
        # ── 参数解析 ──────────────────────────────────────────────────────────
        ma_fast = int(round(raw["ma_fast"]))
        ma_slow = int(round(raw["ma_slow"]))

        # ── 硬约束：fast < slow ───────────────────────────────────────────────
        if ma_fast >= ma_slow:
            return -999.0

        try:
            p = StrategyParams(
                ma_fast      = ma_fast,
                ma_slow      = ma_slow,
                mom_lookback = int(round(raw["mom_lookback"])),
                rsi_period   = int(round(raw["rsi_period"])),
                atr_period   = int(round(raw["atr_period"])),
                atr_stop_mul = raw["atr_stop_mul"],
                entry_mom_th = raw["entry_mom_th"],
                use_ma_filter  = raw["use_ma_filter"]  > 0.5,
                use_mom_filter = raw["use_mom_filter"] > 0.5,
                use_rsi_filter = raw["use_rsi_filter"] > 0.5,
                use_atr_stop   = True,
            )
            p.validate()
        except ValueError:
            return -999.0

        # ── 跨品种 Walk-Forward 平均得分 ─────────────────────────────────────
        cv_scores: list[float] = []
        for symbol, df in self.train_data.items():
            try:
                score = walk_forward_cv(df, p, self.engine)
                cv_scores.append(score)
            except Exception:
                continue

        return float(np.mean(cv_scores)) if cv_scores else -999.0

    def optimize(self) -> tuple[StrategyParams, float]:
        # 注意：use_*_filter 下限设为 0.0，但 _objective 中已通过 validate()
        # 强制要求至少一个过滤器开启（三者全 < 0.5 时 validate 抛异常返回 -999）
        pbounds = {
            "ma_fast":      (5,   60),
            "ma_slow":      (20, 250),
            "mom_lookback": (10, 120),
            "rsi_period":   (7,   28),
            "atr_period":   (10,  30),
            "atr_stop_mul": (1.0, 4.0),
            "entry_mom_th": (0.005, 0.08),
            "use_ma_filter":  (0.0, 1.0),
            "use_mom_filter": (0.0, 1.0),
            "use_rsi_filter": (0.0, 1.0),
        }

        optimizer = BayesianOptimization(
            f=self._objective,
            pbounds=pbounds,
            random_state=42,
            verbose=0,
        )
        log.info("贝叶斯优化开始 (init=%d, iter=%d)…", self.init_points, self.n_iter)
        optimizer.maximize(init_points=self.init_points, n_iter=self.n_iter)

        best_raw   = optimizer.max["params"]
        best_score = optimizer.max["target"]

        best_params = StrategyParams(
            ma_fast      = int(round(best_raw["ma_fast"])),
            ma_slow      = int(round(best_raw["ma_slow"])),
            mom_lookback = int(round(best_raw["mom_lookback"])),
            rsi_period   = int(round(best_raw["rsi_period"])),
            atr_period   = int(round(best_raw["atr_period"])),
            atr_stop_mul = best_raw["atr_stop_mul"],
            entry_mom_th = best_raw["entry_mom_th"],
            use_ma_filter  = best_raw["use_ma_filter"]  > 0.5,
            use_mom_filter = best_raw["use_mom_filter"] > 0.5,
            use_rsi_filter = best_raw["use_rsi_filter"] > 0.5,
            use_atr_stop   = True,
        )
        log.info("最优参数: %s", best_params.to_dict())
        log.info("CV 得分: %.4f", best_score)
        return best_params, best_score


# ==============================================================================
# 9. Block Bootstrap 显著性检验
# ==============================================================================

class BlockBootstrap:
    """
    Circular Block Bootstrap（循环块自举）。

    与置换检验的区别：保留时间序列的短期自相关结构，
    统计结论更为可靠。

    参考：Politis & Romano (1994)
    """

    def __init__(
        self,
        n_simulations: int = CFG.mc_n_simulations,
        block_size:    int  = CFG.mc_block_size,
        seed:          int  = 42,
    ) -> None:
        self.n_simulations = n_simulations
        self.block_size    = block_size
        self.rng           = np.random.default_rng(seed)

    def _bootstrap_once(self, returns: np.ndarray) -> np.ndarray:
        n = len(returns)
        n_blocks = int(np.ceil(n / self.block_size))
        starts = self.rng.integers(0, n, size=n_blocks)
        blocks = [
            np.take(returns, np.arange(s, s + self.block_size) % n)
            for s in starts
        ]
        return np.concatenate(blocks)[:n]

    def test(
        self,
        strategy_returns: pd.Series,
        actual_sharpe: float,
        actual_sortino: float,
        rf_daily: float = 0.0,
        trading_days: int = CFG.trading_days,
    ) -> dict[str, float]:
        arr = strategy_returns.values
        sim_sharpes  = np.zeros(self.n_simulations)
        sim_sortinos = np.zeros(self.n_simulations)

        for i in range(self.n_simulations):
            sim = self._bootstrap_once(arr)
            m = sim.mean()
            s = sim.std(ddof=1)
            sim_sharpes[i] = (m - rf_daily) / s * np.sqrt(trading_days) if s > 1e-12 else 0.0
            ds = sim[sim < rf_daily].std(ddof=1) if (sim < rf_daily).any() else 1e-12
            sim_sortinos[i] = (m - rf_daily) / ds * np.sqrt(trading_days) if ds > 1e-12 else 0.0

        p_sharpe  = float((sim_sharpes  >= actual_sharpe).mean())
        p_sortino = float((sim_sortinos >= actual_sortino).mean())

        # 正态近似置信区间
        ci_sharpe = norm.interval(
            0.95, loc=sim_sharpes.mean(), scale=sim_sharpes.std(ddof=1)
        )
        ci_sortino = norm.interval(
            0.95, loc=sim_sortinos.mean(), scale=sim_sortinos.std(ddof=1)
        )

        return {
            "sharpe_p_value":     p_sharpe,
            "sortino_p_value":    p_sortino,
            "sim_sharpe_mean":    float(sim_sharpes.mean()),
            "sim_sharpe_ci_low":  float(ci_sharpe[0]),
            "sim_sharpe_ci_high": float(ci_sharpe[1]),
            "sim_sortino_mean":   float(sim_sortinos.mean()),
            "sim_sortino_ci_low": float(ci_sortino[0]),
            "sim_sortino_ci_high":float(ci_sortino[1]),
        }


# ==============================================================================
# 10. 持仓建议生成器
# ==============================================================================

def _signal_label(sig: int) -> str:
    return {1: "📈 做多", -1: "📉 做空", 0: "⏸  观望"}.get(sig, "?")


def _fmt(v: float, decimals: int = 4) -> str:
    """统一格式化价格，NaN 显示 N/A"""
    return f"{v:.{decimals}f}" if not np.isnan(v) else "N/A"


def _price_level_tag(price: float, sma_fast: float, sma_slow: float) -> str:
    """标注当前价格相对均线的位置"""
    above_fast = price > sma_fast if not (np.isnan(sma_fast)) else None
    above_slow = price > sma_slow if not (np.isnan(sma_slow)) else None
    if above_fast is None or above_slow is None:
        return "均线数据不足"
    if above_fast and above_slow:
        return "价格在双均线上方 ✓"
    if not above_fast and not above_slow:
        return "价格在双均线下方 ✗"
    return "价格在均线之间 ~"


def compute_price_levels(
    row: pd.Series,
    sig: int,
    params: StrategyParams,
    rr_ratio: float = 2.0,
    entry_buffer_pct: float = 0.005,
) -> dict:
    """
    计算完整价格区间建议：

    做多：
      止损价   = 当前价 - ATR × atr_stop_mul
      风险距离 = 当前价 - 止损价
      目标价   = 当前价 + 风险距离 × rr_ratio  （默认 1:2 风险收益比）
      建议入场区间 = [当前价 × (1 - buffer), 当前价 × (1 + buffer)]
                    （允许略高或略低于当前价建仓，实盘可挂限价单）

    做空：镜像逻辑。

    观望：仅提供观察区间（快慢均线之间的区域），供关注转折点。
    """
    price = row["close"]
    atr_v = row.get("atr_val", float("nan"))
    sma_f = row.get("sma_fast", float("nan"))
    sma_s = row.get("sma_slow", float("nan"))

    result = {
        "price": price,
        "atr":   atr_v,
        "sma_fast": sma_f,
        "sma_slow": sma_s,
        "stop":     float("nan"),
        "target":   float("nan"),
        "entry_lo": float("nan"),
        "entry_hi": float("nan"),
        "watch_lo": float("nan"),
        "watch_hi": float("nan"),
        "risk_pct": float("nan"),
        "reward_pct": float("nan"),
    }

    if sig == 1 and not np.isnan(atr_v):
        risk_dist  = params.atr_stop_mul * atr_v
        stop       = price - risk_dist
        target     = price + risk_dist * rr_ratio
        result.update({
            "stop":       stop,
            "target":     target,
            "entry_lo":   price * (1 - entry_buffer_pct),
            "entry_hi":   price * (1 + entry_buffer_pct),
            "risk_pct":   -risk_dist / price,
            "reward_pct":  risk_dist * rr_ratio / price,
        })

    elif sig == -1 and not np.isnan(atr_v):
        risk_dist  = params.atr_stop_mul * atr_v
        stop       = price + risk_dist
        target     = price - risk_dist * rr_ratio
        result.update({
            "stop":       stop,
            "target":     target,
            "entry_lo":   price * (1 - entry_buffer_pct),
            "entry_hi":   price * (1 + entry_buffer_pct),
            "risk_pct":   -risk_dist / price,
            "reward_pct":  risk_dist * rr_ratio / price,
        })

    else:
        # 观望：给出均线区间作为关注边界
        lo = min(sma_f, sma_s) if not (np.isnan(sma_f) or np.isnan(sma_s)) else float("nan")
        hi = max(sma_f, sma_s) if not (np.isnan(sma_f) or np.isnan(sma_s)) else float("nan")
        result.update({"watch_lo": lo, "watch_hi": hi})

    return result


def print_position_advice(
    test_data_with_signals: dict[str, pd.DataFrame],
    params: StrategyParams,
    rr_ratio: float = 2.0,
) -> None:
    """打印每个品种的完整交易价格区间建议"""
    W = 72
    print("\n" + "═" * W)
    print("  当前持仓建议（基于最新收盘数据）")
    print("═" * W)
    print(f"  参数快照: MA({params.ma_fast}/{params.ma_slow}), "
          f"MOM({params.mom_lookback}d, th={params.entry_mom_th:.3f}), "
          f"RSI({params.rsi_period}), ATR({params.atr_period}×{params.atr_stop_mul:.1f})")
    print(f"  风险收益比: 1 : {rr_ratio:.1f}  │  入场缓冲: ±0.5%")
    print("─" * W)

    long_list  = []
    short_list = []
    flat_list  = []

    for symbol, df in test_data_with_signals.items():
        name  = CFG.commodities.get(symbol, symbol)
        row   = df.iloc[-1]
        sig   = int(row["signal"])
        label = _signal_label(sig)
        rsi_v = row.get("rsi_val", float("nan"))
        mom_v = row.get("mom", float("nan"))

        lv = compute_price_levels(row, sig, params, rr_ratio)
        price_tag = _price_level_tag(lv["price"], lv["sma_fast"], lv["sma_slow"])

        # ── 标题行 ────────────────────────────────────────────────────────────
        print(f"\n  {label}  {name} ({symbol})")
        print(f"  {'─' * (W - 2)}")

        # ── 价格信息 ──────────────────────────────────────────────────────────
        d = 4  # 小数位
        print(f"  {'当前价':8s}  {_fmt(lv['price'], d):>12s}    {price_tag}")
        print(f"  {'MA快线':8s}  {_fmt(lv['sma_fast'], d):>12s}    "
              f"MA慢线  {_fmt(lv['sma_slow'], d):>12s}")
        print(f"  {'RSI':8s}  {rsi_v:>12.1f}    动量    {mom_v:>+11.2%}")

        if sig in (1, -1):
            direction = "做多" if sig == 1 else "做空"
            # ── 建议入场区间 ──────────────────────────────────────────────────
            print(f"\n  ┌─ 建议入场区间 ({'限价买入' if sig == 1 else '限价卖出'}) "
                  f"{'─' * 24}┐")
            print(f"  │  入场低点   {_fmt(lv['entry_lo'], d):>14s}"
                  f"    入场高点  {_fmt(lv['entry_hi'], d):>14s} │")
            print(f"  └{'─' * (W - 4)}┘")

            # ── 止损 / 目标 ───────────────────────────────────────────────────
            print(f"  ┌─ {direction}价格区间 {'─' * 31}┐")
            print(f"  │  🛡 止损价   {_fmt(lv['stop'], d):>14s}"
                  f"    风险幅度  {lv['risk_pct']:>+13.2%} │")
            print(f"  │  🎯 目标价   {_fmt(lv['target'], d):>14s}"
                  f"    收益幅度  {lv['reward_pct']:>+13.2%} │")
            print(f"  │  ATR值      {_fmt(lv['atr'], d):>14s}"
                  f"    风险收益比  1 : {rr_ratio:.1f}{'':>8s} │")
            print(f"  └{'─' * (W - 4)}┘")

        else:
            # ── 观望区间 ──────────────────────────────────────────────────────
            print(f"\n  ┌─ 关注区间（均线支撑/压力）{'─' * 28}┐")
            print(f"  │  下轨（快/慢均线低值）  {_fmt(lv['watch_lo'], d):>14s}"
                  f"{'':>10s} │")
            print(f"  │  上轨（快/慢均线高值）  {_fmt(lv['watch_hi'], d):>14s}"
                  f"{'':>10s} │")
            print(f"  │  突破上轨且动量转正 → 考虑做多入场{'':>18s} │")
            print(f"  │  跌破下轨且动量转负 → 考虑做空入场{'':>18s} │")
            print(f"  └{'─' * (W - 4)}┘")

        if sig == 1:
            long_list.append(name)
        elif sig == -1:
            short_list.append(name)
        else:
            flat_list.append(name)

    print("\n" + "═" * W)
    print(f"  做多品种 ({len(long_list)}):  {', '.join(long_list) or '无'}")
    print(f"  做空品种 ({len(short_list)}):  {', '.join(short_list) or '无'}")
    print(f"  观望品种 ({len(flat_list)}):  {', '.join(flat_list) or '无'}")
    print("═" * W)
    print("  ⚠  以上建议仅供参考，不构成投资建议。实盘请结合基本面与风控规则。")
    print("═" * W)


# ==============================================================================
# 11. 主程序
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="大宗商品期货趋势跟踪系统")
    parser.add_argument(
        "--no-optim", action="store_true",
        help="跳过贝叶斯优化，直接使用默认参数回测"
    )
    parser.add_argument(
        "--no-mc", action="store_true",
        help="跳过 Block Bootstrap 显著性检验（节省时间）"
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="强制重新下载数据（忽略缓存）"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    test_end = datetime.today().strftime("%Y-%m-%d")

    print("═" * 70)
    print("  大宗商品期货趋势跟踪策略 — 实盘级系统")
    print("═" * 70)
    print(f"  训练期: {CFG.train_start} ~ {CFG.train_end}")
    print(f"  测试期: {CFG.train_end}  ~ {test_end}")
    print(f"  品种数: {len(CFG.commodities)}")
    print("═" * 70)

    # ── Step 1: 加载数据 ──────────────────────────────────────────────────────
    log.info("Step 1/5  下载/加载数据")
    all_data = load_all(start=CFG.train_start, end=test_end)
    if len(all_data) < 2:
        log.error("有效品种数不足 2，程序退出")
        sys.exit(1)

    def split(data: dict[str, pd.DataFrame], boundary: str):
        train, test = {}, {}
        for sym, df in data.items():
            bd = pd.to_datetime(boundary)
            tr = df[df.index < bd]
            te = df[df.index >= bd]
            if len(tr) >= CFG.ma_slow + CFG.mom_lookback and len(te) >= 20:
                train[sym] = tr
                test[sym]  = te
        return train, test

    train_data, test_data = split(all_data, CFG.train_end)
    log.info("训练集品种: %d  测试集品种: %d", len(train_data), len(test_data))

    # ── Step 2: 参数寻优 ──────────────────────────────────────────────────────
    engine = BacktestEngine()

    if args.no_optim:
        log.info("Step 2/5  使用默认参数（跳过优化）")
        best_params = StrategyParams()
        best_score  = 0.0
    else:
        log.info("Step 2/5  贝叶斯优化（Walk-Forward CV）")
        optimizer = BayesianOptimizer(train_data, engine)
        best_params, best_score = optimizer.optimize()

    print("\n最优策略参数:")
    for k, v in best_params.to_dict().items():
        print(f"  {k:<18s} = {v}")
    print(f"  Walk-Forward CV 得分: {best_score:.4f}")

    # ── Step 3: 测试集回测 ────────────────────────────────────────────────────
    log.info("Step 3/5  测试集回测")
    print("\n" + "═" * 70)
    print("  测试集回测结果（样本外）")
    print("═" * 70)

    results: dict[str, BacktestResult] = {}
    test_data_with_signals: dict[str, pd.DataFrame] = {}

    for symbol, df in test_data.items():
        name = CFG.commodities.get(symbol, symbol)
        try:
            df_sig = generate_signals(df, best_params)
            test_data_with_signals[symbol] = df_sig
            result = engine.run(df_sig, symbol=symbol)
            results[symbol] = result
            print(f"\n  {name} ({symbol}):")
            print(result.summary())
        except Exception as exc:
            log.error("回测失败 %s: %s", symbol, exc)

    if results:
        all_metrics = {
            k: [getattr(r, k) for r in results.values()]
            for k in ["sharpe", "sortino", "calmar", "max_drawdown", "total_return"]
        }
        print("\n" + "─" * 70)
        print(f"  {'跨品种平均':}")
        for metric, vals in all_metrics.items():
            mean_v = np.mean(vals)
            fmt = ".2%" if "drawdown" in metric or "return" in metric else ".3f"
            print(f"    {metric:<15s} {mean_v:{fmt}}")

    # ── Step 4: Block Bootstrap 检验 ─────────────────────────────────────────
    if not args.no_mc and results:
        log.info("Step 4/5  Block Bootstrap 显著性检验 (n=%d)", CFG.mc_n_simulations)
        print("\n" + "═" * 70)
        print("  Block Bootstrap 显著性检验")
        print("═" * 70)

        all_ret = pd.concat([r.returns for r in results.values()]).reset_index(drop=True)
        ann_ret_mean = all_ret.mean()
        ann_std      = all_ret.std(ddof=1)
        actual_sharpe  = (ann_ret_mean - engine.rf_daily) / ann_std * np.sqrt(engine.trading_days) \
                         if ann_std > 1e-12 else 0.0
        ds = all_ret[all_ret < engine.rf_daily].std(ddof=1)
        actual_sortino = (ann_ret_mean - engine.rf_daily) / ds * np.sqrt(engine.trading_days) \
                         if ds > 1e-12 else 0.0

        bb = BlockBootstrap()
        stats = bb.test(all_ret, actual_sharpe, actual_sortino,
                        rf_daily=engine.rf_daily, trading_days=engine.trading_days)

        print(f"  实际 Sharpe:   {actual_sharpe:.3f}")
        print(f"  实际 Sortino:  {actual_sortino:.3f}")
        print(f"  Sharpe  p值:   {stats['sharpe_p_value']:.4f}  "
              f"Bootstrap均值: {stats['sim_sharpe_mean']:.3f}  "
              f"95%CI: [{stats['sim_sharpe_ci_low']:.3f}, {stats['sim_sharpe_ci_high']:.3f}]")
        print(f"  Sortino p值:   {stats['sortino_p_value']:.4f}  "
              f"Bootstrap均值: {stats['sim_sortino_mean']:.3f}  "
              f"95%CI: [{stats['sim_sortino_ci_low']:.3f}, {stats['sim_sortino_ci_high']:.3f}]")

        alpha = CFG.significance_level
        sharpe_sig  = stats["sharpe_p_value"]  < alpha
        sortino_sig = stats["sortino_p_value"] < alpha
        print(f"\n  Sharpe  显著性 (p<{alpha}): {'✓ 显著优于随机' if sharpe_sig  else '✗ 不显著'}")
        print(f"  Sortino 显著性 (p<{alpha}): {'✓ 显著优于随机' if sortino_sig else '✗ 不显著'}")
    else:
        log.info("Step 4/5  跳过 Block Bootstrap 检验")

    # ── Step 5: 持仓建议 ──────────────────────────────────────────────────────
    log.info("Step 5/5  生成持仓建议")
    if test_data_with_signals:
        print_position_advice(test_data_with_signals, best_params)
    else:
        log.warning("无有效测试数据，无法生成持仓建议")

    print("\n  分析完成！详细日志见 strategy.log")
    print("═" * 70)


if __name__ == "__main__":
    main()
