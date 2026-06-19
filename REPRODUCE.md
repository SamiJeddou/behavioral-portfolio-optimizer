# Reproducing the paper's results

Every table and figure in *Beyond Mean-Variance* (v2.6) regenerates from this repo and **public Yahoo
Finance data** — no API keys, no proprietary datasets. Each command below is self-contained.

## 1. One-time setup

```bash
git clone https://github.com/SamiJeddou/behavioral-portfolio-optimizer
cd behavioral-portfolio-optimizer
python -m pip install -r requirements.txt        # numpy, pandas, scipy, yfinance, streamlit, ...
```

The rolling studies fetch prices live from Yahoo Finance. Run a command, and it prints the table
numbers and writes a per-window CSV.

> Universe shorthands used below
> - **EQ24** (large-cap equity book): `AAPL MSFT NVDA GOOGL AMZN JPM XOM JNJ PG KO WMT HD UNH CVX PFE MRK INTC CSCO ORCL PEP MCD IBM DIS BAC`
> - **BAL28** (balanced reference book): EQ24 `+ TLT IEF LQD GLD`

## 2. Commands → paper artifacts

**Rolling multi-window / multi-universe validation (§12, breach-rate figure)**
```bash
python rolling_backtest.py --tickers AAPL MSFT JPM       --benchmark SPY --start 2005-01-01 --out rb_AAPL_MSFT_JPM.csv
python rolling_backtest.py --tickers XLK XLF XLE         --benchmark SPY --start 2005-01-01 --out rb_XLK_XLF_XLE.csv
python rolling_backtest.py --tickers QQQ TLT GLD         --benchmark SPY --start 2005-01-01 --out rb_QQQ_TLT_GLD.csv
python rolling_backtest.py --tickers KO JNJ PG           --benchmark SPY --start 2005-01-01 --out rb_KO_JNJ_PG.csv
```

**Balanced headline + transaction costs (reference book)**
```bash
python rolling_backtest.py --engine scenario --wmax 0.20 --start 2005-01-01 --benchmark SPY --tc-bps 10 --out rb_bal_tc.csv --tickers <BAL28>
```

**Cap-sensitivity table (`tab:capsens`, EQ24 book)**
```bash
python rolling_backtest.py --engine scenario --wmax 1.0  --start 2005-01-01 --benchmark SPY --tc-bps 10 --out rb_N24_nocap.csv  --tickers <EQ24>
python rolling_backtest.py --engine scenario --wmax 0.33 --start 2005-01-01 --benchmark SPY --tc-bps 10 --out rb_N24_cap33.csv --tickers <EQ24>
python rolling_backtest.py --engine scenario --wmax 0.10 --start 2005-01-01 --benchmark SPY --tc-bps 10 --out rb_N24_cap10.csv --tickers <EQ24>
```

**Floor calibration (`tab:floorcal`)**
```bash
python floor_calibration.py --benchmark SPY --start 2005-01-01 --wmax 0.20
```

**Cost-of-control / hierarchical hybrid three-way (`tab:levers`, two universes)**
```bash
python hybrid_backtest.py --benchmark SPY --start 2005-01-01 --wmax 0.20 --resolution standard --out hybrid_threeway.csv \
  --buckets "EQ:AAPL,MSFT,NVDA,GOOGL,AMZN,JPM,XOM,JNJ,PG,KO,WMT,HD;RATES:TLT,IEF,LQD;GOLD:GLD;ALT:VNQ,XLE"
python hybrid_backtest.py --benchmark SPY --start 2005-01-01 --wmax 0.20 --resolution standard --out hybrid_threeway_u2.csv \
  --buckets "EQ:GE,F,T,VZ,CAT,MMM,BA,HON,UNP,LMT,GD,NOC;RATES:AGG,SHY,TIP;GOLD:GLD;ALT:IYR,EEM"
```

**Naive 1/N benchmarks (DeMiguel comparison)**
```bash
python naive_benchmarks.py --benchmark SPY --start 2005-01-01 --out naive_u1.csv \
  --buckets "EQ:AAPL,MSFT,NVDA,GOOGL,AMZN,JPM,XOM,JNJ,PG,KO,WMT,HD;RATES:TLT,IEF,LQD;GOLD:GLD;ALT:VNQ,XLE"
python naive_benchmarks.py --benchmark SPY --start 2005-01-01 --out naive_u2.csv \
  --buckets "EQ:GE,F,T,VZ,CAT,MMM,BA,HON,UNP,LMT,GD,NOC;RATES:AGG,SHY,TIP;GOLD:GLD;ALT:IYR,EEM"
```

**Scenario-generation comparison (`tab:scengen`)**
```bash
python scengen_experiment.py --benchmark SPY --start 2005-01-01 --wmax 0.20
```

**Convex overlay + implied-volatility (VIX) robustness (§ Convexity, § Implied-volatility robustness)**
```bash
# realized-vol + skew proxy
python rolling_backtest.py --engine scenario --wmax 0.20 --start 2005-01-01 --benchmark SPY --hedge put --hedge-strike 0.90 --hedge-frac 0.05 --hedge-vol-add 0.05 --tc-bps 10 --out rb_bal_put.csv --tickers <BAL28>
# priced at market-implied vol (VIX at entry) + skew premium
python rolling_backtest.py --engine scenario --wmax 0.20 --start 2005-01-01 --benchmark SPY --hedge put --hedge-strike 0.90 --hedge-frac 0.05 --hedge-vol-source vix --hedge-vol-add 0.03 --tc-bps 10 --out rb_bal_put_vix.csv --tickers <BAL28>
```

**International out-of-sample validation (`tab:intl`; 225-window pooled headline)**
```bash
python rolling_backtest.py --engine scenario --wmax 0.20 --start 2005-01-01 --benchmark EFA --tc-bps 10 --out rb_intl_dev.csv --tickers EFA EWJ EWG EWU EWQ EWL E