# Risk Assessment and Portfolio Optimization (Python)

Market risk measurement and portfolio optimization on a 10-stock US equity universe (AAPL, AMZN, JPM, XOM, WMT, PFE, T, MA, GE, SBUX), June 2006 to April 2018.

## What the project does
1. **Risk assessment:** 1-day VaR and CVaR (95% / 99%) using historical simulation, normal, Student-t, Monte Carlo and Cornish-Fisher methods, plus a rolling VaR backtest (Kupiec and Christoffersen tests).
2. **Portfolio construction:** equal weight, minimum variance, maximum Sharpe, risk parity and minimum CVaR (Rockafellar-Uryasev linear program), with Ledoit-Wolf covariance shrinkage, long-only weights and a 35% position cap. Efficient frontier included.
3. **Evaluation:** out-of-sample test (weights from 2006-2014 applied to 2015-2018) and a walk-forward backtest with annual re-optimization and 10 bps transaction costs.
4. **Scenario analysis:** historical stress replays (2007-09 crisis, 2011, 2015-16, Feb 2018), Monte Carlo market regimes (bull, base, bear, crisis) with fat-tailed returns, and market-beta shocks.

## Key findings
- A normal model understates 99% VaR by roughly 17-23% versus fat-tailed estimates; rolling VaR had about twice the expected exceptions, clustered in 2008.
- Max Sharpe performed best out-of-sample but was highly concentrated (AAPL and MA about 70% of weight) and the worst portfolio in the 2015-16 sell-off.
- Min variance and Min CVaR did not reduce out-of-sample volatility or drawdown relative to equal weight in the 2015-2018 test.
- In the walk-forward test, equal weight was hard to beat after costs; Min CVaR gave the shallowest drawdown.
- Even low-risk portfolios have a high probability of a large loss in a simulated crisis regime, because diversification within equities is limited.

## Files
- `risk_portfolio_analysis.py`: complete analysis (creates all figures and `results.json`)
- `prices.csv`: daily adjusted prices (from the PyPortfolioOpt repository's sample data)
- `Risk_Assessment_Portfolio_Optimization_Report.docx`: full written report
- `requirements.txt`

## Run
```bash
pip install -r requirements.txt
python risk_portfolio_analysis.py
```

## Limitations
The universe was chosen from well-known survivors, so results are subject to selection and survivorship bias. Historical means are noisy, the sample is one market cycle, and the scenario parameters are illustrative. Educational use only; this is not financial advice.
