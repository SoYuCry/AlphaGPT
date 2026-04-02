# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaGPT is a crypto quantitative trading system that automatically generates factor formulas using a Transformer model, backtests them, and executes trades on Solana (focused on meme tokens). The core idea: generate stack-based formula expressions, score them via backtesting, and train the generator with policy gradients.

## Setup & Commands

```bash
# Python 3.10+ required (3.11 recommended)
pip install -r requirements.txt
pip install -r requirements-optional.txt  # only for times.py, lord/experiment.py

# Run data pipeline (async, requires Birdeye API key + Postgres)
python -m data_pipeline.run_pipeline

# Train the formula generator
python -m model_core.engine

# Start live trading (requires best_meme_strategy.json from training + Solana private key)
python -m strategy_manager.runner

# Launch dashboard
streamlit run dashboard/app.py
```

No test framework is configured. `test.py` in the root is an ad-hoc script, not a test suite.

## Architecture

The system has four pipeline stages that run sequentially:

1. **data_pipeline/** - Fetches token/OHLCV data from Birdeye/DexScreener APIs into PostgreSQL/TimescaleDB
2. **model_core/** - Trains a Transformer to generate factor formulas, evaluated via backtesting
3. **strategy_manager/** - Live loop that loads the best trained formula, scores tokens, and manages positions
4. **execution/** - Solana RPC + Jupiter aggregator for swap execution

### model_core (the core ML system)

- `alphagpt.py` - Transformer with looped layers, RMSNorm, SwiGLU FFN, multi-task pooling head (MTPHead), and LoRD (Low-Rank Decay) regularization via Newton-Schulz iteration
- `engine.py` (AlphaEngine) - Training loop using policy gradient: generates formulas → backtests → reward-weighted loss
- `vm.py` (StackVM) - Executes generated formulas as stack-based token sequences against feature tensors
- `ops.py` - 12 operators (ADD, SUB, MUL, DIV, NEG, ABS, SIGN, GATE, JUMP, DECAY, DELAY1, MAX3) with JIT-compiled helpers
- `factors.py` - FeatureEngineer (6 features) and AdvancedFactorEngineer (12 features) computing indicators like liquidity score, buy/sell pressure, FOMO acceleration
- `backtest.py` (MemeBacktest) - Scores strategies: `cum_return - 2.0 * big_drawdowns`, median across batch
- `data_loader.py` - Loads from DB, pivots OHLCV by token address, returns feature tensors

### strategy_manager

- `runner.py` (StrategyRunner) - Main live loop: data sync every 15min, signal scan every 60s, buy/sell execution
- `portfolio.py` - Position tracking with moonbag support, persisted to `portfolio_state.json`
- `risk.py` (RiskEngine) - Liquidity checks (min $5k), exit path verification via Jupiter, balance buffer validation
- `config.py` - Max 3 positions, 2.0 SOL entry, -5% stop-loss, 10% TP (sell 50%), trailing stop

### execution

- `trader.py` (SolanaTrader) - Buy/sell via Jupiter quotes, transaction signing
- `jupiter.py` (JupiterAggregator) - Jupiter v6 API client for quotes and swap transactions
- `rpc_handler.py` (QuickNodeClient) - Solana RPC wrapper with retry logic

### Standalone research scripts

- `lord/experiment.py` - LoRD regularization experiments on modular addition (grokking)
- `times.py` - Legacy index-based alpha mining on Chinese stock market via Tushare API

## External Dependencies

Requires running services: **PostgreSQL/TimescaleDB**, **Birdeye API** (key via env), **Solana RPC** (QuickNode), **Jupiter API**. Configuration is spread across per-module `config.py` files. Private keys and API keys are loaded from environment variables / `.env`.

## Key Generated Artifacts

- `best_meme_strategy.json` - Best formula from training (input to strategy_manager)
- `portfolio_state.json` - Live position state

## Language

The codebase and README are bilingual (Chinese/English). Comments in code are primarily English.
