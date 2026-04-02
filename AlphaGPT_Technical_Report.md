# AlphaGPT 技术报告：基于强化学习的自动因子挖掘系统

## 一、项目概述

AlphaGPT 是一套面向 Solana meme 币的加密量化交易系统。核心思路：**将因子挖掘建模为强化学习的序列生成问题**——用 Transformer 作为策略网络自动生成因子公式，用回测得分作为奖励信号训练生成器，最终将高分公式接入链上执行实盘交易。

与传统量化中人工编写因子不同，本系统不是用模型预测价格，而是**让模型学会写策略本身**。

---

## 二、因子空间设计

### 2.1 基础因子（6 维特征）

因子公式的原材料。按 meme 币的定价逻辑分层设计：

| 编号 | 名称 | 含义 | 计算方式 |
|------|------|------|----------|
| 0 | ret | 对数收益率 | `log(close / close_prev)` |
| 1 | liq_score | 流动性健康度 | `clamp(liquidity / fdv × 4, 0, 1)` |
| 2 | pressure | 买卖压力（K线实体占比） | `tanh((close - open) / (high - low) × 3)` |
| 3 | fomo | FOMO 加速度（量的二阶差分） | 成交量变化率的差分 |
| 4 | dev | 偏离均线程度 | `(close - MA20) / MA20` |
| 5 | log_vol | 对数成交量 | `log1p(volume)` |

**标准化方式**：使用 MAD（median absolute deviation）而非 z-score。原因是 meme 币收益分布极端厚尾，均值和标准差对离群值过于敏感。标准化后 clamp 到 [-5, 5]。

此外还预留了 `AdvancedFactorEngineer`（12 维），额外包含波动率聚集、动量反转、RSI、振幅、收盘位置、量趋势等扩展因子，用于后续因子空间扩展。

### 2.2 算子（12 个操作符）

因子公式中的运算符，分三类：

**算术类（构造线性/非线性组合）：**

| 名称 | 元数 | 含义 |
|------|------|------|
| ADD | 2 | x + y |
| SUB | 2 | x - y |
| MUL | 2 | x × y（等价于交互项） |
| DIV | 2 | x / (y + ε) |
| NEG | 1 | -x |
| ABS | 1 | \|x\| |
| SIGN | 1 | sign(x)，信号离散化为 {-1, 0, 1} |

**条件/非线性类（捕捉 regime switching）：**

| 名称 | 元数 | 含义 |
|------|------|------|
| GATE | 3 | condition > 0 ? x : y（条件选择，表达力最强的算子） |
| JUMP | 1 | relu(zscore(x) - 3)，仅在 3σ 以上激活，检测极端跳变 |

**时序类（引入时间维度）：**

| 名称 | 元数 | 含义 |
|------|------|------|
| DECAY | 1 | x + 0.8×lag1 + 0.6×lag2，指数衰减叠加 |
| DELAY1 | 1 | 滞后一期 |
| MAX3 | 1 | max(x, lag1, lag2)，滚动最大值 |

**设计 trade-off**：不放 rolling mean / rolling std，因为 DECAY 和 MAX3 已能近似；保留 GATE 以覆盖分段线性模型的表达力。

---

## 三、公式执行：栈式虚拟机（StackVM）

生成的公式是一个 token 序列（最长 12 个 token），由 StackVM 以逆波兰表达式的方式执行：

- 遇到因子 token（0-5）→ 将对应特征向量压栈
- 遇到算子 token（6-17）→ 从栈中弹出对应数量的操作数，计算后压回结果
- 执行完毕后栈中恰好剩一个元素 → 该元素即为因子信号
- 栈深度不足或执行异常 → 返回 None（无效公式）

**示例**：序列 `[ret, fomo, MUL, DECAY]` 的执行过程：

```
1. 压入 ret              → 栈: [ret]
2. 压入 fomo             → 栈: [ret, fomo]
3. MUL: 弹出两个相乘     → 栈: [ret × fomo]
4. DECAY: 衰减叠加       → 栈: [ret×fomo + 0.8×lag1 + 0.6×lag2]
```

**StackVM 的优势**：
- 天然类型安全，非法公式直接判定为无效
- 所有操作均为 batch tensor 运算，向量化执行效率高

---

## 四、训练方法：Policy Gradient

### 4.1 核心思路

将因子挖掘建模为强化学习问题：

- **State**：已生成的 token 序列
- **Action**：下一个 token 的选择（18 维离散空间）
- **Reward**：公式的回测得分
- **Policy Network**：Transformer（AlphaGPT）

### 4.2 训练流程

```
for step in 1..1000:
    1. Transformer 自回归生成 8192 条公式（每条 12 个 token）
    2. StackVM 逐条执行 → 得到因子信号
    3. MemeBacktest 回测打分 → 得到 reward
       - 无效公式：reward = -5
       - 常数公式（std < 1e-4）：reward = -2
    4. 计算 advantage = (reward - mean) / std
    5. 更新：loss = -Σ log_prob(token) × advantage（标准 REINFORCE）
    6. AdamW 梯度更新 + LoRD 正则化
```

**Batch size = 8192** 是关键设计——因子空间大，小 batch 的 reward 方差太大，policy gradient 无法收敛。大 batch 保证每步有足够正样本来估计 baseline。

### 4.3 语法合法性的学习

系统不硬编码任何语法规则。非法公式（如栈深度不足）通过 -5 的惩罚信号，让模型通过 RL 自然学会生成合法序列。

---

## 五、回测评分（Reward Shaping）

```python
signal = sigmoid(factors)                          # 连续信号 → [0,1]
position = (signal > 0.85) * (liquidity > 500k)    # 高置信 + 流动性安全
slippage = 0.6% + trade_size / liquidity            # 基础费 + 市场冲击
net_pnl = position × target_return - turnover × slippage
score = cum_return - 2.0 × count(single_bar_loss > 5%)
final = median(score)  # 跨所有资产取中位数
```

**关键设计选择**：

| 设计 | 原因 |
|------|------|
| 阈值 0.85 而非 0.5 | 强制高置信开仓，避免 meme 币高滑点下的过度交易 |
| 滑点含市场冲击项 | `trade_size / liquidity`，流动性越差滑点越大，比固定比例更真实 |
| 惩罚大亏次数而非 max drawdown | max drawdown 是单一极值易被 outlier 主导，次数更 robust |
| 系数 2.0 | 每次大亏需两倍正收益弥补，抑制"赚小亏大"的策略 |
| 最低活跃度 ≥ 5 次 | 防止模型学到"不交易"这种 trivially safe 的策略 |
| 取 median 而非 mean | meme 币个别 token pump 10x 会拉飞 mean，median 保证策略在多数资产上有效 |

---

## 六、模型架构：AlphaGPT

### 6.1 基本参数

| 参数 | 值 |
|------|-----|
| d_model | 64 |
| nhead | 4（每头 16 维） |
| num_layers | 2 |
| dim_feedforward | 128 |
| num_loops | 3（等效 6 层深度） |
| vocab_size | 18（6 特征 + 12 算子） |
| max_seq_len | 13 |
| 总参数量 | ~数万级别 |

### 6.2 架构设计

**Looped Transformer**：每层的 attention + FFN 用相同权重循环执行 3 次，再进入下一层。用 2 层参数量达到 6 层计算深度。对于 vocab size 仅 18 的小搜索空间，比堆层数更高效。与 Universal Transformer、ALBERT 的 parameter sharing 思路一致。

**QK-Norm**：对 Q、K 做 L2 归一化后乘可学习 scale，稳定注意力分布，防止 RL 训练中梯度波动导致 attention logits 发散。

**MTPHead（Multi-Task Pooling）**：3 个 task head + router，softmax 加权输出。不同类型的好公式（动量型、均值回归型、流动性过滤型）可能需要不同生成策略，router 学习动态路由。

**LoRD 正则化（Low-Rank Decay）**：通过 Newton-Schulz 迭代逼近正交矩阵，对 Q/K 投影矩阵施加低秩约束：`W -= decay_rate × Y`。因为 vocab 仅 18 个 token，attention 没有理由使用满秩矩阵，低秩约束作为归纳偏置促进泛化。同时通过 StableRankMonitor 监控训练过程中的有效秩变化。

---

## 七、与暴力搜索的对比

搜索空间：18¹² ≈ 1.1 × 10¹⁵ 种组合。

| 维度 | 暴力 / 随机搜索 | RL + Transformer |
|------|-----------------|-----------------|
| 采样独立性 | 每次独立，无学习 | 模型学到组合模式的先验，采样越来越有方向性 |
| 当前规模 | 剪枝后可行但效率低 | 有优势但非压倒性（~800万条公式的预算） |
| 规模扩展 | 搜索空间指数爆炸后不可行 | Transformer 的 guided search 仍然有效 |
| 知识复用 | 无 | 学到的 policy 本身是对"什么组合有效"的知识压缩 |
| vs 遗传编程（GP） | GP 靠 mutation + crossover，无梯度 | 有梯度，参数在所有公式间共享，采样效率更高 |

**当前规模下（18 token × 12 长度）RL 的优势有限，但框架的价值在于可扩展性**——特征扩展到 12 维（AdvancedFactorEngineer 已实现）、算子和长度增加后，RL 的 guided search 优势会指数级放大。

---

## 八、系统架构（全链路）

```
data_pipeline/          model_core/           strategy_manager/        execution/
Birdeye/DexScreener  →  Transformer + RL  →  信号扫描 + 风控       →  Solana + Jupiter
    ↓                       ↓                      ↓                       ↓
 PostgreSQL/          best_meme_strategy    portfolio_state.json      链上交易
 TimescaleDB               .json           (止损/止盈/追踪止损)
```

四层之间通过两个 JSON 文件解耦：
- `best_meme_strategy.json`：训练输出 → 策略输入
- `portfolio_state.json`：策略输出 → 仪表盘输入

训练和交易完全独立运行，训练崩溃不影响在线策略。

---

## 九、策略执行层

- **数据同步**：每 15 分钟拉取最新行情
- **信号扫描**：每 60 秒用 StackVM 执行公式对 Top N 代币打分
- **风控**：流动性检查（最低 $5k）、Jupiter 退出路径验证、余额缓冲验证
- **仓位管理**：最多 3 个持仓，每笔 2.0 SOL，-5% 止损，10% 止盈（卖 50%），5% 触发追踪止损
- **执行**：Jupiter v6 聚合器报价 → Solana RPC 签名发送 → 重试确认
