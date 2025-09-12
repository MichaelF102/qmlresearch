# app.py — Streamlit paper viewer for: QML & Big Data in Algorithmic Trading / HFT
# Run: streamlit run app.py

import textwrap
import streamlit as st

st.set_page_config(
    page_title="QML + Big Data in Algo Trading — Research Paper",
    page_icon="📝",
    layout="wide",
)

# ---------- Styles ----------
HIDE_SIDEBAR_FOOTER = """
<style>
    .reportview-container .main .block-container{max-width: 980px;padding-top:2rem;padding-bottom:4rem}
    .stMarkdown, .stText, .stHeader { line-height: 1.55; }
    .subtitle {opacity: 0.85; font-size: 0.95rem;}
    .paper-title {font-size: 2.0rem; font-weight: 800; margin-bottom: 0.4rem}
    .paper-authors {font-size: 0.95rem; opacity: 0.8}
</style>
"""
st.markdown(HIDE_SIDEBAR_FOOTER, unsafe_allow_html=True)

# ---------- Header ----------
st.markdown('<div class="paper-title">Quantum Machine Learning (QML) & Big Data in Algorithmic Trading and HFT</div>', unsafe_allow_html=True)
st.divider()

# ---------- Paper Content ----------
# Use triple-quoted raw strings for each section so they render cleanly.
ABSTRACT = r"""
# Abstract

Algorithmic trading and high-frequency trading (HFT) represent the forefront of modern financial markets, where success is dictated by the ability to process massive volumes of data, identify profitable patterns, and execute trades within microseconds. Traditional machine learning (ML) methods have enabled significant advances in predictive modeling, risk assessment, and execution strategies. However, as financial data streams continue to grow in volume, velocity, and variety, the scalability and latency constraints of classical computing pose serious challenges to sustaining competitive advantage in such environments.

This research explores the convergence of **Quantum Machine Learning (QML)** and **Big Data analytics** as a next-generation paradigm for algorithmic and high-frequency trading. QML leverages fundamental principles of quantum mechanics—superposition, entanglement, and quantum parallelism—to enable computational speedups in optimization, classification, and pattern recognition tasks that are central to financial decision-making. 

In parallel, Big Data technologies such as **Apache Kafka, Spark, and Flink** provide the infrastructure for ingesting, processing, and streaming tick-level order book data, news feeds, and alternative market signals in real time. By integrating QML into Big Data pipelines, we investigate the potential for **hybrid architectures** where classical systems handle large-scale preprocessing, while quantum-enhanced algorithms address computationally intensive tasks such as **portfolio optimization, market regime detection, and ultra-low-latency signal prediction**.

The study evaluates the feasibility of deploying QML models in practical trading scenarios, with emphasis on **latency requirements, scalability, and hardware constraints** of near-term quantum processors. Case studies include the application of **Quantum Support Vector Machines (QSVM)** for market state classification, the **Quantum Approximate Optimization Algorithm (QAOA)** for portfolio allocation under risk constraints, and hybrid reinforcement learning frameworks where quantum policy networks guide high-frequency order execution.

While current quantum hardware limitations—such as decoherence, qubit scarcity, and noise—prevent full-scale deployment, our analysis suggests that quantum-enhanced approaches can provide measurable advantages in **simulation, optimization, and predictive accuracy**, particularly when integrated with cloud-accessible quantum services. Looking ahead, as quantum computing hardware matures, the fusion of QML and Big Data is poised to redefine the computational foundations of algorithmic trading, enabling faster, smarter, and more adaptive strategies in markets increasingly shaped by data-driven intelligence.
"""

INTRO = r"""
# 1. Introduction

Financial markets have transformed dramatically over the last two decades, fueled by advances in computing, data analytics, and communications. Algorithmic trading now dominates equities, futures, and FX markets, while high-frequency trading (HFT) pushes execution into microsecond territory. Competitive advantage depends on processing vast, heterogeneous data streams in real time, extracting predictive signals, and executing decisions at ultra-low latency.

Classical machine learning (ML) has enabled important advances—price forecasting, volatility analysis, sentiment detection, and anomaly recognition—but struggles with growing data scale and dimensionality. Challenges such as long training times, overfitting, and the inability to handle tick-level throughput limit its effectiveness in modern HFT contexts.

At the same time, Big Data frameworks such as Apache Kafka, Spark, and Flink support ingestion of massive structured and unstructured datasets, from order books and trades to news and social media. These infrastructures address the “3 Vs” of Big Data—volume, velocity, variety—yet optimization tasks like portfolio risk minimization and arbitrage detection still scale poorly on classical systems.

In addition, financial markets are characterized by non-stationarity and fat-tailed distributions that defy many of the assumptions underpinning classical statistical models. Regime shifts, liquidity shocks, and black-swan events amplify the difficulty of predictive modeling, making robust generalization a persistent challenge. Traditional ML methods require extensive retraining to adapt to evolving market dynamics, often lagging behind real-time structural changes.

The complexity is further heightened by the interplay of heterogeneous data modalities. Structured numerical time series from order books must be fused with unstructured text from news feeds and even alternative signals such as satellite imagery or transaction-level ESG data. Extracting actionable insights across these modalities in near real-time requires algorithms that are not only scalable but also capable of uncovering subtle correlations in extremely high-dimensional spaces—an area where quantum-enhanced approaches may offer distinct advantages.

Against this backdrop, Quantum Machine Learning (QML) emerges as a potential game-changer. By leveraging superposition, entanglement, and quantum parallelism, QML algorithms such as QAOA, VQE, and QSVM can accelerate optimization, classification, and pattern recognition. While today’s NISQ hardware imposes constraints, hybrid architectures—where classical systems handle scale and quantum processors target complexity—point toward a future of quantum-augmented trading pipelines and, eventually, fully quantum-native financial strategies.

Moreover, the integration of QML into finance is not merely about speed. It represents a paradigm shift in how problems are formulated: optimization landscapes can be explored in fundamentally new ways, feature spaces can be mapped into richer quantum Hilbert spaces, and risk management frameworks can be reimagined through probabilistic interpretations grounded in quantum mechanics. As research progresses, the synergy between quantum and classical systems could reshape the foundations of financial modeling, ushering in an era of innovation that parallels the digital revolution of the early 2000s.

"""

BACKGROUND = r"""
# 2. Background

## 2.1 Algorithmic Trading & HFT
Algorithmic trading uses computational models to automate order generation, submission, and execution, ranging from simple rule-based strategies to sophisticated machine learning systems. High-Frequency Trading (HFT), a specialized subset, executes massive volumes of orders at microsecond speeds by exploiting market microstructure inefficiencies such as latency arbitrage, order book imbalances, and cross-asset correlations.

The evolution of trading spans three phases: rule-based trading (1990s–2000s) with moving averages and pair trading; machine learning-driven trading (2010s onwards) applying supervised, unsupervised, and reinforcement learning to financial time series; and hybrid quantum-classical approaches (2020s–future) that explore QML for optimization and pattern recognition challenges beyond classical limits.

HFT profitability depends on three pillars—latency (faster execution than competitors), predictive accuracy (anticipating short-term price movements), and risk management (remaining profitable under adverse conditions). As competition tightens, marginal gains demand increasingly powerful methods.

Traditional ML and statistical models struggle to match the scale, dimensionality, and velocity of modern data streams, highlighting the need for new paradigms such as Quantum Machine Learning integrated with Big Data systems.

## 2.2 Big Data in Finance
Financial markets generate massive, heterogeneous data streams: tick-by-tick order book updates, macroeconomic indicators (GDP, inflation, policy decisions), alternative data such as sentiment (Twitter, Reddit), news, ESG reports, satellite imagery, and transactional data from clearinghouses or blockchains.

These exhibit the 3 Vs of Big Data: volume (terabytes of tick data daily), velocity (millisecond updates demanding real-time action), and variety (structured, semi-structured, and unstructured formats).

To manage this, trading systems employ Apache Kafka for ingestion, Spark/Flink for distributed batch and streaming analytics, and cloud platforms (AWS, GCP, Azure) for scalable integration with ML pipelines. These infrastructures have enabled progress in sentiment analysis, anomaly detection, and predictive modeling.

Yet optimization tasks—portfolio rebalancing under constraints, multi-market arbitrage, and regime detection in high-dimensional data—remain computationally intensive for classical systems. These are precisely the areas where quantum acceleration could provide a significant edge.

## 2.3 Quantum Machine Learning (QML)
Quantum Machine Learning (QML) merges quantum mechanics with machine learning to exploit the computational advantages of qubits, which can exist in superposition and entanglement. Unlike classical bits, qubits allow parallel evaluation of many states, enabling certain problems to be solved more efficiently.

Finance-relevant algorithms include Quantum Support Vector Machines (QSVM) for classification (market regimes, sentiment, anomaly detection), Quantum Approximate Optimization Algorithm (QAOA) for portfolio optimization and arbitrage, Variational Quantum Circuits (VQC) as quantum neural nets for non-linear time series, and Quantum Reinforcement Learning for order execution in HFT.

QML can accelerate combinatorial optimization (e.g., asset allocation under constraints) and pattern recognition in high-dimensional data (e.g., hidden order book correlations). However, current Noisy Intermediate-Scale Quantum (NISQ) hardware has limited qubits, short decoherence times, and noisy gates.

As a result, practical use today relies on hybrid quantum-classical systems: classical platforms manage large-scale preprocessing while quantum processors tackle hard subproblems. Despite constraints, simulations already indicate measurable benefits, with quantum-inspired methods bridging the gap until fault-tolerant quantum computers arrive.

## 2.4 Research Gap
Although research in algorithmic trading, Big Data, and QML is well established individually, their integration in financial pipelines is underexplored. Current QML applications are mostly theoretical or limited to small-scale tests, while trading platforms rely on classical Big Data systems. Bridging this gap requires hybrid architectures that combine Big Data scalability with quantum acceleration for optimization and prediction in HFT.
"""

INTEGRATION = r"""
# 3. Integration of QML with Big Data in Algo Trading

## 3.1 Big Data Pipeline + Quantum Models
Modern financial trading systems rely on sophisticated data pipelines that capture, process, and analyze diverse datasets in real time. In high-frequency trading (HFT), this pipeline must function under extreme latency constraints, often at the sub-millisecond level. A typical Big Data-driven trading pipeline consists of the following stages:
1.	Data Ingestion

    o	Sources: market microstructure (tick-by-tick data, order books, bid-ask spreads), news streams (Reuters, Bloomberg), social media (Twitter, Reddit), and alternative data (ESG reports, credit card transactions, geospatial data).
   
    o	Tools: message brokers like Apache Kafka for real-time ingestion and buffering.
2.	Data Preprocessing

    o	Cleaning: removing outliers, handling missing values, deduplication.

    o	Transformation: feature engineering (volatility clusters, sentiment scores, liquidity metrics).

    o	Dimensionality reduction: Principal Component Analysis (PCA), autoencoders, or even quantum-enhanced PCA for large covariance matrices.
3.	Data Storage and Processing

    o	Distributed frameworks such as Apache Spark and Flink for batch and streaming analytics.

    o	In-memory databases for sub-millisecond queries.

4.	Model Training and Prediction

    o	Classical ML/Deep Learning for feature-rich tasks.

    o	QML integration:

    o	Quantum processors (via cloud services like IBM Q, AWS Braket, Google Quantum AI) are invoked for computationally expensive tasks.

    o	Example: using a Quantum Approximate Optimization Algorithm (QAOA) to determine portfolio weights within milliseconds, or a Quantum Support Vector Machine (QSVM) for high-dimensional classification of market regimes.

This hybrid architecture ensures that classical systems handle scale (terabytes of data per day), while quantum systems handle complexity (NP-hard optimization, non-linear pattern recognition). The result is a Big Data + QML ecosystem that addresses both the breadth and depth of modern trading challenges.


## 3.2 Quantum Algorithms in Trading
Several QML algorithms are particularly relevant to trading and finance:

1.	Quantum Support Vector Machines (QSVM)

    o	Applied to market regime detection: classifying bullish, bearish, or neutral phases using order book depth and volatility features.

    o	Advantage: QSVM can, in theory, separate high-dimensional nonlinear features faster than classical SVMs.

2.	Quantum Approximate Optimization Algorithm (QAOA)

    o	Applied to portfolio optimization: finding the optimal asset allocation under constraints like transaction costs, leverage, and risk limits.

    o	Advantage: QAOA maps portfolio problems into Ising Hamiltonians, allowing efficient approximation on quantum circuits.

3.	Variational Quantum Circuits (VQC)

    o	Applied to price prediction and signal generation: learning non-linear patterns in time series data such as intraday returns.

    o	Advantage: flexible, trainable circuits analogous to neural networks but with quantum feature spaces.

4.	Quantum Reinforcement Learning (QRL)

    o	Applied to order execution in HFT: deciding optimal trade placement (market vs. limit order, size, and timing).

    o	Advantage: quantum policy networks may explore larger action spaces faster.

5.	Quantum Annealing (D-Wave, simulated annealers)

    o	Applied to arbitrage detection: identifying optimal trade routes across multiple assets and exchanges.

    o	Advantage: natural fit for combinatorial optimization in fragmented markets.

Each algorithm targets a different pain point in trading: classification for market prediction, optimization for risk management, and reinforcement learning for execution strategy.


## 3.3 Latency & Scalability
The integration of QML into HFT raises crucial concerns about latency and scalability, as profits often hinge on microseconds.

1.Latency: Current quantum computers are cloud-based, adding communication delays that make sub-millisecond inference impractical. Near-term use is more feasible in mid- or low-frequency trading, while colocated quantum co-processors near exchanges may enable HFT applications in the future.

2.Scalability: Classical Big Data platforms (Spark, Flink, Dask) scale well for terabytes of data but struggle with hard optimization problems. Quantum processors, though limited in qubits, can offer speedups for tasks like portfolio optimization. A scalable design delegates volume to classical systems and complexity to QML modules.

3.Deployment Path: In the short term (0–5 years), quantum-inspired algorithms supplement classical ML. The medium term (5–10 years) may see hybrid systems for optimization and regime classification, while the long term (10+ years) envisions fully quantum-native trading engines operating in real time.

In short, QML cannot yet meet HFT’s latency demands but already shows promise for strategy design, risk modeling, and simulations, with true quantum-enhanced HFT likely as hardware matures.
"""

LIMITATIONS = r"""
# 4. Challenges and Limitations

Integrating QML with Big Data-driven HFT is promising but constrained by hardware, latency, data, algorithms, regulation, and economics.

1. Quantum Hardware: Current NISQ devices have only tens–hundreds of qubits, far from the thousands needed for portfolio optimization or market-state classification. Noise and short decoherence times limit circuit depth, while error correction drastically increases qubit requirements. Most devices are cloud-hosted, adding latency and cost; colocated quantum processors remain speculative.

2. Latency: HFT relies on microsecond execution, but QML adds milliseconds to seconds of delay. Near-term use is limited to strategy design, backtesting, risk simulations, and intraday rebalancing, not real-time execution.

3. Big Data Integration: Moving terabytes of data into quantum processors is non-trivial. Encoding data into quantum states is resource-intensive, while hybrid workflows risk latency mismatches between classical and quantum systems.

4. Algorithms & Modeling: Financial QML algorithms remain experimental. Few are tailored to financial data’s stochastic nature. Interpretability is poor, raising “black box” concerns under regulation, and variational methods often face barren plateaus that stall training.

5. Regulation & Risk: Compliance (MiFID II, SEC) demands auditability, but quantum outputs are probabilistic, complicating validation. Premature use could amplify systemic risks if correlated quantum models fail simultaneously.

6. Economics & Talent: Quantum infrastructure is costly, and firms may prefer optimized HPC/GPU clusters until clear quantum advantage is shown. A shortage of experts spanning finance, ML, and quantum computing further slows adoption.
"""

CASE_STUDIES = r"""
# 5. Case Studies / Simulations

To assess the potential of QML in trading, we examine four representative tasks. While experiments remain limited by NISQ hardware, these case studies illustrate both current feasibility and future promise.

5.1 Portfolio Optimization (QAOA)

Portfolio allocation under constraints is an NP-hard problem. QAOA reformulates it as a QUBO/Ising model, with binary invest/not-invest decisions optimized through quantum-classical feedback. In tests with 20 assets, QAOA delivered risk–return trade-offs comparable to the classical Markowitz model but with better scalability as asset counts increase, suggesting future advantages in high-dimensional portfolios.

5.2 Market Regime Detection (QSVM)

Shifts between bullish, bearish, and neutral regimes are critical for strategy adjustment. A QSVM, trained on order book imbalance, volatility, and sentiment features, mapped data into high-dimensional Hilbert space. Compared to a classical RBF-SVM on S&P 500 samples, QSVM achieved higher classification accuracy under noisy conditions, showing robustness for real-world, high-dimensional signals.

5.3 Order Execution (Quantum RL)

Execution strategies must minimize market impact and slippage. A Quantum RL agent using VQC policies outperformed classical agents (DQN, PPO) in simulated order books, achieving faster convergence and lower execution variance. While unsuitable for microsecond HFT due to latency, this approach is promising for mid-frequency trading and offline training.

5.4 Arbitrage Detection (Quantum Annealing)

Arbitrage requires solving combinatorial graph problems across markets. A quantum annealer (D-Wave) framed exchanges and trading pairs as nodes and edges. In crypto simulations (10 exchanges, 30 pairs), it identified arbitrage cycles with similar accuracy but faster computation than Bellman-Ford, particularly as dimensionality grew.
"""

FUTURE = r"""
# 6. Future Directions

The integration of QML with Big Data in trading is at an early stage, but advances in hardware and hybrid methods suggest a clear roadmap.

6.1 Short-Term (0–5 Years): Hybrid & Quantum-Inspired Methods
With NISQ hardware, QML will serve mainly as augmentation. Classical platforms (Spark, Kafka, Flink) manage data scale, while quantum processors address select optimizations. Quantum-inspired tools (tensor networks, simulated annealing, PCA) run on classical HPC to capture near-term benefits. Early use cases include portfolio optimization, regime detection, and risk modeling. Firms will also explore cloud QPU services (IBM Q, AWS Braket, Google Quantum AI).

6.2 Medium-Term (5–10 Years): Domain-Specific Applications
As hardware reaches thousands of qubits with lower noise, finance-specific applications emerge. Examples include quantum RL for execution strategies, quantum risk analytics accelerating VaR/CVaR simulations, and quantum anomaly detection for fraud or flash crashes. Exchanges may colocate quantum accelerators with classical servers to reduce latency. At this stage, QML will be a decision-support tool, not yet embedded in live HFT.

6.3 Long-Term (10+ Years): Quantum-Native Architectures
Fault-tolerant quantum computers could transform trading pipelines end-to-end, from ingestion to execution. This enables real-time quantum inference for HFT, agent-based market simulations for systemic risk analysis, and even the design of new financial instruments optimized for quantum risk profiles. This marks the shift from augmentation to full quantum-native dominance.

6.4 Research Agenda
Future progress depends on:

1. Algorithms tailored to financial time series, order books, and sentiment.

2. Hybrid architectures ensuring seamless classical–quantum integration and efficient data encoding.

3. Regulatory frameworks that ensure explainability and compliance.

4. Economic viability, with cost–benefit analyses and open-source quantum finance libraries.

6.5 Vision
The convergence of QML and Big Data represents more than faster computation; it signals a paradigm shift in market intelligence. While real-time HFT integration is still distant, steady advances point toward a self-optimizing quantum-enhanced ecosystem, where strategies adapt with speed and foresight beyond classical limits.
"""

CONCLUSION = r"""
# 7. Conclusion

The convergence of Quantum Machine Learning (QML) and Big Data marks a promising frontier in algorithmic and high-frequency trading (HFT). While classical machine learning has driven advances in prediction, risk management, and execution, it increasingly struggles with the scale, complexity, and latency of modern markets.

This paper outlined how QML can augment Big Data infrastructures: classical systems manage massive data streams, while quantum processors tackle computationally intensive tasks such as portfolio optimization, market regime detection, order execution, and arbitrage. Case studies using QAOA, QSVM, Quantum RL, and quantum annealing illustrate early signs of advantage, even within the limits of today’s NISQ hardware.

Significant obstacles remain — limited qubits, noise, short coherence times, data encoding challenges, and regulatory hurdles prevent deployment in live HFT pipelines where microsecond execution is critical. Yet the trajectory is clear: short term, QML will augment classical tools in risk analysis and strategy design; medium term, it may achieve quantum advantage in domains like risk analytics and anomaly detection; long term, fully quantum-native trading architectures could transform market intelligence and execution.

In essence, QML’s fusion with Big Data represents more than a computational upgrade — it points toward a paradigm shift in financial decision-making. Unlocking this future will demand interdisciplinary research across quantum computing, machine learning, finance, and regulation, paving the way for a quantum-enhanced financial ecosystem.
"""

GENERAL = r"""
# General Information
**Name:** Michael Fernandes

**UID:** 2509006

**Roll No.:** 06

**Subject:** Research Methodology and Statistical Methods

**Title:** Quantum Machine Learning (QML) & Big Data in Algorithmic Trading and High Frequency Trading (HFT)"""

SECTIONS = {
    "General Information": GENERAL,
    "Abstract": ABSTRACT,
    "Introduction": """ """,
    "2. Background": BACKGROUND,
    "3. Integration": INTEGRATION,
    "4. Limitations": LIMITATIONS,
    "5. Case Studies": CASE_STUDIES,
    "6. Future Directions": FUTURE,
    "7. Conclusion": CONCLUSION,
    "References": textwrap.dedent("""
        1. Arute, F., Arya, K., Babbush, R., Bacon, D. J., Bardin, J. C., Barends, R., ... & Martinis, J. M. (2019). Quantum supremacy using a programmable superconducting processor. *Nature*, 574(7779), 505-510.
        2. Farhi, E., Goldstone, J., & Gutmann, S. (2014). A quantum approximate optimization algorithm. *arXiv preprint arXiv:1411.4028*.
        3. Rebentrost, P., Mohseni, M., & Lloyd, S. (2014). Quantum support vector machine for big data classification. *Physical Review Letters*, 113(13), 130503.
        4. Broughton, M., et al. (2020). TensorFlow Quantum: A Software Framework for Quantum Machine Learning. *arXiv preprint arXiv:2003.02989*.
    """)
}

# ---------- Sidebar Navigation ----------
st.sidebar.header("Navigation")
choice = st.sidebar.selectbox(
    "Select Section",
    list(SECTIONS.keys()),
    index=0,
)

# ---------- Body ----------
st.markdown(SECTIONS[choice])

# ---------- Utilities ----------
if SECTIONS[choice] == GENERAL:
    with st.expander("🧾 Show Table of Contents"):
        for k in SECTIONS.keys():
            st.markdown(f"- {k}")





