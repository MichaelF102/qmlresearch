import streamlit as st
import pandas as pd
import re
import os
import tempfile
import time
import gzip
from collections import Counter
st.set_page_config(
    page_title="RM CIA 2",
    page_icon="📝",
    layout="wide",
)

# --- Sections dictionary: use keys that match the sidebar exactly ---
SECTIONS = {
    "General": """ """,
    "Abstract": """ """,
    "Introduction": """ """,
    "Research Objective": """ """,
    "Research Questions": """ """,
    "Literature Review": """ """,
    "Methodology": """ """,
    #"Analysis": """ """,
    "Challenges and Limitations": """ """,
    "Case Studies": """ """,
    "Future Directions": """ """,
    "Conclusion": """ """,
    "References": """ """,
    
}

# --- Sidebar ---
st.sidebar.header("Navigation")
section = st.sidebar.selectbox("Select Section", list(SECTIONS.keys()))

# --- Main area ---
st.title("📉 QML in Finance")

st.markdown(f"### {section}")

if section == "General":
    st.write("""
**Name:** Michael Fernandes

**UID:** 2509006

**Roll No.:** 06

**Subject:** Research Methodology and Statistical Methods

**Title:** Quantum Machine Learning in Finance""")
    
    st.image("img41.png", width='stretch')

if section == "Abstract":
    st.write("""High-frequency and algorithmic trading are less about glamour and more about grit. At their core, these systems are locked in a race: who can read the flood of orders, spot the tiniest statistical edge, and fire back with a trade before that edge vanishes. Classical machine learning has given traders sharper tools for prediction and risk control, yet anyone working with live market data knows the same story—feeds get heavier, signals arrive faster, and old pipelines start to choke.

This project explores whether quantum methods can offer a way out of that bottleneck. Quantum Machine Learning (QML) is not magic, but it does something classical math cannot: it exploits superposition and entanglement to examine many possibilities at once. That makes it attractive for problems that trading firms wrestle with daily—portfolio optimization, regime shifts, and microsecond-level forecasting. Meanwhile, Big Data platforms such as Kafka, Spark, and Flink do the heavy lifting of capturing and streaming millions of ticks and headlines in real time. The proposal here is simple: let the classical stack manage the volume, and hand off the hardest optimization problems to quantum routines.

We ground this idea with examples. A quantum SVM can help sort markets into risk states. QAOA provides a route for balancing portfolios under constraints. Reinforcement learning, when paired with quantum policy networks, could eventually guide order placement itself. These sketches are not free of caveats—current hardware is noisy, prone to errors, and barely large enough for toy problems. Still, initial tests already hint at measurable improvements, especially in simulation speed and in finding solutions that classical solvers miss.

The larger point is less about today’s results and more about the direction of travel. Quantum processors are slowly becoming more stable and more widely available through cloud services. As they mature, their marriage with Big Data infrastructure may quietly redraw the map of algorithmic trading: faster not only in execution but also in adaptation, learning, and survival in markets where information never stops moving. 
""")
if section == "Introduction":
    st.write("""
Over the last twenty years, markets have stopped looking like crowded pits and started looking like server rooms. Computing power, fiber lines, and endless data feeds have pushed trading into a game of speed. Algorithms now dominate equities, futures, and FX, and high-frequency desks fight over microseconds. Survival comes down to who can wade through firehoses of data, spot something useful, and act before the signal fades.

Machine learning has been the workhorse of this era. It forecasts, measures risk, reads sentiment, and catches odd behavior. But in practice, the models stumble when asked to handle the messy firehose that is modern tick data. Training takes longer than the market gives you. Models overfit. And when dimensionality explodes, throughput at tick level simply breaks them. What looks clever in backtests often buckles live.

Big Data systems—Kafka, Spark, Flink—keep things flowing. They chew on structured order books, trades, tweets, and news blurbs. They handle the scale, the “three Vs.” But scale isn’t the same as solving optimization. Arbitrage detection, portfolio risk balancing—these problems balloon, and classical solvers grind slowly, sometimes too slowly to matter.

Markets themselves pile on more headaches. They shift regimes, lurch through liquidity droughts, and every few years throw a black swan at your models. Distributions are fat-tailed, assumptions break, and retraining ML systems becomes a Sisyphean job. You’re always catching up, never quite ahead.

And then there’s the data mix. Structured prices on one side, unstructured headlines on the other, plus exotic inputs like satellite images or ESG transaction flows. Making sense of that soup in anything close to real time is tough. You need tools that find faint, hidden links across wildly different signals, in spaces far too large for ordinary math to tame. This is the spot where quantum approaches start to look interesting.

Quantum Machine Learning (QML) isn’t a magic wand, but it can explore problem spaces differently. Algorithms such as QAOA, VQE, and QSVM use quantum properties to make optimization and classification less painful. Hardware today is noisy, small, and fragile—closer to prototypes than production. Still, hybrid ideas show a path: classical stacks take care of scale, while quantum chips focus on the hardest puzzles.

The real shift is not just speed but perspective. Finance problems can be framed differently in a quantum setting. Feature spaces stretch into new dimensions. Optimization landscapes are searched in strange, sometimes more efficient ways. Even risk can be reimagined with quantum-style probability. If this direction plays out, it could echo the early 2000s digital revolution—markets rebuilt around a new computational backbone, where being faster also means being smarter. 
""")
    st.image("img38.png", width='stretch')
    st.image("img37.webp", width='stretch')
    st.image("img36.webp",  width='stretch')
    
    st.image("img3.webp",  width='stretch')
    st.image("img39.png",  width='stretch')
    st.image("img20.jpeg", caption="QML vs Classical ML", width='stretch')


if section == "Research Objective":
    st.markdown("""
        <h3>1. Investigate the Current State of QML and Big Data in Finance</h3>
        <h3>2. Develop a Hybrid Big Data–QML Framework</h3>
        <h3>3. Assess Latency and Scalability Trade-offs</h3>
        <h3>4. Conduct Case Studies and Simulation Experiments</h3>
        <h3>5. Analyze Challenges, Risks, and Adoption Barriers</h3>
        <h3>6. Define a Roadmap for Future Research and Development</h3>
    """, unsafe_allow_html=True)

if section == "Research Questions":
    st.markdown("""
<h4>1.	Algorithmic Performance</h4>
                
•	In which areas of algorithmic trading and HFT do classical machine learning methods fail to meet the demands of scale, latency, or complexity?
                
•	Can QML algorithms (e.g., QAOA, QSVM, QRL) provide measurable improvements in optimization, classification, or pattern recognition tasks?

<h4>2.	System Architecture</h4>

•	How can hybrid architectures be designed where classical Big Data systems manage data volume while quantum processors address computational bottlenecks?

•	What orchestration strategies best integrate Kafka, Spark, or Flink with QML modules?

<h4>3.	Latency and Scalability</h4>
                
•	To what extent do current NISQ devices introduce prohibitive latency for real-time HFT execution?
                
•	Are there mid-frequency or simulation-based use cases where QML can deliver practical benefits despite hardware constraints?
                
•	How does QML performance scale as data dimensionality, asset universes, or feature sets expand?
                
<h4>4.	Case Study Validation</h4>
                
•	How do QML approaches compare with classical baselines in portfolio optimization, regime detection, order execution, and arbitrage detection?
                
•	Do QML-enhanced methods demonstrate robustness in noisy, high-dimensional financial environments?
                
<h4>5.	Adoption Challenges</h4>
                
•	What regulatory, interpretability, and systemic risk issues arise from deploying QML-enhanced trading algorithms?
                
•	How do cost-benefit trade-offs and talent shortages affect the economic viability of QML adoption in finance?
                
<h4>6.	Future Roadmap</h4>
                
•	What short-, medium-, and long-term pathways exist for integrating QML into financial markets?
                
•	What advances in hardware, algorithms, and infrastructure are most critical for transitioning from proofs-of-concept to production-grade trading engines?

    """, unsafe_allow_html=True)

if section == "Literature Review":
    st.markdown("""
<h2>1. Algorithmic and High-Frequency Trading (HFT)</h2>
""", unsafe_allow_html=True)
    st.image("img9.png", use_container_width=True)
    st.markdown("""
<h2>2. Big Data in Financial Trading</h2>
""", unsafe_allow_html=True)
    st.image("img2.png",  use_container_width=True)
    st.markdown("""
<h2>3. Quantum Machine Learning Algorithms</h2>
""", unsafe_allow_html=True)
    st.image("img17.png",  use_container_width=True)
    st.markdown("""
<h2>4. Applications of QML in Finance</h2>
""", unsafe_allow_html=True)
    st.image("img11.png",  use_container_width=True)
    st.markdown("""
<h2>5. Industry Perspectives</h2>
""", unsafe_allow_html=True)
    st.image("img12.webp",  use_container_width=True)
    st.markdown("""
<h2>6. Regulatory, Risk, and Adoption Considerations</h2>
""", unsafe_allow_html=True)
    st.image("img13.webp",  use_container_width=True)
    

if section == "Methodology":
    st.markdown("""
<h2>1. Big Data Pipelines</h2>
""", unsafe_allow_html=True)
    st.image("img14.webp",  use_container_width=True)
    st.markdown("""
<h2>2. Quantum Algorithms in Trading</h2>
                
<h3>1.Quantum Support Vector Machines (QSVM)</h3> 
A QSVM is a classical SVM that uses a quantum kernel: a quantum circuit embeds data x into a high-dimensional Hilbert space.

The circuit estimates similarities via Hadamard/swap-style tests, yielding the kernel (Gram) matrix.

A standard SVM then trains on this matrix to find the separating hyperplane.

Idea: quantum feature maps can produce rich, hard-to-classically-simulate kernels, potentially improving generalization on some problems.

Think of a QSVM as a regular SVM that outsources “similarity checking” to a quantum circuit.

We turn each data point into a pattern on qubits (a feature map).

The quantum computer runs a tiny test to measure how alike two points are; do this for all pairs to make a similarity table (kernel).

A normal SVM uses that table to draw the best boundary between classes.

Why bother? The quantum mapping can capture very rich patterns that might be hard for classical methods—though today it’s limited by small, noisy hardware and the cost of many measurements                          
                
Reality today: benefits are constrained by noise, finite shots, and small qubit counts—so QSVMs are mainly demonstrated on small datasets. 
""", unsafe_allow_html=True)
    st.image("img32.png",  use_container_width=True)
    if "show_code" not in st.session_state:
        st.session_state.show_code = False
    if st.button("QSVM ",key="qsvm_code_button"):
        st.session_state.show_code = not st.session_state.show_code

    # ---- The code to display ----
    QSVM_CODE = r'''
    # QSVM — minimal, version-robust implementation (Qiskit + scikit-learn)
    # pip install qiskit qiskit-aer qiskit-machine-learning scikit-learn numpy
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    class QSVM:
        def __init__(self, feature_dim, reps=2, shots=2048, C=1.0, seed=7):
            self.reps, self.shots, self.C, self.seed = reps, shots, C, seed
            self._kernel, self._eval = self._build_kernel(feature_dim)
            self.scaler = StandardScaler()
            self.X_train_ = None
            self.clf = SVC(kernel="precomputed", C=C)

        def _build_kernel(self, feature_dim):
            """Return (kernel_object, eval_fn) that computes Gram matrices."""
            try:
                # New API (preferred)
                from qiskit.circuit.library import ZZFeatureMap
                from qiskit_aer.primitives import Sampler
                from qiskit_machine_learning.kernels import FidelityQuantumKernel
                fmap = ZZFeatureMap(feature_dimension=feature_dim, reps=self.reps, entanglement="full")
                sampler = Sampler(options={"shots": self.shots, "seed": self.seed})
                qk = FidelityQuantumKernel(feature_map=fmap, sampler=sampler)
                return qk, (lambda X, Y=None: qk.evaluate(X, Y))
            except Exception:
                # Fallback (older API)
                from qiskit import BasicAer
                from qiskit.circuit.library import ZZFeatureMap
                from qiskit_machine_learning.kernels import QuantumKernel
                backend = BasicAer.get_backend("qasm_simulator")
                fmap = ZZFeatureMap(feature_dimension=feature_dim, reps=self.reps, entanglement="full")
                qk = QuantumKernel(feature_map=fmap, quantum_instance=backend,
                                shots=self.shots, seed_transpiler=self.seed, seed_simulator=self.seed)
                return qk, (lambda X, Y=None: qk.evaluate(x_vec=X, y_vec=Y))

        def fit(self, X, y):
            Xs = self.scaler.fit_transform(X)
            K = self._eval(Xs)                 # (n, n)
            self.clf.fit(K, y)
            self.X_train_ = Xs                 # keep for test-kernel
            return self

        def decision_function(self, X):
            Xs = self.scaler.transform(X)
            K = self._eval(Xs, self.X_train_)  # (m, n)
            return self.clf.decision_function(K)

        def predict(self, X):
            Xs = self.scaler.transform(X)
            K = self._eval(Xs, self.X_train_)
            return self.clf.predict(K)

    # ---- tiny usage example ----
    if __name__ == "__main__":
        # Toy binary dataset
        rng = np.random.default_rng(0)
        X = rng.normal(size=(120, 6))
        y = np.where((X[:, 0] + X[:, 1]**2 - X[:, 2]*X[:, 3]) > 0, 1, -1)

        Xtr, Xte = X[:80], X[80:]
        ytr, yte = y[:80], y[80:]

        model = QSVM(feature_dim=X.shape[1], reps=2, shots=2048, C=1.0).fit(Xtr, ytr)
        preds = model.predict(Xte)
        acc = (preds == yte).mean()
        print(f"Test accuracy: {acc:.3f}")
    '''

    # Show code + offer download
    if st.session_state.show_code:
        st.code(QSVM_CODE, language="python")
        st.download_button("Download qsvm.py", QSVM_CODE, file_name="qsvm.py", mime="text/x-python")

    st.markdown("""

<h3>2.Quantum Approximate Optimization Algorithm (QAOA)</h3> 
It’s a way to tackle hard “pick the best combo” problems (like splitting a network into two groups) using a small quantum circuit plus a classical optimizer.

You put qubits in a mix of all possibilities, then alternate two simple steps: one nudges the state toward better answers (the “cost” step), the other shuffles options (the “mixer”).

Each step has dial settings (angles). A classical computer tweaks those dials, checking how good the quantum circuit’s answers look.

After tuning, you measure the circuit many times and keep the best bitstring you see—that’s your approximate solution.

More alternations (depth p) can give better answers, but need longer, noisier circuits and more measurements.
""", unsafe_allow_html=True)
    st.image("img31.png",  use_container_width=True)
    if "show_code2" not in st.session_state:
        st.session_state.show_code2 = False
    if st.button("QAOA ",key="qaoa_code_button"):
        st.session_state.show_code2 = not st.session_state.show_code2

    QAOA_CODE = r'''
    # QSVM — minimal, version-robust implementation (Qiskit + scikit-learn)
    # pip install qiskit qiskit-aer qiskit-machine-learning scikit-learn numpy
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    class QSVM:
        def __init__(self, feature_dim, reps=2, shots=2048, C=1.0, seed=7):
            self.reps, self.shots, self.C, self.seed = reps, shots, C, seed
            self._kernel, self._eval = self._build_kernel(feature_dim)
            self.scaler = StandardScaler()
            self.X_train_ = None
            self.clf = SVC(kernel="precomputed", C=C)

        def _build_kernel(self, feature_dim):
            """Return (kernel_object, eval_fn) that computes Gram matrices."""
            try:
                # New API (preferred)
                from qiskit.circuit.library import ZZFeatureMap
                from qiskit_aer.primitives import Sampler
                from qiskit_machine_learning.kernels import FidelityQuantumKernel
                fmap = ZZFeatureMap(feature_dimension=feature_dim, reps=self.reps, entanglement="full")
                sampler = Sampler(options={"shots": self.shots, "seed": self.seed})
                qk = FidelityQuantumKernel(feature_map=fmap, sampler=sampler)
                return qk, (lambda X, Y=None: qk.evaluate(X, Y))
            except Exception:
                # Fallback (older API)
                from qiskit import BasicAer
                from qiskit.circuit.library import ZZFeatureMap
                from qiskit_machine_learning.kernels import QuantumKernel
                backend = BasicAer.get_backend("qasm_simulator")
                fmap = ZZFeatureMap(feature_dimension=feature_dim, reps=self.reps, entanglement="full")
                qk = QuantumKernel(feature_map=fmap, quantum_instance=backend,
                                shots=self.shots, seed_transpiler=self.seed, seed_simulator=self.seed)
                return qk, (lambda X, Y=None: qk.evaluate(x_vec=X, y_vec=Y))

        def fit(self, X, y):
            Xs = self.scaler.fit_transform(X)
            K = self._eval(Xs)                 # (n, n)
            self.clf.fit(K, y)
            self.X_train_ = Xs                 # keep for test-kernel
            return self

        def decision_function(self, X):
            Xs = self.scaler.transform(X)
            K = self._eval(Xs, self.X_train_)  # (m, n)
            return self.clf.decision_function(K)

        def predict(self, X):
            Xs = self.scaler.transform(X)
            K = self._eval(Xs, self.X_train_)
            return self.clf.predict(K)

    # ---- tiny usage example ----
    if __name__ == "__main__":
        # Toy binary dataset
        rng = np.random.default_rng(0)
        X = rng.normal(size=(120, 6))
        y = np.where((X[:, 0] + X[:, 1]**2 - X[:, 2]*X[:, 3]) > 0, 1, -1)

        Xtr, Xte = X[:80], X[80:]
        ytr, yte = y[:80], y[80:]

        model = QSVM(feature_dim=X.shape[1], reps=2, shots=2048, C=1.0).fit(Xtr, ytr)
        preds = model.predict(Xte)
        acc = (preds == yte).mean()
        print(f"Test accuracy: {acc:.3f}")
    '''
    if st.session_state.show_code2:
        st.code(QAOA_CODE, language="python")
    st.markdown("""

<h3>3.Variational Quantum Circuits (VQC)</h3> 
It’s a quantum-plus-classical method to find the lowest energy (ground state) of a system—think “find the most relaxed arrangement of a molecule.”

We build a small quantum circuit with dials (angles). Run it, measure the energy, then a classical optimizer tweaks the dials to make that energy smaller.

Repeat this loop until the energy won’t go lower—the circuit now encodes our best approximation of the ground state.

It’s used for quantum chemistry and materials (bond energies, reaction paths) because quantum circuits can represent rich quantum states.

Limits today: needs lots of measurements, depends on a good circuit design (“ansatz”), and is sensitive to noise on current hardware.
                
Start with qubits in ∣0⟩.Run a small quantum circuit with knobs (angles 𝜃) → it spits out measurement results.

A classical computer takes those results and scores them 
                
The classical optimizer then tweaks the knobs to try to make the score better.

Send the new 𝜃 back to the quantum circuit and repeat the loop.

When the score stops improving, the final θ and state are your best approximate solution (this is the core of VQE/QAOA-style algorithms).
""", unsafe_allow_html=True)
    st.image("img35.png",  use_container_width=True)
    if "show_code3" not in st.session_state:
        st.session_state.show_code3 = False
    if st.button("VQC ",key="vqc_code_button"):
        st.session_state.show_code3 = not st.session_state.show_code3

    VQC_CODE = r'''
    
    # VQC — Variational Quantum Classifier (minimal, version-robust)
# pip install qiskit qiskit-aer qiskit-machine-learning qiskit-algorithms scikit-learn numpy
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ------- Qiskit primitives (Aer if available, else basic) -------
try:
    from qiskit_aer.primitives import Estimator as AerEstimator
    EstimatorImpl = AerEstimator
except Exception:
    from qiskit.primitives import Estimator as BasicEstimator
    EstimatorImpl = BasicEstimator

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes

# ------- Qiskit ML: QNN + classifier wrapper -------
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

# ------- Optimizer (new and old import paths) -------
try:
    from qiskit_algorithms.optimizers import COBYLA
except Exception:
    try:
        from qiskit.algorithms.optimizers import COBYLA  # older path
    except Exception:
        COBYLA = None  # fallback handled below


@dataclass
class VQC:
    feature_dim: int
    ansatz_reps: int = 2
    shots: Optional[int] = None  # None = exact (if backend supports); else sampling
    seed: int = 7
    maxiter: int = 150

    def __post_init__(self):
        # Feature map encodes classical x into gates
        self.feature_map = ZZFeatureMap(feature_dimension=self.feature_dim, reps=1, entanglement="full")
        # Variational ansatz (trainable parameters)
        self.ansatz = RealAmplitudes(num_qubits=self.feature_dim, reps=self.ansatz_reps, entanglement="full")

        # Compose encoding + ansatz
        circuit = self.feature_map.compose(self.ansatz)

        # Estimator primitive (set shots if provided)
        if self.shots is None:
            self.estimator = EstimatorImpl(options={"seed": self.seed})
        else:
            self.estimator = EstimatorImpl(options={"seed": self.seed, "shots": self.shots})

        # QNN maps inputs -> real number in [-1, 1] (⟨Z⟩ on measured qubits internally)
        self.qnn = EstimatorQNN(
            circuit=circuit,
            input_params=list(self.feature_map.parameters),
            weight_params=list(self.ansatz.parameters),
            estimator=self.estimator,
        )

        # Optimizer (fallback to internal default if COBYLA unavailable)
        optimizer = COBYLA(maxiter=self.maxiter) if COBYLA is not None else None

        # Neural network classifier wrapper (binary cross-entropy by default)
        # one_hot=False so labels are {0,1}
        self.clf = NeuralNetworkClassifier(neural_network=self.qnn, optimizer=optimizer, one_hot=False)

        # Feature scaler
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X: (n_samples, feature_dim), y: (n_samples,) with labels in {0,1}
        """
        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs, y.astype(int))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        # NeuralNetworkClassifier outputs class probabilities if available
        try:
            proba = self.clf.predict_proba(Xs)  # shape (n, 2)
            return proba
        except Exception:
            # Fallback: convert raw outputs in [-1,1] to probabilities (p=(1+z)/2)
            raw = self.clf._neural_network.forward(Xs)  # (n, 1)
            p1 = (1.0 + raw.reshape(-1)) * 0.5
            p0 = 1.0 - p1
            return np.vstack([p0, p1]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        try:
            return self.clf.predict(Xs).astype(int)
        except Exception:
            # Fallback via proba
            proba = self.predict_proba(X)
            return (proba[:, 1] >= 0.5).astype(int)


# ---------------------- Tiny usage example ----------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, d = 200, 4  # samples, features (= qubits)
    X = rng.normal(size=(n, d))
    # Nonlinear binary label
    y = (np.sin(X[:, 0]) + X[:, 1] * X[:, 2] - 0.25 * X[:, 3] > 0).astype(int)

    # Train/test split
    n_tr = 140
    Xtr, Xte = X[:n_tr], X[n_tr:]
    ytr, yte = y[:n_tr], y[n_tr:]

    vqc = VQC(feature_dim=d, ansatz_reps=2, shots=None, seed=42, maxiter=120).fit(Xtr, ytr)
    preds = vqc.predict(Xte)
    acc = accuracy_score(yte, preds)
    print(f"Test accuracy: {acc:.3f}")

    # Show a few probabilities
    print("Proba (first 5):")
    print(np.round(vqc.predict_proba(Xte[:5]), 3))

    '''
    if st.session_state.show_code3:
        st.code(VQC_CODE, language="python")
    st.markdown("""

<h3>4.Quantum Reinforcement Learning (QRL)</h3> 
Regular RL = an agent tries actions, gets rewards, and learns a policy to maximize long-term reward.

In QRL, that policy is a small quantum circuit with tunable angles; the state is encoded on qubits.

Run the circuit → measurements give a probability over actions; sample one, act, get a reward.

A classical optimizer updates the circuit’s angles so high-reward actions become more likely (same hybrid loop as VQE/QAOA).

Quantum states can represent rich superpositions and enable different exploration/sampling tricks—promising but experimental today due to noise, few qubits, and measurement cost.
""", unsafe_allow_html=True)
    st.image("img34.jpg",  use_container_width=True)
    if "show_code4" not in st.session_state:
        st.session_state.show_code4 = False
    if st.button("QRL ",key="qrl_code_button"):
        st.session_state.show_code4 = not st.session_state.show_code4

    QRL_CODE = r'''
    
# Quantum Reinforcement Learning (QRL) — minimal policy-gradient (REINFORCE) with a quantum policy
# pip install qiskit qiskit-aer numpy
from __future__ import annotations
import numpy as np

# --------- Qiskit Sampler (Aer if available, else basic) ----------
try:
    from qiskit_aer.primitives import Sampler as AerSampler
    SamplerImpl = AerSampler
except Exception:
    from qiskit.primitives import Sampler as BasicSampler
    SamplerImpl = BasicSampler

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes

# ======================= Contextual Bandit Env =======================
class ContextualBandit:
    """
    Simple 2-action contextual bandit.
    Reward for action=1 is Bernoulli(sigmoid(w·x)), for action=0 it's 1 - that.
    """
    def __init__(self, n_features=2, seed=0):
        self.n_features = n_features
        self.rng = np.random.default_rng(seed)
        # Ground-truth mapping from context to optimal action
        self.w = self.rng.normal(size=n_features)
        self.bias = 0.0

    def sample_context(self):
        # Zero-mean normalized context
        x = self.rng.normal(size=self.n_features)
        x = x / (np.linalg.norm(x) + 1e-9)
        return x

    def reward(self, x, action: int) -> float:
        p1 = 1.0 / (1.0 + np.exp(-(self.w @ x + self.bias)))
        p = p1 if action == 1 else (1.0 - p1)
        return float(self.rng.random() < p)

# ======================= Quantum Policy (2 actions) =======================
class QuantumPolicy2A:
    """
    Bernoulli policy pi(a|x; theta) from a variational quantum circuit.
    - Encode context x with single-qubit RY rotations.
    - Trainable ansatz RealAmplitudes provides parameters theta.
    - Action probability p1 = Pr(qubit0 == 1).
    Gradient uses parameter-shift rule on theta.
    """
    def __init__(self, n_features=2, reps=1, shots=512, seed=7):
        assert n_features >= 1, "Need at least 1 feature/qubit"
        self.nq = n_features
        self.reps = reps
        self.shots = shots
        self.rng = np.random.default_rng(seed)

        # Parameters
        self.x_params = ParameterVector("x", self.nq)           # input angles
        self.theta_params = ParameterVector("th", self.nq * reps * 2)  # trainable
        # Build circuit: feature encoding (RY(x_i)), entangle (CZ ladder), variational layers
        qc = QuantumCircuit(self.nq)
        # Encode features
        for q in range(self.nq):
            qc.ry(self.x_params[q], q)
        # Variational layers
        p = 0
        for _ in range(reps):
            # Entangling CZ ladder
            for i in range(self.nq - 1):
                qc.cz(i, i + 1)
            # Single-qubit trainable rotations
            for q in range(self.nq):
                qc.ry(self.theta_params[p], q); p += 1
            for q in range(self.nq):
                qc.rz(self.theta_params[p], q); p += 1

        self.circuit = qc
        self.ordered_params = list(self.circuit.parameters)
        # Build maps to speed up binding
        self._is_x = {par: (par in set(self.x_params)) for par in self.ordered_params}
        # Sampler primitive
        self.sampler = SamplerImpl(options={"shots": shots, "seed": seed})
        # Initialize trainable parameters small
        self.theta = 0.1 * self.rng.normal(size=len(self.theta_params))

    def _values_vector(self, x, theta):
        # Create parameter vector in the circuit's parameter order
        vals = []
        ix = 0
        it = 0
        for par in self.ordered_params:
            if self._is_x[par]:
                vals.append(float(x[ix]))
                ix += 1
            else:
                vals.append(float(theta[it]))
                it += 1
        return [vals]

    def _p_action1(self, x, theta) -> float:
        """Probability of action=1 (qubit 0 measured to '1')."""
        result = self.sampler.run(
            circuits=[self.circuit],
            parameter_values=self._values_vector(x, theta)
        ).result()
        dist = result.quasi_dists[0]  # dict: bitstring -> probability
        # In Qiskit bitstrings are big-endian: last char corresponds to qubit 0
        p1 = sum(p for b, p in dist.items() if b[-1] == "1")
        # Clip to avoid numerical issues
        return float(np.clip(p1, 1e-6, 1 - 1e-6))

    def action_and_logprob(self, x):
        """Sample action ~ Bernoulli(p1) and return log pi(a|x)."""
        p1 = self._p_action1(x, self.theta)
        a = int(self.rng.random() < p1)
        logp = np.log(p1) if a == 1 else np.log(1.0 - p1)
        return a, p1, logp

    def grad_logprob(self, x, a, p1, shift=np.pi / 2) -> np.ndarray:
        """
        Parameter-shift gradient of log pi(a|x) wrt theta.
        For Bernoulli: d/dθ log π = (a - p1)/(p1*(1-p1)) * d/dθ p1
        """
        denom = p1 * (1.0 - p1)
        coeff = (a - p1) / max(denom, 1e-8)
        g = np.zeros_like(self.theta)
        for i in range(len(self.theta)):
            th_plus = self.theta.copy();  th_plus[i] += shift
            th_minus = self.theta.copy(); th_minus[i] -= shift
            p_plus = self._p_action1(x, th_plus)
            p_minus = self._p_action1(x, th_minus)
            dp = 0.5 * (p_plus - p_minus)
            g[i] = coeff * dp
        return g

# ======================= REINFORCE Training Loop =======================
def train_qrl(
    episodes=2000,
    n_features=2,
    lr=0.2,
    lr_decay=0.995,
    gamma=1.0,           # (not used in bandit; kept for API)
    reps=1,
    shots=512,
    seed=0
):
    env = ContextualBandit(n_features=n_features, seed=seed)
    policy = QuantumPolicy2A(n_features, reps=reps, shots=shots, seed=seed)
    baseline = 0.0      # value baseline (EMA of returns)
    beta = 0.01         # baseline update rate
    rng = np.random.default_rng(seed)

    avg_return = 0.0
    for ep in range(1, episodes + 1):
        x = env.sample_context()
        a, p1, logp = policy.action_and_logprob(x)
        r = env.reward(x, a)

        # Policy gradient with baseline
        g = policy.grad_logprob(x, a, p1)
        policy.theta += lr * (r - baseline) * g

        # Update baseline (EMA) and learning rate
        baseline = (1 - beta) * baseline + beta * r
        lr *= lr_decay

        avg_return = avg_return * 0.95 + 0.05 * r
        if ep % 200 == 0:
            # Evaluate greedy performance over 200 contexts
            eval_rewards = []
            for _ in range(200):
                xx = env.sample_context()
                p = policy._p_action1(xx, policy.theta)
                aa = int(p >= 0.5)
                eval_rewards.append(env.reward(xx, aa))
            print(f"Episode {ep:4d} | train avg R~{avg_return:.3f} | eval avg R={np.mean(eval_rewards):.3f}")

    return policy, env

# ======================= Run demo =======================
if __name__ == "__main__":
    policy, env = train_qrl(
        episodes=1500,    # increase for better convergence
        n_features=2,
        lr=0.25,
        reps=1,
        shots=512,
        seed=42
    )

    # Quick sanity check
    rng = np.random.default_rng(123)
    test_rewards = []
    for _ in range(500):
        x = env.sample_context()
        p = policy._p_action1(x, policy.theta)
        a = int(p >= 0.5)  # greedy
        test_rewards.append(env.reward(x, a))
    print(f"Greedy average reward over 500 trials: {np.mean(test_rewards):.3f}")


    '''
    if st.session_state.show_code4:
        st.code(QRL_CODE, language="python")
    st.markdown("""
<h2>3. Latency and Scalability</h2>
""", unsafe_allow_html=True)
    st.image("img19.png",  use_container_width=True)
    

if section == "Analysis":
    st.markdown("""
<h2>1. Evaluating the Need for QML in Trading</h2>
""", unsafe_allow_html=True)
    st.image("img2.png",  use_container_width=True)
    st.markdown("""
<h2>2.Hybrid System Architecture Performance</h2>
""", unsafe_allow_html=True)
    st.image("img2.png",  use_container_width=True)
    st.markdown("""
<h2>3.Latency and Scalability Trade-offs</h2>
""", unsafe_allow_html=True)
    st.image("img2.png",  use_container_width=True)
    st.markdown("""
<h2>4. Case Study Insights</h2>
""", unsafe_allow_html=True)
    st.image("img2.png",  use_container_width=True)
    st.markdown("""
<h2>5. Adoption Barriers</h2>
""", unsafe_allow_html=True)
    st.image("img2.png",  use_container_width=True)
    st.markdown("""
<h2>6. Synthesis</h2>
""", unsafe_allow_html=True)
    st.image("img2.png",  use_container_width=True)

if section == "Challenges and Limitations":
    st.markdown("""
        <h3>1. Quantum Hardware</h3>
        <h3>2. Latency in HFT</h3>
        <h3>3. Big Data Integration</h3>
        <h3>4. Algorithmic Limitations</h3>
        <h3>5. Regulation and Risk</h3>
        <h3>6. Economics and Talent</h3>
        <h3>7. Data Quality and Encoding Bottlenecks</h3>
        <h3>8. Standardization and Benchmarking</h3>
        <h3>9. Security and Cryptographic Risks</h3>
        <h3>10. Market Impact and Systemic Risk</h3>
    """, unsafe_allow_html=True)

if section == "Case Studies":
    st.markdown("""
<h2>1. Portfolio Optimization (QAOA)</h2>
""", unsafe_allow_html=True)
    st.image("img22.jpg",  use_container_width=True)
    st.markdown("""
<h2>2.Market Trend Detection </h2>
""", unsafe_allow_html=True)
    st.image("img29.png",  use_container_width=True)
    st.markdown("""
<h2>3.Order Execution </h2>
""", unsafe_allow_html=True)
    st.image("img25.webp",  use_container_width=True)
    st.markdown("""
<h2>4. Arbitrage Detection </h2>
""", unsafe_allow_html=True)
    st.image("img26.png",  use_container_width=True)
    st.markdown("""
<h2>5. Risk Analysis & Stress Testing </h2>
""", unsafe_allow_html=True)
    st.image("img27.png",  use_container_width=True)
    st.markdown("""
<h2>6. Derivatives Pricing (Quantum PDE Solvers)</h2>
""", unsafe_allow_html=True)
    st.image("img28.jpeg",  use_container_width=True)


if section == "Future Directions":
    st.markdown("""
<h2>1. Short-Term (0–5 Years): Hybrid & Quantum-Inspired Methods</h2>
""", unsafe_allow_html=True)
    
    st.markdown("""
<h2>2. Medium-Term (5–10 Years): Domain-Specific Applications (QSVM)</h2>
""", unsafe_allow_html=True)
   
    st.markdown("""
<h2>3. Long-Term (10+ Years): Quantum-Native Trading</h2>
""", unsafe_allow_html=True)
    st.image("img40.png",  use_container_width=True)
    st.image("img30.png",  use_container_width=True)


    
if section == "Conclusion":
    st.markdown("""
The convergence of Quantum Machine Learning (QML) and Big Data is a promising frontier for algorithmic and high-frequency trading (HFT). While classical ML has advanced prediction, risk management, and execution, it increasingly struggles with the scale and latency of high-dimensional, real-time markets.

This paper highlighted how QML can augment Big Data systems through hybrid architectures: classical platforms manage data scale, while quantum processors address optimization, classification, and execution challenges. Case studies with QAOA, QSVM, Quantum RL, and quantum annealing demonstrate early advantages in scalability and robustness, even under NISQ hardware constraints.
                
Significant barriers remain — limited qubits, noise, short coherence times, and issues of data encoding, interpretability, and compliance prevent full deployment in live HFT pipelines. Yet the roadmap is clear: short term, QML supports strategy design, risk analysis, and simulations; medium term, it may deliver quantum advantage in risk and anomaly detection; long term, fault-tolerant systems could enable quantum-native trading engines operating in real time.
                
Beyond technical hurdles, the economic and regulatory dimensions of QML adoption are equally important. The high cost of quantum infrastructure may initially restrict access to large financial institutions, potentially widening the competitive gap in global markets. Regulators will need to address concerns around transparency, auditability, and systemic risk, ensuring that quantum acceleration enhances — rather than destabilizes — financial stability.
                
For practitioners, QML offers both opportunity and caution. Early adopters can experiment with hybrid pipelines for portfolio optimization, fraud detection, and scenario modeling, gaining incremental advantages while building organizational expertise. However, premature reliance on noisy devices or black-box models could introduce new risks into already fragile systems. Responsible innovation will require careful balancing of ambition with rigor.
                
For researchers, the fusion of quantum computing and finance opens new avenues in algorithm design, hybrid orchestration, and cross-disciplinary methods. Quantum-enhanced kernels for time series, variational circuits for market simulation, and quantum-inspired heuristics for large-scale optimization all represent fertile ground for exploration. Collaboration between academia, industry, and regulators will be essential to validate progress and establish shared standards.
                
Ultimately, QML’s integration with Big Data is not just a computational upgrade but a paradigm shift in financial intelligence. It promises to redefine how strategies are designed, risks are managed, and markets are understood. Unlocking this vision will require sustained progress in hardware, algorithms, infrastructure, and governance — but the destination is clear: a self-optimizing, quantum-enhanced financial ecosystem capable of adapting with speed, scale, and foresight beyond classical limits.
""")
if section == "References":
    st.markdown("""
1.	Boucher, A., & Kondratyev, A. (2021). Quantum Machine Learning in Finance: From Theory to Applications. Journal of Financial Data Science, 3(4), 1–19.
2.	Rebentrost, P., Mohseni, M., & Lloyd, S. (2014). Quantum support vector machine for big data classification. Physical Review Letters, 113(13), 130503.
3.	Farhi, E., Goldstone, J., & Gutmann, S. (2014). A Quantum Approximate Optimization Algorithm. arXiv preprint arXiv:1411.4028.
4.	Preskill, J. (2018). Quantum Computing in the NISQ era and beyond. Quantum, 2, 79.
5.	Schuld, M., Sinayskiy, I., & Petruccione, F. (2015). An introduction to quantum machine learning. Contemporary Physics, 56(2), 172–185.
6.	Applications of Quantum Machine Learning for Quantitative Finance (arXiv, 2024)
7.	AI integration in financial services: a systematic review of 1989-2024 (Vuković et al., 2025)
8.	Exploring the application of quantum technologies to industrial and real-world use cases by Osaba, Villar-Rodriguez, Oregi (2025)
9.	Financial Fraud Detection: A Comparative Study of Quantum Machine Learning Models (Innan, Al-Zafar Khan, Bennai, 2023)
10.	AI and Financial Model Risk Management: Applications (S. Joshi, 2025)
11.	Quantum-Computing in Finance: Regulatory Readiness, Legal Gaps and the Future of Secure Tech Innovation


""")




