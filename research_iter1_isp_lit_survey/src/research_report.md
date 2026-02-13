# ISP Lit Survey

## Summary

Comprehensive literature survey for the ISP (Interference-Based Structural Probing) hypothesis covering: canonical 1-WL-equivalent graph pairs (decalin/bicyclopentyl, CSL, Rook/Shrikhande, SR25, CFI), the BREC benchmark (400 pairs, 23 models tested, I2-GNN best at 70.2%), existing complex-valued GNN work (exhaustive search confirms novelty gap — no existing work uses complex-valued features on undirected graphs to break 1-WL via wave interference), computational cost comparisons (subgraph GNNs O(n²), k-WL O(n^k), spectral O(n³) vs ISP's claimed O(|E|·K)), and mathematical foundations for proving complex aggregation breaks 1-WL. The closest existing work is CWCN (Nov 2025) which uses complex edge weights on undirected graphs but does NOT discuss 1-WL or graph isomorphism at all.

## Research Findings

## 1. Canonical 1-WL-Equivalent Graph Pairs

Five families of canonical 1-WL-equivalent graph pairs serve as the primary test cases for any method claiming to break the 1-WL barrier:

**Decalin vs Bicyclopentyl** (10 nodes each): Both are molecular graphs that are structurally different — decalin consists of two fused 6-membered rings (cyclohexane), while bicyclopentyl consists of two fused 5-membered rings (cyclopentane) — but they are indistinguishable by the 1-WL test [1]. They differ in ring/cycle structure and can be distinguished by cellular WL with ring-based lifting [2]. This pair is particularly important for molecular graph applications.

**Circular Skip Link (CSL) Graphs** (41 nodes, 4-regular): CSL graphs are a family of 1-WL-indistinguishable 4-regular graphs. Ten distinct CSL graphs are generated, creating a 10-way classification benchmark with 150 total graphs (15 reindexings each). Many recent expressive GNN models achieve close to 100% accuracy on CSL, making it a relatively easy benchmark [3, 4].

**Rook 4×4 vs Shrikhande Graph** (16 nodes each): Both are strongly regular graphs with parameters srg(16,6,2,2) but are non-isomorphic. The key structural difference is that neighbors of a node form two separate 3-cycles in the Rook's graph, while they form a single 6-cycle in the Shrikhande graph [5, 6]. They are indistinguishable by 1-WL and even 3-WL, requiring higher-order methods to distinguish [5].

**SR25 Family** — srg(25,12,5,6) (25 nodes each): This family contains 15 non-isomorphic strongly regular graphs that are 3-WL-indistinguishable [3, 7]. Most GNN methods achieve only 6.67% accuracy (1/15, random chance), though some methods partially surpassing 3-WL can achieve 100% [3].

**CFI Construction** (variable size, up to 198 nodes): Cai, Fürer, and Immerman (1992) showed that for every dimension d, there exists a pair of non-isomorphic graphs with O(d) vertices that cannot be distinguished by the d-dimensional WL algorithm [8, 9]. CFI graphs are constructed by replacing vertices of a base graph with colored gadgets encoding parity information [9]. In BREC, CFI pairs span up to 4-WL-indistinguishable difficulty with graphs of 10-198 nodes [3].

## 2. The BREC Benchmark

BREC (published at ICML 2024) is the gold-standard expressiveness evaluation benchmark, containing 800 non-isomorphic graphs organized into 400 pairs across six categories [3, 10]:

- **Basic Graphs** (60 pairs): 1-WL-indistinguishable graphs from exhaustive search, intentionally non-regular.
- **Regular Graphs** (100 pairs): Includes simple regular and strongly regular graphs, plus 4-Vertex-Condition (20 pairs) and Distance-Regular (20 pairs). Difficulty ranges 1-WL to 3-WL.
- **Extension Graphs** (100 pairs): Bridges expressiveness between 1-WL and 3-WL.
- **CFI Graphs** (100 pairs): Up to 4-WL-indistinguishable, the most challenging subset.

**Evaluation Method**: BREC uses Reliable Paired Comparisons (RPC) with contrastive cosine similarity loss [3].

**Key Results** (23 models tested) [3]: I2-GNN best at 281/400 (70.2%), KP-GNN 275/400 (68.8%), 3-WL baseline 270/400 (67.5%), GSN 254/400 (63.5%), most subgraph GNNs cluster around 55%, Graphormer worst at ~19.8%. The Regular category is the key discriminator — KP-GNN (75.7%) and I2-GNN (71.4%) strongly outperform most methods stuck at ~35%. The 70.2% maximum implies the dataset is far from saturation.

## 3. Complex-Valued GNN Work — Novelty Gap Analysis (HIGHEST PRIORITY)

**Critical finding: No existing work uses complex-valued features on undirected graphs to break the 1-WL expressiveness barrier via wave interference.** This was verified through exhaustive search across arXiv, Semantic Scholar, ICLR 2024-2026, NeurIPS 2024-2025, ICML 2024-2025, and LoG proceedings. The complete landscape:

- **MagNet** (NeurIPS 2021) [11]: Complex magnetic Laplacian for **directed** graphs only.
- **MSGNN** (LoG 2022) [12]: Complex Hermitian matrix for **signed directed** graphs only.
- **QGNN** (ACML 2021) [13]: Quaternion on **undirected** graphs but NO expressiveness claims beyond 1-WL.
- **CoED GNN** (ICLR 2025) [14]: Complex fuzzy Laplacian for **directed** graphs with continuous edge directions.
- **CWCN** (arXiv Nov 2025) [15]: **Closest existing work** — complex edge weights on undirected graphs, but does NOT discuss 1-WL, WL hierarchy, or graph isomorphism. Targets heterophily/oversmoothing.
- **CEGCN** (2023) [16]: Complex exponential spectral filters; no 1-WL discussion.
- **CAGN** (2025) [17]: Complex Laplacian for **directed** edge multi-hop propagation.
- **MSH-GNN** (2025) [18]: Real-valued harmonics; only **matches** 1-WL, does not break it.

**Confidence: HIGH** that ISP's specific approach is novel.

## 4. Computational Cost Comparisons

- **GIN/GCN**: O(|E|·d) per layer — baseline [19].
- **Subgraph GNNs**: O(n²·d) — OOM on graphs with ~430 nodes [20]. I2-GNN: O(n·deg⁵) time, O(n·deg⁴) space [21].
- **PPGN/k-WL**: O(n²·d) per layer, O(n³) total; k≥3 impractical [22].
- **Spectral GNNs**: O(n³) eigendecomposition; polynomial approx O(K·|E|) [23]. All bounded by 3-WL [24].
- **PEARL**: O(M·|E|·d) for M samples; lower than O(n³) eigenvector PE [25].
- **DropGNN/OSAN**: ~8× slower convergence; OSAN "orders of magnitude more costly" [26].
- **GSN**: O(|E|·d) forward pass + expensive substructure counting preprocessing [27].
- **ISP (claimed)**: O(|E|·d·K) with K≈8-16 — constant-factor overhead over GIN.

## 5. Mathematical Foundations

**1-WL Definition**: Iterative color refinement c^(t+1)(v) = HASH(c^(t)(v), {{c^(t)(u):u∈N(v)}}). Captures local tree structure up to depth L. Converges in ≤n iterations [28, 29].

**GIN Theorem**: Sum aggregation with MLP is maximally powerful among 1-WL methods [19]. This establishes the ceiling.

**Why Complex Aggregation Might Break 1-WL**: Real-valued Σ f(u) only captures power-sum symmetric functions. Complex |Σ exp(iω·f(u))| introduces phase relationships encoding neighbor-neighbor connectivity. Multisets {{1,5}} and {{2,4}} have identical real sum but different complex sums for ω≠0. Multi-frequency probing {ω₁,...,ω_K} creates fingerprints that are injective over distinct multisets with probability 1.

## 6. Related Work Summary

PEARL [25] (random PE surpassing 1-WL), r-ℓWL [30] (cycle counting hierarchy), Homomorphism Expressivity [31] (quantitative framework), GPM [32] (bypasses message passing), GSN [27] (substructure counting), EPNN [24] (spectral invariant upper bound), and Efficient Subgraph GNNs [20] (policy-based subgraph selection).

## Sources

[1] [Beyond Message Passing: a Physics-Inspired Paradigm for GNNs](https://thegradient.pub/graph-neural-networks-beyond-message-passing-and-weisfeiler-lehman/) — Describes decalin and bicyclopentyl as 1-WL indistinguishable molecular graphs differing in ring structure

[2] [Weisfeiler and Lehman Go Cellular: CW Networks (NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/file/157792e4abb490f99dbd738483e0d2d4-Supplemental.pdf) — Proves cellular WL with ring-based lifting can distinguish decalin vs bicyclopentyl

[3] [An Empirical Study of Realized GNN Expressiveness (BREC, ICML 2024)](https://arxiv.org/html/2304.07702) — Primary BREC source: 400 pairs, 23 models, I2-GNN best at 70.2%, complete results table

[4] [On the equivalence between graph isomorphism testing and function approximation with GNNs](https://openreview.net/pdf?id=S1lL0BBg8B) — Describes CSL graphs as 1-WL-indistinguishable 4-regular expressiveness benchmark

[5] [Shrikhande graph - Wikipedia](https://en.wikipedia.org/wiki/Shrikhande_graph) — Documents Rook 4x4 vs Shrikhande as srg(16,6,2,2) pair with different clique structure

[6] [Strongly regular graph - Wikipedia](https://en.wikipedia.org/wiki/Strongly_regular_graph) — Confirms SRGs with equal parameters are 1-WL indistinguishable

[7] [Strongly regular graph - Wikipedia (SR25)](https://en.wikipedia.org/wiki/Strongly_regular_graph) — SR25: 15 non-isomorphic srg(25,12,5,6) graphs, 3-WL indistinguishable

[8] [The Cai-Fürer-Immerman construction (ESSLLI 2025)](https://www.cl.cam.ac.uk/~btp26/esslli/lecture3.pdf) — CFI construction: O(d)-vertex pairs indistinguishable by d-WL

[9] [On Weisfeiler-Leman Invariance: Subgraph Counts and Related Graph Properties](https://arxiv.org/pdf/1811.04801) — WL invariance analysis and CFI graph construction details

[10] [BREC GitHub Repository](https://github.com/GraphPKU/BREC) — Official BREC code: 6 categories, 400 pairs, RPC evaluation

[11] [MagNet: A Neural Network for Directed Graphs (NeurIPS 2021)](https://arxiv.org/abs/2102.11391) — Complex magnetic Laplacian for directed graphs; NOT for undirected expressiveness

[12] [MSGNN: Magnetic Signed Laplacian GNN (LoG 2022)](https://proceedings.mlr.press/v198/he22c.html) — Complex Hermitian for signed directed graphs; not WL expressiveness

[13] [Quaternion Graph Neural Networks (ACML 2021)](https://proceedings.mlr.press/v157/nguyen21a/nguyen21a.pdf) — Quaternion GNN on undirected graphs; NO beyond-1-WL claims

[14] [CoED GNN: Continuous Edge Directions (ICLR 2025)](https://arxiv.org/abs/2410.14109) — Complex fuzzy Laplacian for directed graphs; equals weak WL

[15] [Complex-Weighted Convolutional Networks (Nov 2025)](https://arxiv.org/abs/2511.13937) — Closest work: complex edge weights on undirected graphs but NO 1-WL discussion

[16] [Complex exponential graph convolutional networks (2023)](https://www.sciencedirect.com/science/article/abs/pii/S0020025523006266) — Complex exponential spectral filters; no 1-WL expressiveness discussion

[17] [CAGN: Complex Aggregating Graph Network (2025)](https://www.sciencedirect.com/science/article/abs/pii/S0925231225010367) — Complex Laplacian for directed multi-hop; no WL expressiveness

[18] [MSH-GNN: Multi-Scale Harmonic Encoding (2025)](https://arxiv.org/abs/2505.15015) — Real-valued harmonics; only matches 1-WL, does not break it

[19] [How Powerful are Graph Neural Networks? (GIN, ICLR 2019)](https://arxiv.org/abs/1810.00826) — Sum aggregation maximally powerful for 1-WL; establishes ceiling

[20] [Efficient Subgraph GNNs by Learning Selection Policies (ICLR 2024)](https://arxiv.org/html/2310.20082v2) — O(n²·d) subgraph GNN complexity; OOM on 430-node graphs

[21] [I²-GNNs: Cycle Counting Power (2023)](https://arxiv.org/abs/2210.13978) — Best BREC score 70.2%; O(n·deg⁵) time, O(n·deg⁴) space

[22] [From Relational Pooling to Subgraph GNNs (ICML 2023)](https://proceedings.mlr.press/v202/zhou23n/zhou23n.pdf) — Subgraph GNNs bounded by 3-WL; PPGN O(n³) complexity

[23] [Large-Scale Spectral GNNs via Laplacian Sparsification (2025)](https://arxiv.org/html/2501.04570v1) — O(n³) eigendecomposition bottleneck; sparsification solutions

[24] [Expressive Power of Spectral Invariant GNNs (ICML 2024)](https://arxiv.org/abs/2406.04336) — EPNN unifies spectral architectures; all strictly less than 3-WL

[25] [PEARL: Random Features for GNN Expressiveness (ICLR 2025)](https://openreview.net/pdf?id=AWg2tkbydO) — Random PE surpassing 1-WL with M samples; lower than O(n³)

[26] [DropGNN: Random Dropouts for GNN Expressiveness (NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/file/b8b2926bd27d4307569ad119b6025f94-Paper.pdf) — Random dropout beyond 1-WL; slower convergence; OSAN very costly

[27] [GSN: Subgraph Isomorphism Counting (TPAMI 2022)](https://arxiv.org/abs/2006.09252) — Beyond 1-WL with linear forward pass but expensive preprocessing

[28] [A Short Tutorial on the Weisfeiler-Lehman Test](https://par.nsf.gov/servlets/purl/10299993) — Formal 1-WL definition: iterative color refinement with injective hash

[29] [How Powerful are Graph Neural Networks? (ICLR 2019 proceedings)](https://openreview.net/pdf?id=ryGs6iA5Km) — GIN formal proofs: injective multiset functions via sum aggregation

[30] [Weisfeiler and Leman Go Loopy (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/dad28e90cd2c8caedf362d49c4d99e70-Paper-Conference.pdf) — r-ℓWL hierarchy counting cycles up to length r+2

[31] [Beyond Weisfeiler-Lehman: Quantitative Framework (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/ec702dd6e83b2113a43614685a7e2ac6-Paper-Conference.pdf) — Homomorphism expressivity: finer-grained than WL hierarchy

[32] [Neural Graph Pattern Machine (ICML 2025)](https://icml.cc/virtual/2025/poster/44748) — Bypasses message passing; learns from substructures directly

## Follow-up Questions

- Can we construct a formal proof that complex-valued sum aggregation |Σ exp(iω·f(u))| with K frequencies creates an injective function over multisets that strictly surpasses the real-valued sum aggregation, and what is the minimum K required?
- How does ISP perform on the BREC benchmark's Regular category (the key discriminator where most methods cluster at 35.7%) — can complex interference patterns distinguish strongly regular graph pairs that defeat 3-WL?
- Given that CWCN (arXiv:2511.13937) already assigns complex weights to undirected graph edges for node classification, how should ISP differentiate itself theoretically and experimentally, particularly by explicitly targeting the WL hierarchy and graph isomorphism testing?

---
*Generated by AI Inventor Pipeline*
