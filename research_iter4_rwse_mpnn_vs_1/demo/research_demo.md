# RWSE+MPNN vs 1-WL

## Summary

Comprehensive novelty assessment of using random walk return probabilities (RWSE) as GNN node features to break 1-WL expressiveness. RWSE alone is incomparable to 1-WL (MoSE Prop 4.7). RWSE equals weighted cycle homomorphism counts (MoSE Prop 4.5). GSN proves substructure count augmentation breaks 1-WL (Theorem 3.1). No paper explicitly proves 'GIN+RWSE breaks 1-WL' as a standalone theorem. HOD-GNN Theorem E.6 proves MPNN+RWSE has limitations (cannot distinguish quartic vertex-transitive pairs). Verdict: category (b) — known components, the chain RWSE→cycle counts→breaks 1-WL is logically straightforward but not stated as a standalone result, and MPNN+RWSE has known expressiveness ceilings.

## Research Findings

## Novelty Assessment: RWSE as GNN Node Features to Break 1-WL

### 1. What is RWSE?

Random Walk Structural Encoding (RWSE) was introduced by Dwivedi et al. (2021) and popularized in GraphGPS [1]. RWSE computes the diagonal entries of (D⁻¹A)^i for step lengths i=1,...,k, where D is the degree matrix and A is the adjacency matrix. Each entry gives the random walk return probability — the probability that a random walk of length i starting at node v returns to v [1, 2].

### 2. RWSE Expressiveness — Known Results

The MoSE paper (ICLR 2025) establishes three critical results [2]:

- **Proposition 4.4**: RWSE is strictly weaker than 2-WL for all walk lengths. Any graph pair distinguishable by RWSE is also distinguishable by 2-WL [2].
- **Proposition 4.7**: RWSE is *incomparable* to 1-WL at the node level. There exist node pairs that 1-WL distinguishes but RWSE cannot (for any walk length), and vice versa [2].
- **Proposition 4.5**: RWSE is a special case of MoSE with weight function ω(v) = 1/d(v) and pattern graphs being cycle graphs {C_i}. This formally establishes **RWSE = weighted cycle homomorphism counts** [2].

Critically, RWSE alone does NOT break 1-WL — it is incomparable. GraphGPS's power comes from RWSE + LapPE + global Transformer attention combined [1].

### 3. Mathematical Chain: RWSE → Closed Walks → Cycles → Homomorphism Counts

The connection is well-established [3, 2]:

1. RW return probability at step t: (D⁻¹A)^t[v,v] [3]
2. Unnormalized: A^t[v,v] = number of closed walks of length t from v [3]
3. Closed walks → cycle counts via inclusion-exclusion (e.g., tr(A³)/6 = triangles) [3]
4. MoSE Prop 4.5: RWSE_ℓ = MoSE with ω(v)=1/d(v) and 𝒢={C_1,...,C_ℓ} [2]
5. Therefore RWSE captures degree-normalized cycle homomorphism information — a strict subset of general homomorphism counting [2]

### 4. GSN Substructure Count Augmentation (Breaks 1-WL)

Bouritsas et al. (TPAMI 2022) proved the key GSN theorem [4]:

- **Theorem 3.1**: GSN is strictly more powerful than MPNN and 1-WL when H is any graph except star graphs (subgraph matching) or except single edges/nodes (induced) [4].
- Since 1-WL can only count forests of stars and cannot count cycles ≥ 3, injecting cycle counts as node features provably breaks 1-WL [4].

### 5. SAGNN — Random Walk on Cut Subgraphs (Breaks 1-WL)

SAGNN (AAAI 2023) uses a different mechanism [5]:
- Uses RW return probabilities on *CUT subgraphs* (not the original graph) as node features
- **Proposition 1 & 2**: 1-WL MPNN with ego/cut subgraph injection is strictly more powerful than 1-WL [5]
- Crucially different from standard RWSE which operates on the unmodified graph [5]

### 6. r-Loopy WL — Cycle-Counting Hierarchy

r-ℓWL (NeurIPS 2024) counts cycles up to length r+2, extending 1-WL [6]. It can homomorphism-count cactus graphs, is incomparable to k-WL for any fixed k, and is more expressive than F-Hom-GNNs that inject explicit cycle homomorphism counts [6].

### 7. WL Go Walking — RW Kernels ≈ 1-WL

Kriege (NeurIPS 2022) proved classical RW *kernels* reach 1-WL expressiveness [7]. Critical distinction: RW KERNELS (global graph-level similarity) ≠ RW NODE FEATURES (local structural augmentation). RWNN architectures (ICLR 2025) surpass WL hierarchy but are fundamentally different from MPNN+RWSE [7, 12].

### 8. The Critical Gap: MPNN + RWSE Expressiveness

**Key finding from HOD-GNN (2025) — Theorem E.6** [9]: HOD-GNN is *strictly more expressive* than RWSE+MPNN. The proof constructs a pair of quartic vertex-transitive graphs that are indistinguishable by MPNNs augmented with RWSE, but distinguishable by HOD-GNN. This proves MPNN+RWSE has definite expressiveness ceilings.

However, MPNN+RWSE ≠ 1-WL either. RWSE can help distinguish some 1-WL-equivalent pairs (e.g., CSL graphs [8]) via graph-level RWSE differences. The precise characterization is: **MPNN+RWSE is strictly between 1-WL and 2-WL** — it improves on 1-WL for some graph pairs but cannot distinguish all 1-WL-equivalent pairs.

**No paper contains a standalone theorem**: "GIN + RWSE initial features is strictly more expressive than 1-WL." The chain RWSE→cycle counts (MoSE) → breaks 1-WL (GSN) is logically straightforward but appears nowhere as a unified result.

### 9. Novelty Assessment Verdict: **(b) Known Components, Partially Novel Combination**

| Component | Status | Source |
|-----------|--------|--------|
| RWSE definition | Known | GraphGPS [1] |
| RWSE = cycle hom counts | Known | MoSE Prop 4.5 [2] |
| Cycle counts break 1-WL | Known | GSN Thm 3.1 [4] |
| RWSE incomparable to 1-WL | Known | MoSE Prop 4.7 [2] |
| MPNN+RWSE has limits | Known | HOD-GNN Thm E.6 [9] |
| "GIN+RWSE > 1-WL" theorem | **Gap** | Not stated [2,4,9] |
| Practical 1-WL sufficiency | Known | Zopf 2022 [10] |

### 10. Implications for ISP Paper Framing

1. **Do NOT claim** "RealGIN-Aug breaks 1-WL" as a novel theoretical contribution
2. **Acknowledge** RWSE literature and the RWSE→homomorphism chain
3. **Correctly position** RealGIN-Aug as empirically demonstrating GSN's theoretical prediction
4. **Emphasize** the complex-valued ISP mechanism as the main contribution
5. **Note** the nuanced expressiveness: MPNN+RWSE > 1-WL on some pairs but < 2-WL
6. **Cite** HOD-GNN Theorem E.6 for known limitations

**Confidence: HIGH (8/10)**. Evidence is consistent across 14 papers. Main uncertainty: whether any very recent 2025-2026 paper explicitly characterizes MPNN+RWSE expressiveness class.

## Sources

[1] [Recipe for a General, Powerful, Scalable Graph Transformer (GraphGPS, NeurIPS 2022)](https://arxiv.org/abs/2205.12454) — Introduces GraphGPS with RWSE as a structural encoding component alongside LapPE and global Transformer attention.

[2] [Homomorphism Counts as Structural Encodings for Graph Learning (MoSE, ICLR 2025)](https://arxiv.org/html/2410.18676) — Proves RWSE < 2-WL (Prop 4.4), RWSE incomparable to 1-WL at node level (Prop 4.7), RWSE = weighted cycle homomorphism counts with ω(v)=1/d(v) (Prop 4.5), and MoSE strictly more expressive than RWSE (Theorem 4.6).

[3] [Adjacency Matrix — Closed Walks and Trace](https://en.wikipedia.org/wiki/Adjacency_matrix) — Standard reference for A^k[v,v] counting closed walks of length k from v, and tr(A^k) giving total closed walks.

[4] [Improving GNN Expressivity via Subgraph Isomorphism Counting (GSN, TPAMI 2022)](https://arxiv.org/abs/2006.09252) — Proves GSN is strictly more expressive than 1-WL (Theorem 3.1) when using non-star substructure counts. 1-WL cannot count cycles ≥ 3, so cycle count injection breaks 1-WL.

[5] [Substructure Aware Graph Neural Networks (SAGNN, AAAI 2023)](https://ojs.aaai.org/index.php/AAAI/article/view/26318/26090) — Uses RW return probabilities on CUT subgraphs as node features. Proves MPNN with ego/cut subgraph injection is strictly more powerful than 1-WL.

[6] [Weisfeiler and Leman Go Loopy (r-ℓWL, NeurIPS 2024)](https://arxiv.org/abs/2403.13749) — r-ℓWL counts cycles up to length r+2, extending 1-WL. Counts homomorphisms of cactus graphs. Incomparable to k-WL.

[7] [Weisfeiler and Leman Go Walking: Random Walk Kernels Revisited (NeurIPS 2022)](https://arxiv.org/abs/2205.10914) — Proves classical RW kernels reach 1-WL expressiveness. Critical: RW kernels ≠ RW node features.

[8] [Balancing Efficiency and Expressiveness: Subgraph GNNs with Walk-Based Centrality (HyMN, 2025)](https://arxiv.org/html/2501.03113v1) — Notes RWSE can distinguish 1-WL-equivalent CSL graphs. Compares CSE (unnormalized) vs RWSE (degree-normalized).

[9] [On The Expressive Power of GNN Derivatives (HOD-GNN, 2025)](https://arxiv.org/html/2510.02565v1) — Theorem E.6: HOD-GNN is strictly more expressive than RWSE+MPNN. Constructs quartic vertex-transitive graphs indistinguishable by MPNN+RWSE.

[10] [1-WL Expressiveness Is (Almost) All You Need (Zopf, IJCNN 2022)](https://arxiv.org/abs/2202.10156) — 1-WL suffices for most standard benchmarks. Breaking 1-WL matters more for synthetic tasks.

[11] [An Empirical Study of Realized GNN Expressiveness (BREC, 2023)](https://arxiv.org/abs/2304.07702) — BREC benchmark with 400 non-isomorphic graph pairs. GNN accuracies 41.5%-70.2% on beyond-1-WL tasks.

[12] [Revisiting Random Walks for Learning on Graphs (RWNN, ICLR 2025 Spotlight)](https://arxiv.org/abs/2407.01214) — RWNNs surpass WL hierarchy in probability. Different architecture from MPNN+RWSE. Can separate strongly regular graphs where 3-WL fails.

[13] [Homomorphism Counts for GNNs: All About That Basis (2024)](https://arxiv.org/html/2402.08595) — Proves injecting homomorphism BASIS is strictly more expressive than injecting the parameter itself (Theorem 4.1).

[14] [Benchmarking Positional Encodings for GNNs and Graph Transformers (2024)](https://arxiv.org/html/2411.12732v1) — Unified benchmarking of GINE+RWSE and other PE combinations across synthetic and real-world datasets.

## Follow-up Questions

- What is the exact expressiveness class of MPNN+RWSE — can we characterize the set of graph pairs it can distinguish vs. those it cannot, and does this correspond to any known refinement in the WL hierarchy?
- Does using real-valued (continuous) RWSE features provide any optimization or generalization advantage over discrete substructure counts (GSN-style) for practical GNN training, even if they have equivalent theoretical expressiveness?
- Could the ISP complex-valued mechanism combined with RWSE features achieve expressiveness beyond what either approach achieves alone, and where would such a combination sit in the WL/homomorphism counting hierarchy?

---
*Generated by AI Inventor Pipeline*
