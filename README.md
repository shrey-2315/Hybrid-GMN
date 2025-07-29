# Structural SQL Similarity via Hybrid Graph Matching and AST-Guided Tree Edit Metrics

This repository combines two powerful modules for advanced SQL understanding and transformation:

* **AST Normalization Module** — Normalizes SQL structure using rule-based AST rewrites and tree edit distance.
* **Graph Matching Network (GMN) Hybrid** — Embeds SQL into hybrid AST graphs and compares them using GMNs.

---

## Contents

| Module               | Purpose                                                               |
| -------------------- | --------------------------------------------------------------------- |
| SQL Normalization    | Converts SQL to AST, applies rewriting rules, and computes similarity |
| SQL Graph Similarity | Converts SQL to hybrid ASTs and computes similarity via GMN           |

---

## Component Breakdown

### 1. AST Normalization and Similarity

Uses `sqlglot` to convert SQL queries into ASTs, applies normalization rules, and computes similarity via tree edit distance using `zss`.

---

### 2. Graph Matching Network (GMN) Hybrid

Uses hybrid ASTs and relational operator mapping to generate graph structures for GNN-based similarity comparison.

---

## Getting Started

```bash
pip install -r requirements.txt
```

---

## Example Use Cases

* Normalize SQL structure for consistent comparison
* Compute similarity scores using structure-aware models

---

## Credits

Built on top of:

* [sqlglot](https://github.com/tobymao/sqlglot)
* [zss (tree edit distance)](https://github.com/timtadh/zhang-shasha)
* [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)

---
