# SALA: Replication package
## Article Binary and multi-class classification of Self-Admitted Technical Debt: How far can we go?
## Authors
- Francesca Arcelli Fontana (Università degli studi di Milano–Bicocca)
- Juri Di Rocco (Università degli studi dell’Aquila)
- Davide Di Ruscio (Università degli studi dell’Aquila)
- Amleto Di Salle (Gran Sasso Science Institute)
- Phuong T. Nguyen (Università degli studi dell’Aquila)

### Abstract
**Context.** Aiming for a trade-off between short-term efficiency and long-term stability, software teams resort to sub-optimal solutions, neglecting the best software development practices. Such solutions may induce technical debt (TD), triggering maintenance issues. To facilitate future fixing, developers mark code with any issues using textual comments, resulting in Self-Admitted Technical Debt (SATD). Detecting SATD in source code is crucial since it helps programmers locate potentially erroneous snippets, allowing for suitable interventions, and improving code quality. There are two main types of SATD detection, i.e., binary classification and multi-class classification, grouping TD comments into SATD/Non-SATD categories, and multiple categories, respectively.

**Objective.** We attempt to understand to which extent state-of-the-art research has addressed the issue of detecting SATD, both binary and multi-class classification. Based on this investigation, we also propose a practical approach for the detection of SATD using Large Language Models (LLMs).

**Methods.** First, we conducted a literature review to understand to which extent the two types of classification have been tackled by existing research. Second, we developed SALA, a dual-purpose tool on top of Natural Language Processing (NLP) techniques and neural networks to deal with both types of classification. An empirical evaluation has been performed to compare SALA with state-of-the-art baselines.

**Results.** The literature review reveals that while binary classification has been well studied, multiclass classification has not received adequate attention. The empirical evaluation shows that SALA obtains a promising performance, and outperforms the baselines with respect to various quality metrics.

**Conclusion.** We conclude that more effort needs to be spent to tackle multi-class classification of SATD. To this end, LLMs hold the potential, albeit with more rigorous investigation on possible fine-tuning and prompt engineering strategies.

### Description
The replication package repository refers to the qualitative and quantitative analysis.

In particular, the `RQ1` folder contains the qualitative analysis. Meanwhile, the `RQ2` and `RQ3` folders contain the Python scripts and datasets used for binary and multi-class classification.  
