\# TwinGuard-X



Official implementation of \*\*TwinGuard-X: Reliability-Calibrated Digital Twin Evidence Fusion for Interpretable Building Anomaly Detection\*\*.



TwinGuard-X constructs an operational digital twin of building electricity consumption and detects abnormal operation through contextual twin-reality disagreement, residual dynamics, and multi-scale persistence evidence. Building-specific reliability calibration combines these evidence channels while preserving intrinsic interpretability.



\## Dataset



Experiments use the Building Data Genome Project 2 (BDG2).



The repository does not redistribute the original dataset. Please obtain BDG2 from its official source and configure the local dataset path before running the experiments.



\## Experimental Buildings



\- Robin\_public\_Cami

\- Bear\_public\_Valorie

\- Hog\_public\_Gerard



\## Anomaly Types



\- Spike

\- Persistent shift

\- Gradual drift

\- Stuck value

\- Contextual after-hours deviation



\## Repository Structure



```text

EIRT2026\_XDT\_Build/

├── scripts/

├── results/

│   ├── figures/

│   └── tables/

├── requirements.txt

├── .gitignore

└── README.md

