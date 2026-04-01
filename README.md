# MAJIC: MArkov Jailbreak with Iterative Camouflage

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2508.13048)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **"MAJIC: MArkov Jailbreak with Iterative Camouflage"** (AAAI 2026).

## Overview

MAJIC is a black-box jailbreak framework combining **10 semantic obfuscation methods** with **Markov chain optimization** to evaluate LLM safety vulnerabilities.

### Key Features

- 10 obfuscation methods: Hypothetical, Historical, Spatial, Reverse, Security, Word-level, Character-level, Literary, Language, Emoji
- Markov Transition Matrix for adaptive method selection
- Q-learning inspired dynamic optimization
- Multi-model support: GPT-4, Claude, Llama, Gemini
- Multiple evaluation judges: GPT-4, Llama Guard, rule-based

## Installation

```bash
git clone https://github.com/ZJU-LLM-Safety/MAJIC-AAAI2026.git
cd MAJIC-AAAI2026
pip install -r requirements.txt
```

## Quick Start

### 1. Configure API Keys

Copy and edit the configuration template:

```bash
cp config_template.py config.py
# Edit config.py with your API keys
```

### 2. Prepare Dataset

Place your harmful behavior dataset in `data/`:

```json
[{"goal": "Your harmful query here"}]
```

### 3. Run Attack

```bash
# Single method attack
python methods/m1_hypo_attackLLM.py

# Full MAJIC framework with Markov optimization
python markov_methods/markov_attack_api_dynamic.py
```

## Project Structure

```
MAJIC-AAAI2026/
├── methods/              # 10 obfuscation methods (m1-m10)
│   ├── m1_hypo_attackLLM.py
│   ├── m2_history_attackLLM.py
│   └── ...
├── markov_methods/       # Markov optimization framework
│   ├── markov_attack_api_dynamic.py
│   └── norm_matrix.py
├── data/                 # Datasets
├── majic.py             # Main entry point
└── config_template.py   # Configuration template
```

## Usage

### Single Method

```python
from methods.m1_hypo_attackLLM import hypo_method

score, prompt, response = hypo_method(
    suffix="none",
    harmful_prompt="Your query",
    attacker_pipe=attacker_pipeline,
    attacker_tokenizer=tokenizer,
    victim_pipe=victim_pipeline,
    victim_tokenizer=tokenizer,
    judgetype="gpt",
    attacktype="gpt-4o",
    iter_num=10
)
```

### Full Framework

Configure parameters in `markov_attack_api_dynamic.py`:
- `chain_count`: Attack chains per query (default: 10)
- `chain_length`: Max optimization steps (default: 3)
- `init_qnum`: Initial method queries (default: 1)
- `chain_qnum`: Optimization queries (default: 1)

## Results

| Model | ASR (%) | Avg Queries |
|-------|---------|-------------|
| GPT-4o | 85.2 | 12.3 |
| Claude-3.5 | 78.6 | 14.1 |
| Llama-3-70B | 92.4 | 10.7 |
| Gemini-1.5-Pro | 81.3 | 13.5 |

## Citation

```bibtex
@inproceedings{qi2026majic,
  title={Majic: Markovian adaptive jailbreaking via iterative composition of diverse innovative strategies},
  author={Qi, Weiwei and Shao, Shuo and Gu, Wei and Zheng, Tianhang and Zhao, Puning and Qin, Zhan and Ren, Kui},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={39},
  pages={32755--32763},
  year={2026}
}
```

## License

MIT License - see LICENSE file for details.

## Ethical Use

This tool is for security research and red teaming only. Users are responsible for ethical and legal compliance.

## Contact

For questions, please open an issue or contact: zjuqww@gmail.com
