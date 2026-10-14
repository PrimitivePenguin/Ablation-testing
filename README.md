# Ablation-testing


### Goal
Learn ablation for open-weight LLMs

### 1. Setup

Requirements:
- Python
- uv

Generating UV files:
```
uv init --init
```
Declare python version
```
uv python pin 3.12.6
```
Generate UV files:
```
uv add torch transformers accelerate transformer_lens einops jaxtyping datasets matplotlib tqdm
```
Note: first uv add generates .venv and uv.lock

Dev only
```
uv add --dev jupyter pytest nbstripout
```
### 2. Data Collection

### 3. Selection

### Future

Create open-source ablation models so I can set up 