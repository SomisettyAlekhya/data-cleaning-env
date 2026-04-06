---
title: Data Cleaning Environment
colorFrom: red
colorTo: red
sdk: streamlit
app_file: app.py
---

# AI Data Cleaning Environment

## Overview

This project implements a **real-world OpenEnv environment** where an AI agent learns to clean messy datasets.

The environment simulates common data preprocessing tasks performed before training machine learning models.

It follows the **OpenEnv specification**:
- reset()
- step(action)
- state()

---

## Problem Domain

Data cleaning is a critical real-world task:
- Handling missing values
- Converting data types
- Removing duplicates
- Normalizing features

This environment allows an agent to learn these tasks step-by-step.

---

## Features

- Typed models (Pydantic)
- step / reset / state API
- 3 tasks: easy, medium, hard
- Reward shaping (0–1)
- Manual + AI agent cleaning
- Hugging Face deployment ready
- Docker support
- Realistic noisy datasets

---

## Actions Supported

| Action | Description |
|--------|------------|
| remove_duplicates | Removes duplicate rows |
| fill_missing_mean | Fill missing values with mean |
| fill_missing_median | Fill missing values with median |
| fill_missing_mode | Fill missing values with mode |
| convert_to_numeric | Convert column to numeric |
| normalize_column | Normalize numeric column (0–1) |
| normalize_text | Lowercase + trim text |

---

## Important Instruction (VERY IMPORTANT)

### Median & Mean Operations

Before applying:
- `fill_missing_median`
- `fill_missing_mean`
- `fill_missing_mode`

The column **must be numeric**

### Correct Sequence:

1. Convert column to numeric  
2. Then apply mean/median  


### Manual Operation:
It updates the above values dynamically when you apply step-by-step.

### Agent:
It just cleanes the table and display the table.

### Grader:
Grader evaluates the quality of your dataset cleaning. It considers:

- Remaining missing values (lower is better)
- Presence of duplicates (should be 0)
- Correct numeric conversions
- Proper normalization of numeric features
- The final score is between 0–1.

### Example:
```text
convert_to_numeric → age  
fill_missing_median → age

## Run
pip install -r requirements.txt
streamlit run app.py

## Inference
python inference.py


