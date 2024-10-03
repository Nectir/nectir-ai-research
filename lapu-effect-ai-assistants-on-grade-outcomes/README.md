[![DOI](https://zenodo.org/badge/866249692.svg)](https://doi.org/10.5281/zenodo.13883503)

# An Experimental Study of the Effect of AI Course Assistants on Student Grade Outcomes

This repository contains the code used in the paper *An Experimental Study of the Effect of AI Course Assistants on Student Grade Outcomes* by George Hanshaw, Ethan Wicker, and Kristen Denlinger

This analysis assumes the data is stored in an Excel file in Google Drive, and a GCP service account has already been configured with the appropriate permissions.

## How to Run

1. Clone the repository

```bash
git clone https://github.com/Nectir/nectir-ai-research.git
cd nectir-ai-research
```

2. Set up the environment

```bash
conda create -n lapu-effect-ai-on-grades python=3.11
conda activate lapu-effect-ai-on-grades
pip install poetry==1.8.3
```

3. Change directories and install dependencies

```bash
cd lapu-effect-ai-assistants-on-grade-outcomes
poetry install
```

4. Configure the environment variables

Copy `env.example` to `.env` and fill in the necessary variables.

```bash
cp env.example .env
```

5. Configure `configs.yaml`

Copy `configs-template.yaml` to `configs.yaml` and add the Google Drive file ID.

```bash
cp configs-template.yaml configs.yaml
```

6. Create `data` and `figures` directories

```bash
mkdir data figures 
```

7. Run the Python scripts in order

These scripts are located in `lapu-effect-ai-assistants-on-grade-outcomes/src`.

- `00-descriptive-statistics.py`: Descriptive statistics of the data.
- `01-mann-whitney-u-test.py`: Mann-Whitney U test on GPA data.
- `02-permutation-test.py`: Permutation test on GPA differences.
- `03-psm.py`: Propensity score matching procedure.
- `04-psm-wilicoxon-signed-rank-test.py`: Wilcoxon Signed-Rank test on PSM pairs.
- `05-psm-permutation-test.py`: Permutation test on PSM pairs.

You may need to update you `PYTHONPATH`.  If so:

```bash
cd ..
export PYTHONPATH=$(pwd)
python lapu-effect-ai-assistants-on-grade-outcomes/src/00-descriptive-statistics.py
```

## Contact

For questions, please open an issue or contact Ethan Wicker <ethan.wicker@nectir.io>.
