# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

For all tasks, use the CLIF schema files in the references/mCIDE directory and the clif_2.1.0.txt file to understand the data structure.

## Standard Workflow
0. Check the screenshots folder to see if any new screenshots have been added, if so analyze them before planning the task
1. First think through the problem, read the codebase for relevant files, and write a plan to todo.md.
2. The plan should have a list of todo items that you can check off as you complete them
3. Before you begin working, check in with me and I will verify the plan.
4. Then, begin working on the todo items, marking them as complete as you go.
5. Please every step of the way just give me a high level explanation of what changes you made
6. Make every task and code change you do as simple as possible. We want to avoid making any massive or complex changes. Every change should impact as little code as possible. Everything is about simplicity.
7. Finally, add a review section to the todo.md file with a summary of the changes you made and any other relevant information.



## Project Overview

This is a medical research project that identifies transplant recipient hospitalizations from the CLIF (Critical Care Literature Information Framework) database using CPT procedure codes. The project processes comprehensive adult inpatient data to extract transplant recipients across heart, lung, liver, and kidney transplants.

## Project Structure

- `config/` - Configuration files for site-specific settings
- `code/` - Analysis scripts and templates (Python/R)
- `utils/` - Utility functions including configuration loading
- `outlier-thresholds/` - CSV files defining outlier thresholds for data cleaning
- `output/` - Results directory with `final/` for deliverables and `intermediate/` for processing files

## Configuration

1. Copy `config/config_template.json` to `config/config.json`
2. Update `config.json` with site-specific settings:
   - `site_name`: Institution identifier
   - `tables_path`: Path to CLIF tables
   - `file_type`: Data format (csv/parquet/fst)
3. Use `utils/config.py` to load configuration in Python scripts

## Workflow

The project follows a three-step analysis workflow:

1. **Cohort Identification**: Apply inclusion/exclusion criteria using CPT codes for transplant procedures
2. **Quality Control**: Handle outliers using thresholds from `outlier-thresholds/` directory
3. **Analysis**: Generate final results and statistics

## Data Processing

- Input: CLIF 2.0 and 2.1 tables (patient, hospitalization, vitals, labs, medication_admin_*, respiratory_support, crrt_therapy, ecmo_mcs, patient_procedure)
- Output: Filtered parquet files in CLIF format plus transplant table with patient_id, transplant_type, recorded_dttm
- Cohort identified using specific CPT codes for different organ transplants

## Environment Setup

```bash
python3 -m venv .valeos_inpatient
source .valeos_inpatient/bin/activate
pip install -r requirements.txt
```

## File Naming Convention

Results should follow the pattern: `[RESULT_NAME]_[SITE_NAME]_[SYSTEM_TIME].pdf`

Use the config object to get the site name for consistent file naming across the project.