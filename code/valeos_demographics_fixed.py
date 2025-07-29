import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import sys
    from pathlib import Path
    
    # Add utils directory relative to current script location
    CURRENT_DIR = Path(__file__).resolve().parent
    UTILS_DIR = CURRENT_DIR.parent / 'utils'
    sys.path.append(str(UTILS_DIR))
    
    from load_config import load_config
    return Path, load_config, mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Valeos inpatient transplant dashboard""")
    return


@app.cell
def _(load_config):
    # Load configuration
    config = load_config()
    print(f"Site: {config['site_name']}")
    print(f"Tables path: {config['tables_path']}")
    return (config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load Valeos Database""")
    return


@app.cell
def _(Path, config, pd):
    # Load Valeos patient and transplant tables
    tables_path = Path(config['tables_path'])
    valeos_dir = tables_path / 'Valeos'
    site_name = config['site_name']
    
    # Load patient table
    patient_file = valeos_dir / f"{site_name}_valeos_inpatient_patient.parquet"
    if patient_file.exists():
        valeos_patient_df = pd.read_parquet(patient_file)
        print(f"Loaded patient table: {len(valeos_patient_df)} patients")
    else:
        print(f"Patient file not found: {patient_file}")
        valeos_patient_df = None
    
    # Load transplant table
    transplant_file = valeos_dir / f"{site_name}_valeos_inpatient_transplant.parquet"
    if transplant_file.exists():
        valeos_transplant_df = pd.read_parquet(transplant_file)
        print(f"Loaded transplant table: {len(valeos_transplant_df)} transplants")
    else:
        print(f"Transplant file not found: {transplant_file}")
        valeos_transplant_df = None
    
    return valeos_patient_df, valeos_transplant_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Data Inspection""")
    return


@app.cell
def _(valeos_patient_df, valeos_transplant_df):
    # First, let's inspect the patient table columns
    if valeos_patient_df is not None:
        print("Patient table columns:", list(valeos_patient_df.columns))
        print("Patient table shape:", valeos_patient_df.shape)
        print("\nSample patient data:")
        print(valeos_patient_df.head())
    
    if valeos_transplant_df is not None:
        print("\nTransplant table columns:", list(valeos_transplant_df.columns))
        print("Transplant table shape:", valeos_transplant_df.shape)
    
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Patient Demographics by Organ Type""")
    return


@app.cell
def _(pd, valeos_patient_df, valeos_transplant_df):
    def create_demographics_table(patient_df, transplant_df):
        """Create demographics table with columns for each organ type using CLIF column names"""
        if patient_df is None or transplant_df is None:
            return None
        
        # Merge patient and transplant data
        merged_df = pd.merge(patient_df, transplant_df, on='patient_id', how='inner')
        
        # Define demographic calculations using CLIF standard column names
        def calculate_demographics(group_df):
            stats = {}
            
            # Count
            stats['N'] = len(group_df)
            
            # Age - calculate from birth_date (CLIF standard column)
            if 'birth_date' in group_df.columns:
                birth_dates = pd.to_datetime(group_df['birth_date'], errors='coerce').dropna()
                if len(birth_dates) > 0:
                    today = pd.Timestamp.now()
                    ages = (today - birth_dates).dt.days / 365.25
                    stats['Age (mean ± SD)'] = f"{ages.mean():.1f} ± {ages.std():.1f}"
                else:
                    stats['Age (mean ± SD)'] = "N/A"
            else:
                stats['Age (mean ± SD)'] = "N/A"
            
            # Sex - use CLIF sex_category column
            if 'sex_category' in group_df.columns:
                sex_counts = group_df['sex_category'].value_counts()
                male_count = sex_counts.get('Male', 0)
                male_pct = (male_count / len(group_df)) * 100
                stats['Male (%)'] = f"{male_count} ({male_pct:.1f}%)"
            else:
                stats['Male (%)'] = "N/A"
            
            # Race - use CLIF race_category column
            if 'race_category' in group_df.columns:
                race_counts = group_df['race_category'].dropna().value_counts()
                if len(race_counts) > 0:
                    top_race = race_counts.index[0]
                    top_race_pct = (race_counts.iloc[0] / len(group_df)) * 100
                    stats['Race (most common)'] = f"{top_race}: {race_counts.iloc[0]} ({top_race_pct:.1f}%)"
                else:
                    stats['Race (most common)'] = "N/A"
            else:
                stats['Race (most common)'] = "N/A"
            
            return pd.Series(stats)
        
        # Calculate demographics for each organ type - fix FutureWarning
        demographics_by_organ = merged_df.groupby('transplant_type', group_keys=False).apply(calculate_demographics)
        
        # Calculate overall demographics
        overall_stats = calculate_demographics(merged_df)
        overall_stats.name = 'Overall'
        
        # Combine into final table
        demographics_table = pd.concat([demographics_by_organ, overall_stats.to_frame().T])
        
        return demographics_table.T  # Transpose so organs are columns
    
    # Create the demographics table
    demographics_table_fixed = create_demographics_table(valeos_patient_df, valeos_transplant_df)
    
    return demographics_table_fixed, create_demographics_table


@app.cell
def _(demographics_table_fixed, mo):
    if demographics_table_fixed is not None:
        mo.md(f"""
        ### Demographics Summary (Using CLIF Standard Columns)
        
        {mo.as_html(demographics_table_fixed)}
        """)
    else:
        mo.md("**No data available for demographics table**")
    return


if __name__ == "__main__":
    app.run()