import marimo

__generated_with = "0.14.13"
app = marimo.App(
    width="medium",
    layout_file="layouts/valeos_inpatient_dashboard.grid.json",
)


@app.cell(hide_code=True)
def _(config, mo):
    mo.md(f"""# Valeos inpatient transplant dashboard: {config["site_name"]}""")
    return


@app.cell
def _(demographics_table):
    demographics_table
    return


@app.cell
def _(mo):
    mo.md(r"""## Transplant Volume Over Time""")
    return


@app.cell
def _(demographics_table, mo):
    organ_selected = mo.ui.radio(
        options=demographics_table.columns,
        label="Select organ type for volume analysis:",
        value="Overall"  # Set Overall as default
    )

    organ_selected
    return (organ_selected,)


@app.cell
def _(chart_display):
    chart_display
    return


@app.cell
def _(alt, config, mo, organ_selected, pd, valeos_transplant_df):
    def create_volume_chart_data(transplant_df, selected_organ):
        """Prepare data for Altair chart"""
        if transplant_df is None or selected_organ is None:
            return None, None

        # Convert transplant_date to datetime with UTC timezone
        work_df = transplant_df.copy()
        work_df['transplant_date'] = pd.to_datetime(work_df['transplant_date'], utc=True)
        work_df['month_year'] = work_df['transplant_date'].dt.to_period('M')

        # Get full date range to ensure zero months are included
        min_date = work_df['month_year'].min()
        max_date = work_df['month_year'].max()
        full_date_range = pd.period_range(start=min_date, end=max_date, freq='M')

        # Define consistent color scheme for all organ types
        organ_colors = {
            'kidney': '#1f77b4',    # Blue
            'liver': '#ff7f0e',     # Orange  
            'heart': '#2ca02c',     # Green
            'lung': '#d62728',      # Red
        }

        if selected_organ == 'Overall':
            # Create stacked bar chart data by organ type
            organ_monthly = work_df.groupby(['month_year', 'transplant_type']).size().reset_index(name='count')

            # Create complete date x organ grid to include zero months
            date_organ_grid = []
            for date in full_date_range:
                for organ in ['kidney', 'liver', 'heart', 'lung']:
                    if organ in work_df['transplant_type'].unique():
                        existing = organ_monthly[
                            (organ_monthly['month_year'] == date) & 
                            (organ_monthly['transplant_type'] == organ)
                        ]
                        if len(existing) > 0:
                            date_organ_grid.append({
                                'month_year': date.to_timestamp(),
                                'transplant_type': organ,
                                'count': existing['count'].iloc[0]
                            })
                        else:
                            date_organ_grid.append({
                                'month_year': date.to_timestamp(),
                                'transplant_type': organ,
                                'count': 0
                            })

            chart_data = pd.DataFrame(date_organ_grid)
            chart_data['month_year_str'] = chart_data['month_year'].dt.strftime('%m/%y')

            # Create Altair stacked bar chart
            chart = alt.Chart(chart_data).mark_bar().add_selection(
                alt.selection_interval()
            ).encode(
                x=alt.X('month_year:T', 
                       axis=alt.Axis(title='Month', 
                                   format='%m/%y',
                                   labelAngle=45)),
                y=alt.Y('count:Q', 
                       axis=alt.Axis(title='Number of Transplants'),
                       scale=alt.Scale(domain=[0, chart_data.groupby('month_year')['count'].sum().max() * 1.1])),
                color=alt.Color('transplant_type:N', 
                              scale=alt.Scale(domain=list(organ_colors.keys()),
                                            range=list(organ_colors.values())),
                              legend=alt.Legend(title="Organ Type")),
                order=alt.Order('transplant_type:N', sort=['kidney', 'liver', 'heart', 'lung']),
                tooltip=['month_year_str:N', 'transplant_type:N', 'count:Q']
            ).properties(
                width=600,
                height=400,
                title=f'Monthly Transplant Volume - {config["site_name"]}'
            )

        else:
            # Handle single organ or combinations
            if '+' in selected_organ:
                # For multi-organ combinations
                combo_organs = selected_organ.split('+')
                patient_organs = work_df.groupby('patient_id')['transplant_type'].apply(set).reset_index()
                combo_patients = patient_organs[
                    patient_organs['transplant_type'].apply(lambda x: set(combo_organs).issubset(x))
                ]['patient_id']
                filtered_df = work_df[work_df['patient_id'].isin(combo_patients)]
                bar_color = '#7f7f7f'  # Gray for combinations
            else:
                # Single organ type
                filtered_df = work_df[work_df['transplant_type'] == selected_organ]
                bar_color = organ_colors.get(selected_organ, '#1f77b4')

            if len(filtered_df) == 0:
                return None, f"No data available for {selected_organ}"

            # Aggregate by month and include zero months
            monthly_counts = filtered_df.groupby('month_year').size()
            monthly_counts = monthly_counts.reindex(full_date_range, fill_value=0)

            chart_data = pd.DataFrame({
                'month_year': monthly_counts.index.to_timestamp(),
                'count': monthly_counts.values
            })
            chart_data['month_year_str'] = chart_data['month_year'].dt.strftime('%m/%y')

            # Create Altair bar chart
            chart = alt.Chart(chart_data).mark_bar(
                color=bar_color,
                opacity=0.8
            ).add_selection(
                alt.selection_interval()
            ).encode(
                x=alt.X('month_year:T', 
                       axis=alt.Axis(title='Month', 
                                   format='%m/%y',
                                   labelAngle=45)),
                y=alt.Y('count:Q', 
                       axis=alt.Axis(title='Number of Transplants'),
                       scale=alt.Scale(domain=[0, chart_data['count'].max() * 1.1])),
                tooltip=['month_year_str:N', 'count:Q']
            ).properties(
                width=600,
                height=400,
                title=f'Monthly {selected_organ} Transplant Volume - {config["site_name"]}'
            )

        return chart, None

    # Get the selected organ value
    selected_organ = organ_selected.value if hasattr(organ_selected, 'value') else None

    # Create chart data
    volume_chart, error_msg = create_volume_chart_data(valeos_transplant_df, selected_organ)

    if error_msg:
        chart_display = mo.md(f"**{error_msg}**")
    elif volume_chart is not None:
        chart_display = mo.ui.altair_chart(volume_chart)
    else:
        chart_display = mo.md("**No data available for chart**")

    return (chart_display,)


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import altair as alt

    # Add utils directory relative to current script location
    CURRENT_DIR = Path(__file__).resolve().parent
    UTILS_DIR = CURRENT_DIR.parent / 'utils'
    sys.path.append(str(UTILS_DIR))

    from load_config import load_config
    return Path, alt, load_config, mo, pd


@app.cell
def _(load_config):
    # Load configuration
    config = load_config()
    print(f"Site: {config['site_name']}")
    print(f"Tables path: {config['tables_path']}")
    return (config,)


@app.cell
def _(mo):
    mo.md(r"""## Load Valeos Database""")
    return


@app.cell
def _(Path, config, pd):
    # Load Valeos tables
    tables_path = Path(config['tables_path'])
    valeos_dir = tables_path / 'Valeos'
    site_name = config['site_name']

    def load_valeos_table(table_name):
        """Load a Valeos table and return DataFrame or None"""
        file_path = valeos_dir / f"{site_name}_valeos_inpatient_{table_name}.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            print(f"Loaded {table_name} table: {len(df):,} records")
            return df
        else:
            print(f"{table_name} file not found: {file_path}")
            return None

    # Load core tables
    valeos_patient_df = load_valeos_table('patient')
    valeos_transplant_df = load_valeos_table('transplant')

    # Load additional clinical tables
    valeos_vitals_df = load_valeos_table('vitals')
    valeos_labs_df = load_valeos_table('labs')
    valeos_respiratory_df = load_valeos_table('respiratory_support')
    valeos_med_continuous_df = load_valeos_table('medication_admin_continuous')

    return (
        valeos_labs_df,
        valeos_med_continuous_df,
        valeos_patient_df,
        valeos_respiratory_df,
        valeos_transplant_df,
        valeos_vitals_df,
    )


@app.cell
def _(mo):
    mo.md(r"""## Clinical Data Summary""")
    return


@app.cell
def _(
    valeos_labs_df,
    valeos_med_continuous_df,
    valeos_respiratory_df,
    valeos_vitals_df,
):
    # Display summary of loaded clinical tables
    clinical_tables = {
        'Vitals': valeos_vitals_df,
        'Labs': valeos_labs_df,
        'Respiratory Support': valeos_respiratory_df,
        'Medication Admin (Continuous)': valeos_med_continuous_df
    }

    print("=== CLINICAL DATA SUMMARY ===")
    for table_name, df in clinical_tables.items():
        if df is not None:
            unique_patients = df['hospitalization_id'].nunique() if 'hospitalization_id' in df.columns else 'N/A'
            print(f"{table_name}: {len(df):,} records across {unique_patients} hospitalizations")
        else:
            print(f"{table_name}: No data available")

    return


@app.cell
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


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Patient Demographics by Organ Type

    **Privacy Notice**: This dashboard shows all demographic breakdowns including small cell counts. 
    When exporting or sharing results externally, cells with <10 patients should be suppressed 
    to protect patient privacy and prevent potential re-identification.
    """
    )
    return


@app.cell
def _(pd, valeos_patient_df, valeos_transplant_df):
    def create_demographics_table(patient_df, transplant_df):
        """Create demographics table with organ types as columns and demographic categories as rows"""
        if patient_df is None or transplant_df is None:
            return None

        # Merge patient and transplant data
        merged_df = pd.merge(patient_df, transplant_df, on='patient_id', how='inner')

        # Identify multi-organ recipients (>10 recipients for combinations)
        multi_organ_patients = merged_df.groupby('patient_id')['transplant_type'].apply(list).reset_index()
        multi_organ_patients['organ_combo'] = multi_organ_patients['transplant_type'].apply(
            lambda x: '+'.join(sorted(set(x))) if len(set(x)) > 1 else None
        )

        # Get all multi-organ combinations
        combo_counts = multi_organ_patients['organ_combo'].dropna().value_counts()
        all_combos = combo_counts.index.tolist()

        print(f"Multi-organ combinations found: {all_combos}")
        if len(combo_counts) > 0:
            print("Combination counts:", dict(combo_counts))

        # Create columns list: single organs + all multi-organ combos + overall
        single_organs = merged_df['transplant_type'].unique()
        all_columns = list(single_organs) + all_combos + ['Overall']

        # Initialize results dictionary
        demographics_data = {}

        # Calculate for single organ types
        for organ in single_organs:
            organ_data = merged_df[merged_df['transplant_type'] == organ]
            demographics_data[organ] = calculate_organ_demographics(organ_data)

        # Calculate for multi-organ combinations (if any)
        for combo in all_combos:
            combo_organs = combo.split('+')
            combo_patients = multi_organ_patients[multi_organ_patients['organ_combo'] == combo]['patient_id']
            combo_data = merged_df[merged_df['patient_id'].isin(combo_patients)]
            demographics_data[combo] = calculate_organ_demographics(combo_data)

        # Calculate overall
        demographics_data['Overall'] = calculate_organ_demographics(merged_df)

        # Convert to DataFrame with demographics as rows and organs as columns
        demographics_df = pd.DataFrame(demographics_data)

        return demographics_df

    def calculate_organ_demographics(group_df):
        """Calculate demographics for a group of patients"""
        stats = {}

        # Count
        stats['N'] = len(group_df.drop_duplicates('patient_id'))  # Unique patients

        # Get unique patients for demographic calculations
        unique_patients = group_df.drop_duplicates('patient_id')

        # Age at transplant - calculate from birth_date and transplant_date
        if 'birth_date' in unique_patients.columns and 'transplant_date' in unique_patients.columns:
            ages_at_transplant = []
            for _, patient in unique_patients.iterrows():
                if pd.notna(patient['birth_date']) and pd.notna(patient['transplant_date']):
                    birth = pd.to_datetime(patient['birth_date'], utc=True)
                    transplant = pd.to_datetime(patient['transplant_date'], utc=True)
                    age_at_transplant = (transplant - birth).days / 365.25
                    ages_at_transplant.append(age_at_transplant)

            if len(ages_at_transplant) > 0:
                ages_series = pd.Series(ages_at_transplant)
                stats['Age at transplant (mean ± SD)'] = f"{ages_series.mean():.1f} ± {ages_series.std():.1f}"
            else:
                stats['Age at transplant (mean ± SD)'] = "N/A"
        else:
            stats['Age at transplant (mean ± SD)'] = "N/A"

        # Sex
        if 'sex_category' in unique_patients.columns:
            sex_counts = unique_patients['sex_category'].value_counts()
            male_count = sex_counts.get('Male', 0)
            male_pct = (male_count / len(unique_patients)) * 100
            stats['Male (%)'] = f"{male_count} ({male_pct:.1f}%)"
        else:
            stats['Male (%)'] = "N/A"

        # Race categories - each as separate row
        if 'race_category' in unique_patients.columns:
            race_counts = unique_patients['race_category'].fillna('Unknown').value_counts()
            total_patients = len(unique_patients)
            for race in race_counts.index:
                count = race_counts[race]
                pct = (count / total_patients) * 100
                stats[f'Race - {race}'] = f"{count} ({pct:.1f}%)"

        # Ethnicity categories - each as separate row
        if 'ethnicity_category' in unique_patients.columns:
            ethnicity_counts = unique_patients['ethnicity_category'].fillna('Unknown').value_counts()
            total_patients = len(unique_patients)
            for ethnicity in ethnicity_counts.index:
                count = ethnicity_counts[ethnicity]
                pct = (count / total_patients) * 100
                stats[f'Ethnicity - {ethnicity}'] = f"{count} ({pct:.1f}%)"

        # Language categories - each as separate row
        if 'language_category' in unique_patients.columns:
            language_counts = unique_patients['language_category'].fillna('Unknown').value_counts()
            total_patients = len(unique_patients)
            for language in language_counts.index:
                count = language_counts[language]
                pct = (count / total_patients) * 100
                stats[f'Language - {language}'] = f"{count} ({pct:.1f}%)"

        return stats

    # Create the demographics table
    demographics_table = create_demographics_table(valeos_patient_df, valeos_transplant_df)

    return (demographics_table,)


if __name__ == "__main__":
    app.run()
