import marimo

__generated_with = "0.14.13"
app = marimo.App(width="columns")


@app.cell(column=0)
def _(mo, site_name):
    mo.md(f"""# Valeos inpatient transplant dashboard: {site_name}""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Patient Demographics by Organ Type

    **Privacy Notice**: This dashboard shows all demographic breakdowns, including small cell counts. 
    When exporting or sharing results externally, cells with <10 patients should be suppressed 
    to protect patient privacy and prevent potential re-identification.
    """
    )
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
    organ_selected = mo.ui.dropdown(
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


@app.cell(hide_code=True)
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
    import seaborn as sns

    # Add utils directory relative to current script location
    CURRENT_DIR = Path(__file__).resolve().parent
    UTILS_DIR = CURRENT_DIR.parent / 'utils'
    sys.path.append(str(UTILS_DIR))

    from load_config import load_config
    return Path, alt, load_config, mo, pd, plt


@app.cell
def _(load_config):
    # Load configuration
    config = load_config()
    print(f"Site: {config['site_name']}")
    print(f"Tables path: {config['tables_path']}")
    return (config,)


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
    valeos_hospitalization_df = load_valeos_table('hospitalization')
    valeos_vitals_df = load_valeos_table('vitals')
    valeos_labs_df = load_valeos_table('labs')
    valeos_respiratory_df = load_valeos_table('respiratory_support')
    valeos_med_continuous_df = load_valeos_table('medication_admin_continuous')

    return (
        site_name,
        valeos_hospitalization_df,
        valeos_labs_df,
        valeos_med_continuous_df,
        valeos_patient_df,
        valeos_respiratory_df,
        valeos_transplant_df,
        valeos_vitals_df,
    )


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

        # Race categories - top 3 only
        if 'race_category' in unique_patients.columns:
            race_counts = unique_patients['race_category'].fillna('Unknown').value_counts()
            total_patients = len(unique_patients)
            # Get top 3 race categories
            top_3_races = race_counts.head(3)
            for race in top_3_races.index:
                count = top_3_races[race]
                pct = (count / total_patients) * 100
                stats[f'Race - {race}'] = f"{count} ({pct:.1f}%)"

        # Hispanic ethnicity as separate line
        if 'ethnicity_category' in unique_patients.columns:
            ethnicity_counts = unique_patients['ethnicity_category'].fillna('Unknown').value_counts()
            total_patients = len(unique_patients)
            # Only show Hispanic if it exists
            if 'Hispanic' in ethnicity_counts.index:
                hispanic_count = ethnicity_counts['Hispanic']
                hispanic_pct = (hispanic_count / total_patients) * 100
                stats['Hispanic/Latino (%)'] = f"{hispanic_count} ({hispanic_pct:.1f}%)"
            else:
                stats['Hispanic/Latino (%)'] = "0 (0.0%)"

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


@app.cell(column=1, hide_code=True)
def _(mo, valeos_transplant_df):
    # create organ selector 
    mo.md(r"""## Patient Selector""")
    # Create organ
    organ_options = valeos_transplant_df['transplant_type'].unique().tolist()

    organ_selected_individual = mo.ui.dropdown(
        options=organ_options,
        label="Select organ type for patient selection:",
        value='heart'  # Set Overall as default
    )


    organ_selected_individual
    return (organ_selected_individual,)


@app.cell
def _(patient_selected):
    patient_selected
    return


@app.cell
def _(mo):
    mo.md(r"""## Individual Patient Vasoactive Course""")
    return


@app.cell
def _(patient_vasoactive_display):
    patient_vasoactive_display
    return


@app.cell
def _(mo):
    mo.md(r"""## Individual Patient Respiratory Support""")
    return


@app.cell
def _(patient_respiratory_display):
    patient_respiratory_display
    return


@app.cell
def _(mo):
    mo.md(r"""## Individual Patient Liver Function Timeline""")
    return


@app.cell
def _(patient_liver_display):
    patient_liver_display
    return


@app.cell
def _(
    mo,
    organ_selected,
    organ_selected_individual,
    valeos_hospitalization_df,
    valeos_med_continuous_df,
    valeos_transplant_df,
):
    # Get the selected organ value
    selected_organ_individual = organ_selected_individual.value if hasattr(organ_selected, 'value') else None

    # Filter patients based on the selected organ
    if selected_organ_individual and valeos_transplant_df is not None:
        filtered_patient_options = valeos_transplant_df[valeos_transplant_df['transplant_type'] == selected_organ_individual]['patient_id'].unique().tolist()

        # Find a patient with any vasoactive medication data
        vasoactive_meds = [
            'norepinephrine', 'epinephrine', 'phenylephrine', 'angiotensin', 
            'vasopressin', 'dopamine', 'dobutamine', 'milrinone', 'isoproterenol'
        ]
        default_patient = None
        if valeos_med_continuous_df is not None and valeos_hospitalization_df is not None:
            for pt_id in filtered_patient_options:
                # Get patient's hospitalizations
                patient_hosp_ids = valeos_hospitalization_df[valeos_hospitalization_df['patient_id'] == pt_id]['hospitalization_id'].unique()

                # Check if patient has any vasoactive medication data
                has_vasoactives = valeos_med_continuous_df[
                    (valeos_med_continuous_df['hospitalization_id'].isin(patient_hosp_ids)) & 
                    (valeos_med_continuous_df['med_category'].isin(vasoactive_meds))
                ].shape[0] > 0

                if has_vasoactives:
                    default_patient = pt_id
                    break

        # Create or update the patient selector with the filtered options
        patient_selected = mo.ui.dropdown(
            options=filtered_patient_options,
            label="Select patient ID:",
            value=default_patient if default_patient else (filtered_patient_options[0] if filtered_patient_options else None)
        )

    else:
        mo.md("**No transplant data available for patient selection**")
        patient_selected = None
    return (patient_selected,)


@app.cell(hide_code=True)
def _(pd, plt):
    def create_patient_vasoactive_chart(patient_id, med_df, transplant_df, hosp_df):
        """Create individual patient vasoactive course chart showing all vasoactives using proper CLIF data structure"""
        if patient_id is None or med_df is None or transplant_df is None or hosp_df is None:
            return None, "Missing required data"

        # Define all vasoactive medications from mCIDE schema
        vasoactive_meds = [
            'norepinephrine', 'epinephrine', 'phenylephrine', 'angiotensin', 
            'vasopressin', 'dopamine', 'dobutamine', 'milrinone', 'isoproterenol'
        ]

        # Load vasoactive dose ranges for relative scaling
        import os
        dose_ranges_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vasoactive_dose_ranges.csv')
        try:
            dose_ranges_df = pd.read_csv(dose_ranges_path)
            dose_ranges = dict(zip(dose_ranges_df['medication'], dose_ranges_df['typical_max_dose']))
        except FileNotFoundError:
            # Fallback ranges if CSV not found
            dose_ranges = {
                'norepinephrine': 1.0, 'epinephrine': 1.0, 'phenylephrine': 200.0,
                'angiotensin': 20.0, 'vasopressin': 0.04, 'dopamine': 20.0,
                'dobutamine': 20.0, 'milrinone': 0.75, 'isoproterenol': 0.2
            }

        # Get patient transplant info
        patient_transplant = transplant_df[transplant_df['patient_id'] == patient_id]
        if patient_transplant.empty:
            return None, f"No transplant data found for patient {patient_id}"

        transplant_date = pd.to_datetime(patient_transplant.iloc[0]['transplant_date'], utc=True)
        organ_type = patient_transplant.iloc[0]['transplant_type']

        # Get patient's hospitalizations to find hospitalization_ids (CLIF key structure)
        patient_hospitalizations = hosp_df[hosp_df['patient_id'] == patient_id].copy()
        if patient_hospitalizations.empty:
            return None, f"No hospitalization data found for patient {patient_id}"

        # Get hospitalization IDs for this patient
        patient_hosp_ids = patient_hospitalizations['hospitalization_id'].unique()

        # Filter medication data for all vasoactives using hospitalization_id (CLIF standard approach)
        patient_med_data = med_df[
            (med_df['hospitalization_id'].isin(patient_hosp_ids)) & 
            (med_df['med_category'].isin(vasoactive_meds))
        ].copy()

        if patient_med_data.empty:
            return None, f"No vasoactive medication data found for patient {patient_id}"

        # Find the transplant hospitalization (contains transplant_date)
        patient_hospitalizations['admission_dttm'] = pd.to_datetime(patient_hospitalizations['admission_dttm'], utc=True)
        patient_hospitalizations['discharge_dttm'] = pd.to_datetime(patient_hospitalizations['discharge_dttm'], utc=True)

        # Find hospitalization that contains the transplant date
        transplant_hosp = patient_hospitalizations[
            (patient_hospitalizations['admission_dttm'] <= transplant_date) &
            (patient_hospitalizations['discharge_dttm'] >= transplant_date)
        ]

        if transplant_hosp.empty:
            return None, f"Cannot find transplant hospitalization for patient {patient_id}"

        admission_date = transplant_hosp.iloc[0]['admission_dttm']
        discharge_date = transplant_hosp.iloc[0]['discharge_dttm']
        discharge_category = transplant_hosp.iloc[0]['discharge_category'] if 'discharge_category' in transplant_hosp.columns else 'Unknown'

        # Process medication dates and calculate days since admission
        patient_med_data['admin_dttm'] = pd.to_datetime(patient_med_data['admin_dttm'], utc=True)
        patient_med_data = patient_med_data.sort_values('admin_dttm')

        patient_med_data['days_since_admission'] = (
            patient_med_data['admin_dttm'] - admission_date
        ).dt.total_seconds() / (24 * 3600)

        # Calculate relative dose percentages
        patient_med_data['relative_dose_percent'] = patient_med_data.apply(
            lambda row: (row['med_dose'] / dose_ranges.get(row['med_category'], 1.0)) * 100 
            if pd.notna(row['med_dose']) and dose_ranges.get(row['med_category'], 1.0) > 0 else 0, 
            axis=1
        )

        transplant_days = (transplant_date - admission_date).total_seconds() / (24 * 3600)
        discharge_days = (discharge_date - admission_date).total_seconds() / (24 * 3600)

        # Create plot with relative dosing scale
        plt.style.use('default')  # Reset to default style
        fig, ax = plt.subplots(figsize=(12, 10))  # Increased height for legend below

        # Create consistent color mapping for all vasoactive medications
        vasoactive_color_map = {
            'norepinephrine': '#1f77b4',    # Blue
            'epinephrine': '#ff7f0e',       # Orange  
            'phenylephrine': '#2ca02c',     # Green
            'angiotensin': '#d62728',       # Red
            'vasopressin': '#9467bd',       # Purple
            'dopamine': '#8c564b',          # Brown
            'dobutamine': '#e377c2',        # Pink
            'milrinone': '#7f7f7f',         # Gray
            'isoproterenol': '#bcbd22'      # Olive
        }

        # Get unique vasoactives present in the data
        present_vasoactives = patient_med_data['med_category'].unique()

        # Plot each vasoactive medication using relative percentage (dots only, no lines)
        for med in present_vasoactives:
            med_data = patient_med_data[patient_med_data['med_category'] == med]
            if not med_data.empty:
                color = vasoactive_color_map.get(med, '#000000')  # Default to black if not found
                ax.scatter(med_data['days_since_admission'], med_data['relative_dose_percent'], 
                          color=color, s=50, alpha=0.8, label=med.title())

        # Add reference lines with discharge category in legend
        ax.axvline(x=0, color='blue', linestyle=':', linewidth=2, alpha=0.8, label='Admission (Day 0)')
        ax.axvline(x=transplant_days, color='green', linestyle='--', linewidth=3, alpha=0.8, 
                  label=f'Transplant (Day {transplant_days:.1f})')
        ax.axvline(x=discharge_days, color='orange', linestyle=':', linewidth=2, alpha=0.8, 
                  label=f'Discharge (Day {discharge_days:.1f}) to {discharge_category}')

        # Formatting
        ax.set_xlabel('Days Since Admission', fontsize=12)
        ax.set_ylabel('Relative Dose (% of Typical Maximum)', fontsize=12)
        ax.set_title(f'Vasoactive Medications - {organ_type.title()} Transplant - Patient {patient_id}', 
                    fontsize=14, fontweight='bold')

        # Set y-axis limits - cap at 100% unless patient data exceeds it
        max_dose_percent = patient_med_data['relative_dose_percent'].max()
        y_max = 100 if max_dose_percent <= 100 else max_dose_percent * 1.1
        ax.set_ylim(bottom=0, top=y_max)

        # Add main legend
        legend = ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

        # Grid for better readability
        ax.grid(True, alpha=0.3)

        # Set x-axis limits to admission and discharge
        ax.set_xlim(left=-0.5, right=discharge_days + 0.5)

        # Create dosing scale legend text
        dose_scale_text = "Dosing Scale (100% = max):\n"
        for med in present_vasoactives:
            if med in dose_ranges:
                dose_scale_text += f"• {med.title()}: {dose_ranges[med]} "
                # Add units from CSV if available
                try:
                    med_units = dose_ranges_df[dose_ranges_df['medication'] == med]['units'].iloc[0]
                    dose_scale_text += f"{med_units}\n"
                except:
                    dose_scale_text += "units\n"

        # Position the dosing scale annotation below the legend in upper right
        # Get legend position and place text box below it
        ax.text(0.98, 0.65, dose_scale_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", alpha=0.8))

        plt.tight_layout()

        return fig, None


    return (create_patient_vasoactive_chart,)


@app.cell(hide_code=True)
def _(pd, plt):
    def create_patient_respiratory_chart(patient_id, respiratory_df, transplant_df, hosp_df):
        """Create individual patient respiratory support chart using proper CLIF data structure"""
        if patient_id is None or respiratory_df is None or transplant_df is None or hosp_df is None:
            return None, "Missing required data"

        # Get patient transplant info
        patient_transplant = transplant_df[transplant_df['patient_id'] == patient_id]
        if patient_transplant.empty:
            return None, f"No transplant data found for patient {patient_id}"

        transplant_date = pd.to_datetime(patient_transplant.iloc[0]['transplant_date'], utc=True)
        organ_type = patient_transplant.iloc[0]['transplant_type']

        # Get patient's hospitalizations to find hospitalization_ids (CLIF key structure)
        patient_hospitalizations = hosp_df[hosp_df['patient_id'] == patient_id].copy()
        if patient_hospitalizations.empty:
            return None, f"No hospitalization data found for patient {patient_id}"

        # Get hospitalization IDs for this patient
        patient_hosp_ids = patient_hospitalizations['hospitalization_id'].unique()

        # Filter respiratory data using hospitalization_id (CLIF standard approach)
        patient_resp_data = respiratory_df[
            respiratory_df['hospitalization_id'].isin(patient_hosp_ids)
        ].copy()

        if patient_resp_data.empty:
            return None, f"No respiratory support data found for patient {patient_id}"

        # Find the transplant hospitalization (contains transplant_date)
        patient_hospitalizations['admission_dttm'] = pd.to_datetime(patient_hospitalizations['admission_dttm'], utc=True)
        patient_hospitalizations['discharge_dttm'] = pd.to_datetime(patient_hospitalizations['discharge_dttm'], utc=True)

        # Find hospitalization that contains the transplant date
        transplant_hosp = patient_hospitalizations[
            (patient_hospitalizations['admission_dttm'] <= transplant_date) &
            (patient_hospitalizations['discharge_dttm'] >= transplant_date)
        ]

        if transplant_hosp.empty:
            return None, f"Cannot find transplant hospitalization for patient {patient_id}"

        admission_date = transplant_hosp.iloc[0]['admission_dttm']
        discharge_date = transplant_hosp.iloc[0]['discharge_dttm']
        discharge_category = transplant_hosp.iloc[0]['discharge_category'] if 'discharge_category' in transplant_hosp.columns else 'Unknown'

        # Process respiratory dates and calculate days since admission
        patient_resp_data['recorded_dttm'] = pd.to_datetime(patient_resp_data['recorded_dttm'], utc=True)
        patient_resp_data = patient_resp_data.sort_values('recorded_dttm')

        patient_resp_data['days_since_admission'] = (
            patient_resp_data['recorded_dttm'] - admission_date
        ).dt.total_seconds() / (24 * 3600)

        transplant_days = (transplant_date - admission_date).total_seconds() / (24 * 3600)
        discharge_days = (discharge_date - admission_date).total_seconds() / (24 * 3600)

        # Create seaborn plot
        plt.style.use('default')  # Reset to default style
        fig, ax = plt.subplots(figsize=(10, 6))

        # Get unique device categories and assign colors
        device_categories = patient_resp_data['device_category'].unique()
        colors = plt.cm.Set3(range(len(device_categories)))
        device_color_map = dict(zip(device_categories, colors))

        # Plot respiratory support devices as horizontal bars
        for i, device in enumerate(device_categories):
            device_data = patient_resp_data[patient_resp_data['device_category'] == device]
            y_position = i

            # Create horizontal bars for each time period
            for _, row in device_data.iterrows():
                ax.barh(y_position, 0.1, left=row['days_since_admission'], 
                       color=device_color_map[device], alpha=0.7, height=0.8)

        # Add reference lines with discharge category in legend
        ax.axvline(x=0, color='blue', linestyle=':', linewidth=2, alpha=0.8, label='Admission (Day 0)')
        ax.axvline(x=transplant_days, color='green', linestyle='--', linewidth=3, alpha=0.8, 
                  label=f'Transplant (Day {transplant_days:.1f})')
        ax.axvline(x=discharge_days, color='orange', linestyle=':', linewidth=2, alpha=0.8, 
                  label=f'Discharge (Day {discharge_days:.1f}) to {discharge_category}')

        # Formatting
        ax.set_xlabel('Days Since Admission', fontsize=12)
        ax.set_ylabel('Respiratory Device Category', fontsize=12)
        ax.set_title(f'Respiratory Support - {organ_type.title()} Transplant - Patient {patient_id}', 
                    fontsize=14, fontweight='bold')

        # Set y-axis labels
        ax.set_yticks(range(len(device_categories)))
        ax.set_yticklabels(device_categories)

        # Add legend in better position
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

        # Grid for better readability
        ax.grid(True, alpha=0.3)

        # Set x-axis limits to admission and discharge
        ax.set_xlim(left=-0.5, right=discharge_days + 0.5)

        plt.tight_layout()

        return fig, None

    return (create_patient_respiratory_chart,)


@app.cell
def _(pd, plt):
    def create_patient_liver_function_chart(patient_id, labs_df, transplant_df, hosp_df):
        """Create individual patient liver function timeline showing MELD components"""
        if patient_id is None or labs_df is None or transplant_df is None or hosp_df is None:
            return None, "Missing required data"

        # Get patient transplant info
        patient_transplant = transplant_df[transplant_df['patient_id'] == patient_id]
        if patient_transplant.empty:
            return None, f"No transplant data found for patient {patient_id}"

        transplant_date = pd.to_datetime(patient_transplant.iloc[0]['transplant_date'], utc=True)
        organ_type = patient_transplant.iloc[0]['transplant_type']

        # Get patient's hospitalizations to find hospitalization_ids (CLIF key structure)
        patient_hospitalizations = hosp_df[hosp_df['patient_id'] == patient_id].copy()
        if patient_hospitalizations.empty:
            return None, f"No hospitalization data found for patient {patient_id}"

        # Get hospitalization IDs for this patient
        patient_hosp_ids = patient_hospitalizations['hospitalization_id'].unique()

        # Define MELD component lab categories
        meld_labs = {
            'INR': 'inr',
            'Total Bilirubin': 'bilirubin_total', 
            'Creatinine': 'creatinine',
            'Sodium': 'sodium'
        }

        # Filter lab data for MELD components using hospitalization_id (CLIF standard approach)
        patient_lab_data = labs_df[
            (labs_df['hospitalization_id'].isin(patient_hosp_ids)) & 
            (labs_df['lab_category'].isin(meld_labs.values()))
        ].copy()

        if patient_lab_data.empty:
            return None, f"No MELD component lab data found for patient {patient_id}"

        # Find the transplant hospitalization (contains transplant_date)
        patient_hospitalizations['admission_dttm'] = pd.to_datetime(patient_hospitalizations['admission_dttm'], utc=True)
        patient_hospitalizations['discharge_dttm'] = pd.to_datetime(patient_hospitalizations['discharge_dttm'], utc=True)

        # Find hospitalization that contains the transplant date
        transplant_hosp = patient_hospitalizations[
            (patient_hospitalizations['admission_dttm'] <= transplant_date) &
            (patient_hospitalizations['discharge_dttm'] >= transplant_date)
        ]

        if transplant_hosp.empty:
            return None, f"Cannot find transplant hospitalization for patient {patient_id}"

        admission_date = transplant_hosp.iloc[0]['admission_dttm']
        discharge_date = transplant_hosp.iloc[0]['discharge_dttm']
        discharge_category = transplant_hosp.iloc[0]['discharge_category'] if 'discharge_category' in transplant_hosp.columns else 'Unknown'

        # Process lab dates and calculate days since admission
        patient_lab_data['lab_result_dttm'] = pd.to_datetime(patient_lab_data['lab_result_dttm'], utc=True)
        patient_lab_data = patient_lab_data.sort_values('lab_result_dttm')

        patient_lab_data['days_since_admission'] = (
            patient_lab_data['lab_result_dttm'] - admission_date
        ).dt.total_seconds() / (24 * 3600)

        transplant_days = (transplant_date - admission_date).total_seconds() / (24 * 3600)
        discharge_days = (discharge_date - admission_date).total_seconds() / (24 * 3600)

        # Create subplots (4 rows, 1 column)
        plt.style.use('default')  # Reset to default style
        fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

        # Define colors and normal ranges for each lab
        lab_colors = {'inr': '#d62728', 'bilirubin_total': '#ff7f0e', 'creatinine': '#2ca02c', 'sodium': '#1f77b4'}
        normal_ranges = {
            'inr': (0.8, 1.2),
            'bilirubin_total': (0.2, 1.2),
            'creatinine': (0.7, 1.3), 
            'sodium': (135, 145)  # Note: CLIF uses mmol/L which is same numeric range as mEq/L
        }

        lab_titles = ['INR', 'Total Bilirubin (mg/dL)', 'Creatinine (mg/dL)', 'Sodium (mmol/L)']

        # Plot each MELD component
        for i, (display_name, lab_cat) in enumerate(meld_labs.items()):
            ax = axes[i]
            lab_data = patient_lab_data[patient_lab_data['lab_category'] == lab_cat]

            # Add normal range shading first
            if lab_cat in normal_ranges:
                normal_min, normal_max = normal_ranges[lab_cat]
                ax.axhspan(normal_min, normal_max, alpha=0.2, color=lab_colors[lab_cat])

            if not lab_data.empty:
                # Filter out non-numeric values and use lab_value_numeric
                numeric_data = lab_data.dropna(subset=['lab_value_numeric'])

                if not numeric_data.empty:
                    # Plot lab values as scatter points
                    ax.scatter(numeric_data['days_since_admission'], numeric_data['lab_value_numeric'], 
                              color=lab_colors[lab_cat], s=40, alpha=0.8)

                    # Set appropriate y-axis limits based on data and normal ranges
                    data_min = numeric_data['lab_value_numeric'].min()
                    data_max = numeric_data['lab_value_numeric'].max()

                    if lab_cat in normal_ranges:
                        norm_min, norm_max = normal_ranges[lab_cat]
                        y_min = min(data_min * 0.9, norm_min * 0.8)
                        y_max = max(data_max * 1.1, norm_max * 1.2)
                    else:
                        y_min = data_min * 0.9
                        y_max = data_max * 1.1

                    ax.set_ylim(y_min, y_max)

            # Add reference lines
            ax.axvline(x=0, color='blue', linestyle=':', linewidth=2, alpha=0.8)
            ax.axvline(x=transplant_days, color='green', linestyle='--', linewidth=3, alpha=0.8)
            ax.axvline(x=discharge_days, color='orange', linestyle=':', linewidth=2, alpha=0.8)

            # Formatting for each subplot
            ax.set_ylabel(lab_titles[i], fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(left=-0.5, right=discharge_days + 0.5)

            # Add legend only for first subplot
            if i == 0:
                # Create legend handles manually for proper labels
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], color='blue', linestyle=':', linewidth=2, label='Admission (Day 0)'),
                    Line2D([0], [0], color='green', linestyle='--', linewidth=3, label=f'Transplant (Day {transplant_days:.1f})'),
                    Line2D([0], [0], color='orange', linestyle=':', linewidth=2, label=f'Discharge (Day {discharge_days:.1f}) to {discharge_category}')
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        # Set shared x-axis label
        axes[-1].set_xlabel('Days Since Admission', fontsize=12)

        # Add overall title
        fig.suptitle(f'Liver Function (MELD Components) - {organ_type.title()} Transplant - Patient {patient_id}', 
                     fontsize=14, fontweight='bold')

        plt.tight_layout()

        return fig, None

    return (create_patient_liver_function_chart,)


@app.cell
def _(
    create_patient_vasoactive_chart,
    mo,
    patient_selected,
    valeos_hospitalization_df,
    valeos_med_continuous_df,
    valeos_transplant_df,
):

    # get patient ID from patient_selected
    patient_id = patient_selected.value if hasattr(patient_selected, 'value') else None

    # Create chart
    patient_chart, vaso_error_msg = create_patient_vasoactive_chart(
        patient_id, valeos_med_continuous_df, valeos_transplant_df, valeos_hospitalization_df
    )

    if vaso_error_msg:
        patient_vasoactive_display = mo.md(f"**{vaso_error_msg}**")
    elif patient_chart is not None:
        patient_vasoactive_display = mo.as_html(patient_chart)
    else:
        patient_vasoactive_display = mo.md("**No chart data available**")
    return patient_id, patient_vasoactive_display


@app.cell
def _(
    create_patient_respiratory_chart,
    mo,
    patient_id,
    valeos_hospitalization_df,
    valeos_respiratory_df,
    valeos_transplant_df,
):
    # Create respiratory chart
    patient_resp_chart, resp_error_msg = create_patient_respiratory_chart(
        patient_id, valeos_respiratory_df, valeos_transplant_df, valeos_hospitalization_df
    )

    if resp_error_msg:
        patient_respiratory_display = mo.md(f"**{resp_error_msg}**")
    elif patient_resp_chart is not None:
        patient_respiratory_display = mo.as_html(patient_resp_chart)
    else:
        patient_respiratory_display = mo.md("**No respiratory chart data available**")
    return (patient_respiratory_display,)


@app.cell
def _(
    create_patient_liver_function_chart,
    mo,
    patient_id,
    valeos_hospitalization_df,
    valeos_labs_df,
    valeos_transplant_df,
):
    # Create liver function chart
    patient_liver_chart, liver_error_msg = create_patient_liver_function_chart(
        patient_id, valeos_labs_df, valeos_transplant_df, valeos_hospitalization_df
    )

    if liver_error_msg:
        patient_liver_display = mo.md(f"**{liver_error_msg}**")
    elif patient_liver_chart is not None:
        patient_liver_display = mo.as_html(patient_liver_chart)
    else:
        patient_liver_display = mo.md("**No liver function chart data available**")
    return (patient_liver_display,)


if __name__ == "__main__":
    app.run()
