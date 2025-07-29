import marimo

__generated_with = "0.14.13"
app = marimo.App(
    width="columns",
    layout_file="layouts/valeos_inpatient_dashboard.grid.json",
)


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


@app.cell(column=1)
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
def _(vasoactive_selected):
    vasoactive_selected
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
def _(mo, organ_selected, organ_selected_individual, valeos_transplant_df):
    # Get the selected organ value
    selected_organ_individual = organ_selected_individual.value if hasattr(organ_selected, 'value') else None

    # Filter patients based on the selected organ
    if selected_organ_individual and valeos_transplant_df is not None:
        filtered_patient_options = valeos_transplant_df[valeos_transplant_df['transplant_type'] == selected_organ_individual]['patient_id'].unique().tolist()

        # Create or update the patient selector with the filtered options
        patient_selected = mo.ui.dropdown(
            options=filtered_patient_options,
            label="Select patient ID:",
            value=filtered_patient_options[0] if filtered_patient_options else None
        )

        # Create vasoactive drug selector
        vasoactive_selected = mo.ui.radio(
            options=['norepinephrine', 'dobutamine'],
            label="Select vasoactive drug:",
            value='dobutamine'
        )
    else:
        mo.md("**No transplant data available for patient selection**")
        patient_selected = None
        vasoactive_selected = None
    return patient_selected, vasoactive_selected


@app.cell
def _(alt, pd):
    def create_patient_vasoactive_chart(patient_id, vasoactive_drug, med_df, transplant_df, hosp_df):
        """Create individual patient vasoactive course chart using proper CLIF data structure"""
        if patient_id is None or vasoactive_drug is None or med_df is None or transplant_df is None or hosp_df is None:
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

        # Filter medication data using hospitalization_id (CLIF standard approach)
        patient_med_data = med_df[
            (med_df['hospitalization_id'].isin(patient_hosp_ids)) & 
            (med_df['med_category'] == vasoactive_drug)
        ].copy()

        if patient_med_data.empty:
            return None, f"No {vasoactive_drug} data found for patient {patient_id}"

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

        transplant_days = (transplant_date - admission_date).total_seconds() / (24 * 3600)
        discharge_days = (discharge_date - admission_date).total_seconds() / (24 * 3600)

        # Create base chart
        base_chart = alt.Chart(patient_med_data).add_selection(
            alt.selection_interval()
        )

        # Medication dose line
        dose_line = base_chart.mark_line(
            point=alt.OverlayMarkDef(filled=False, fill='white', strokeWidth=2),
            color='steelblue',
            strokeWidth=2
        ).encode(
            x=alt.X('days_since_admission:Q', 
                   axis=alt.Axis(title='Days Since Admission')),
            y=alt.Y('med_dose:Q', 
                   axis=alt.Axis(title=f'{vasoactive_drug.title()} Dose (mcg/kg/min)')),
            tooltip=['admin_dttm:T', 'med_dose:Q', 'days_since_admission:Q']
        )

        # Create combined reference lines with unified legend
        reference_lines_data = pd.DataFrame({
            'day': [0, transplant_days, discharge_days],
            'event_type': ['Admission', 'Transplant', 'Discharge'],
            'description': [
                'Admission (Day 0)',
                f'Transplant (Day {transplant_days:.1f})',
                f'Discharge (Day {discharge_days:.1f})'
            ]
        })

        # Create reference lines with unified legend
        reference_lines = alt.Chart(reference_lines_data).mark_rule(
            strokeWidth=2
        ).encode(
            x='day:Q',
            color=alt.Color('event_type:N',
                          scale=alt.Scale(
                              domain=['Admission', 'Transplant', 'Discharge'],
                              range=['blue', 'green', 'orange']
                          ),
                          legend=alt.Legend(title="Key Events", orient='top-right')),
            strokeDash=alt.StrokeDash('event_type:N',
                                   scale=alt.Scale(
                                       domain=['Admission', 'Transplant', 'Discharge'],
                                       range=[[2, 2], [5, 5], [2, 2]]
                                   )),
            tooltip=['description:N']
        )

        # Discharge category annotation
        discharge_annotation = alt.Chart(pd.DataFrame({
            'x': [discharge_days],
            'y': [patient_med_data['med_dose'].max() * 0.9],  # Position near top of chart
            'text': [f'Discharge: {discharge_category}']
        })).mark_text(
            align='center',
            baseline='bottom',
            fontSize=12,
            fontWeight='bold',
            color='orange'
        ).encode(
            x='x:Q',
            y='y:Q',
            text='text:N'
        )

        # Combine charts
        chart = (dose_line + reference_lines + discharge_annotation).resolve_scale(
            y='independent'
        ).properties(
            width=700,
            height=400,
            title=f'{vasoactive_drug.title()} Course - {organ_type.title()} Transplant - Patient {patient_id}'
        )

        return chart, None


    return (create_patient_vasoactive_chart,)


@app.cell
def _(
    create_patient_vasoactive_chart,
    mo,
    patient_selected,
    valeos_hospitalization_df,
    valeos_med_continuous_df,
    valeos_transplant_df,
    vasoactive_selected,
):
    # get patient ID from patient_selected
    patient_id = patient_selected.value if hasattr(patient_selected, 'value') else None

    # get vasoactive drug from vasoactive_selected
    vasoactive_drug = vasoactive_selected.value if hasattr(vasoactive_selected, 'value') else None

    # Create chart
    patient_chart, vaso_error_msg = create_patient_vasoactive_chart(
        patient_id, vasoactive_drug, valeos_med_continuous_df, valeos_transplant_df, valeos_hospitalization_df
    )

    if vaso_error_msg:
        patient_vasoactive_display = mo.md(f"**{vaso_error_msg}**")
    elif patient_chart is not None:
        patient_vasoactive_display = mo.ui.altair_chart(patient_chart)
    else:
        patient_vasoactive_display = mo.md("**No chart data available**")
    return (patient_vasoactive_display,)


if __name__ == "__main__":
    app.run()
