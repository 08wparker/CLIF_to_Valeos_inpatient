# CLIF to Valeos inpatient

## CLIF VERSION 2.1

## Objective

Use CPT procedure codes to identify transplant recipient hospitalizations in comprehensive CLIF database (e.g. all adult inpatients)

## Required CLIF tables and fields

Please refer to the online [CLIF data dictionary](https://clif-consortium.github.io/website/data-dictionary.html), [ETL tools](https://github.com/clif-consortium/CLIF/tree/main/etl-to-clif-resources), and [specific table contacts](https://github.com/clif-consortium/CLIF?tab=readme-ov-file#relational-clif) for more information on constructing the required tables and fields. 

The following 2.0 tables are required:
1. **patient**: 
2. **hospitalization**: 
3. **vitals**: 
4. **labs**: 
5. **medication_admin_continuous**
6. **medication_admin_intermittment**:
7. **respiratory_support**

the following 2.1 tables are required
8. **crrt_therapy**
9. **ecmo_mcs**
10. **patient_procedure**

## Cohort identification

The cohort will be identified using the following CPT code list

| cpt_code | proc_name                                                     | organ  |
|----------|---------------------------------------------------------------|--------|
| 33945    | PR HEART TRANSPLANT W/WO RECIPIENT CARDIECTOMY               | heart  |
| 33935    | PR HEART-LUNG TRNSPL W/RECIPIENT CARDIECTOMY-PNUMEC          | lung   |
| 32854    | PR LUNG TRANSPLANT 2 W/CARDIOPULMONARY BYPASS                | lung   |
| 32852    | PR LUNG TRANSPLANT 1 W/CARDIOPULMONARY BYPASS                | lung   |
| 32853    | PR LUNG TRANSPLANT 2 W/O CARDIOPULMONARY BYPASS              | lung   |
| 32851    | PR LUNG TRANSPLANT 1 W/O CARDIOPULMONARY BYPASS              | lung   |
| 47135    | PR LVR ALTRNSPLJ ORTHOTOPIC PRTL/WHL DON ANY AGE             | liver  |
| 50360    | PR RENAL ALTRNSPLJ IMPLTJ GRF W/O RCP NEPHRECTOMY            | kidney |
| 50360    | HC RENAL ALLOTRANSPLANTATION, IMPLANT GRAFT W/O DONOR        | kidney |
| 50365    | PR RENAL ALTRNSPLJ IMPLTJ GRF W/RCP NEPHRECTOMY              | kidney |



## Expected Results

The initial results will be a database of CLIF tables filtered to transplant recipients `[your institution]_valeos_inpatient_[CLIF table].parquet` These will be in CLIF format, with an additional `transplant` table with columns `patient_id`, `transplant_type`, `recorded_dttm`

Summary tables, statistics, and initial exploratory data analysis will be available in the [`output/final`](output/README.md) directory.

## Detailed Instructions for running the project

## 1. Update `config/config.json`
Follow instructions in the [config/README.md](config/README.md) file for detailed configuration steps.

## 2. Set up the project environment

Open your terminal and run the setup script:
```
source ./setup_env.sh
```

This will create and activate the virtual environment in your current shell session.

## 3. Run code
Detailed instructions on the code workflow are provided in the [code directory](code/README.md)

---


