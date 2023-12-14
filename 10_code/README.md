## This directory contains the code used for the project

**Contents:**
- [`10_chunking_states.py`](10_chunking_states.py): A python script that chunks the orignal Washington Post tsv file (around 100gbs) into individual states and saved into the `00_data` folder.

- [`11_confrim_states.py`](11_confrim_states.py): A python script used for sanity checking for states gathered and also used to see difference in naming of counties of Washington Post vs US Vital Statistics

- [`14_mortality_txt_to_csv.py`](14_mortality_txt_to_csv.py): A python script that converts `00_data/US_VitalStatistics` .txt files into appropriate .csv fileS

- [`15_opioid_groupby.py`](15_opioid_groupby.py): A python script that aggregates data of the three datasets used for this project for our analysis

- [`20_control_states.py`](20_control_states.py): A python script that identifies control states and does preliminary sample pre-post and diff-in-diff viz for opioid consumption

- [`21_pre_post_viz.py`](21_pre_post_viz.py): A python script that does preliminary sample pre-post viz for Florida opioid consumption

- [`22_FL_merge_test.py`](22_FL_merge_test.py): A python script that does merge tests on Florida when we initally didn't understand why our three datasets weren't merging correctly

- [`23_mortality_merge&checks.py`](23_mortality_merge&checks.py): A python script that does preliminary pre-post and diff-diff viz for mortality and does checks on our data

- [`30_final_opiod_viz.py`](30_final_opiod_viz.py): A python script that is our final `30_results` viz for conumption

- [`31_mortality_main.py`](31_mortality_main.py): A python script that is our final `30_results` viz for mortality 