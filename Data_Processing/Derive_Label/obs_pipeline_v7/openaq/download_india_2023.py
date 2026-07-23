"""
One-off: download OpenAQ India NO2 for 2023 (missing from raw/india/, which
stops at 2022). Reuses async_openaq_2.py unchanged, only redirecting output to
the storage1 raw dir so 2023.csv lands next to 2016-2022.csv.

Run:  python download_india_2023.py   (needs `openaq` pip package + internet)
"""
import sys
import async_openaq_2 as dl

dl.BASE_OUTPUT_DIR = ("/path/to/NO2_DL_global/"
                      "TrainingDatasets/Global_NO2_v7/no2_ground/openaq/raw")
sys.argv = ["async_openaq_2.py", "--countries", "IN",
            "--start-year", "2023", "--end-year", "2023"]
dl.main()
