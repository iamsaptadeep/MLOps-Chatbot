import pandas as pd
from evidently.report import Report
from evidently.metrics import DataDriftTable, DatasetDriftMetric

REFERENCE_PATH = "data/raw/bitext_customer_support.csv"
CURRENT_PATH = "monitoring/data/inference_log.csv"
OUTPUT_PATH = "monitoring/reports/drift.html"


def run_drift():

    ref = pd.read_csv(REFERENCE_PATH)
    cur = pd.read_csv(CURRENT_PATH)

    # Align columns
    ref = ref[["instruction"]].rename(columns={"instruction": "text"})
    cur = cur[["text"]]

    report = Report(
        metrics=[
            DatasetDriftMetric(),
            DataDriftTable()
        ]
    )

    report.run(
        reference_data=ref,
        current_data=cur
    )

    report.save_html(OUTPUT_PATH)


if __name__ == "__main__":
    run_drift()
