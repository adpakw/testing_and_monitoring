import asyncio
from collections import deque

import mlflow
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from evidently.ui.workspace import RemoteWorkspace

EVIDENTLY_URL = "http://158.160.2.37:8000/"
PROJECT_ID = "019d061f-cc08-7b5e-b932-d792a1f258e2"
REFERENCE_SIZE = 1000
DRIFT_CHECK_SIZE = 500


class DriftCollector:
    def __init__(self):
        self.buffer = deque(maxlen=DRIFT_CHECK_SIZE)
        self.reference_data = None
        self.lock = asyncio.Lock()

    async def collect(self, features: dict, prediction: int, probability: float):
        record = {**features, "prediction": prediction, "probability": probability}
        async with self.lock:
            self.buffer.append(record)
            if len(self.buffer) >= DRIFT_CHECK_SIZE and self.reference_data is not None:
                await self._send_report()

    async def set_reference(self, reference_df: pd.DataFrame):
        self.reference_data = reference_df

    async def _send_report(self):
        async with self.lock:
            current_df = pd.DataFrame(list(self.buffer))
            self.buffer.clear()
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=self.reference_data, current_data=current_df)
        workspace = RemoteWorkspace(EVIDENTLY_URL)
        workspace.add_run(PROJECT_ID, report)
