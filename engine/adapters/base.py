from typing import Protocol
import pandas as pd


class Adapter(Protocol):
    """Anything that can produce schema-conformant data. Evaluation dummy data = new adapter, nothing else changes."""
    def load(self) -> pd.DataFrame: ...
