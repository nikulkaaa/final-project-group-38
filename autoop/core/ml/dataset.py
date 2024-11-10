from autoop.core.ml.artifact import Artifact
import pandas as pd
import io


class Dataset(Artifact):
    """A dataset class to store data."""
    def __init__(self, *args, **kwargs) -> None:
        """Inititalize the Dataset class."""
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame,
                       name: str, asset_path: str,
                       version: str = "1.0.0") -> 'Dataset':
        """Static method to create a dataset from a dataframe."""
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """Get the values store in the data attribute."""
        bytes = self.data
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """Save the data to the dataset as bytes."""
        bytes = data.to_csv(index=False).encode()
        self.data = bytes
        return self.data
