from pydantic import BaseModel, Field
import base64
from typing import Optional, Any
import pandas as pd 

class Artifact(BaseModel):
    """
    Artifact class.
    """
    name: str = Field(..., description="The name of the artifact")
    asset_path: str = Field(..., description="Path or identifier of the asset")
    data: Optional[Any] = Field(None, description="Raw data or dataset")  # New attribute to hold data
    encoded_data: Optional[str] = Field(None, description="Base64 encoded data")
    metadata: Optional[dict] = Field(None, description="Optional metadata associated with the artifact")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.data is not None:
            self.load_data(self.data)

    def load_data(self, data: Any) -> None:
        """
        Load the data into the artifact. If it's a DataFrame or binary data, it will be handled accordingly.
        """
        if isinstance(data, pd.DataFrame):
            # Convert DataFrame to CSV string for simplicity
            self.encoded_data = base64.b64encode(data.to_csv(index=False).encode('utf-8')).decode('utf-8')
        elif isinstance(data, bytes):
            self.encoded_data = base64.b64encode(data).decode('utf-8')
        else:
            # Handle string or other types of data
            self.encoded_data = str(data)

    def get_data(self) -> Optional[Any]:
        """
        Retrieve the original data. If it was a DataFrame, decode it back to CSV format; 
        otherwise, return the decoded data or None if no data was provided.
        """
        if self.encoded_data:
            try:
                # Attempt to decode as CSV string and return as DataFrame
                decoded_data = base64.b64decode(self.encoded_data.encode('utf-8')).decode('utf-8')
                # Check if it is a DataFrame-like CSV (you can adjust this as needed)
                return pd.read_csv(pd.compat.StringIO(decoded_data))
            except Exception:
                # If not a DataFrame, return as bytes
                return base64.b64decode(self.encoded_data.encode('utf-8'))
        return None

    def read(self) -> bytes:
        """Read the artifact's data in its raw binary format."""
        return self.data

    def save(self, data: bytes) -> bytes:
        """Save binary data to the artifact."""
        self.data = data
        return self.data
