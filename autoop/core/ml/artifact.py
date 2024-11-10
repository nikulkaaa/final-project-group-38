from pydantic import BaseModel, Field
import base64
from typing import Optional, List
import pandas as pd
import uuid


class Artifact(BaseModel):
    """
    Artifact class represents a stored asset within an AutoML system.
    It encapsulates all necessary metadata along with the data itself.
    """
    name: str = Field(..., description="The name of the artifact")
    asset_path: str = Field(..., description="Path or identifier of the asset")
    data: Optional[object] = Field(None, description="Raw data or dataset")
    encoded_data: Optional[str] = Field(None,
                                        description="Base64 encoded data")
    metadata: Optional[dict] = Field(None,
                                     description="Optional metadata"
                                     "associated with the artifact")
    version: str = Field(..., description="Version of the artifact")
    tags: List[str] = Field(default_factory=list,
                            description="Tags for categorizing"
                            " and searching the artifact")
    type: str = Field(..., description="Type of the artifact.")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()),
                    description="Unique identifier for the artifact")

    class Config:
        """Configurate settings for Artifact."""
        # Allow the use of arbitrary types,
        # to store custom objects
        arbitrary_types_allowed = True

    def __init__(self, **kwargs) -> None:
        """Initialize the Artifact class."""
        super().__init__(**kwargs)
        if self.data is not None:
            self.load_data(self.data)

    def load_data(self, data: object) -> None:
        """
        Load the data into the artifact.
        If it's a DataFrame or binary data, it will be handled accordingly.
        """
        if isinstance(data, pd.DataFrame):
            # Convert DataFrame to raw bytes and store in `data`
            self.data = data.to_csv(index=False).encode('utf-8')
            self.encoded_data = base64.b64encode(self.data).decode('utf-8')
        elif isinstance(data, bytes):
            # Store raw bytes directly
            self.data = data
            self.encoded_data = base64.b64encode(data).decode('utf-8')
        else:
            # Convert other types to strings, store as bytes
            self.data = str(data).encode('utf-8')
            self.encoded_data = base64.b64encode(self.data).decode('utf-8')

    def read(self) -> bytes:
        """
        Read the artifact's data in its
        raw binary format in the dataset class.
        """
        return self.data

    def save(self, data: bytes) -> bytes:
        """Save binary data to the artifact."""
        self.data = data
        return self.data
