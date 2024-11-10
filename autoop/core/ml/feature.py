
from pydantic import BaseModel, Field

class Feature(BaseModel):
    """A class for a feature of a dataset."""
    name: str = Field(..., description='Name of the Feature')
    type: str = Field(..., description='Feature type.')

    def __str__(self) -> str:
        """Returns a string representation of a Feature instance."""
        return f"Feature(name='{self.name}', type='{self.type}')"
