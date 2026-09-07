"""MMM Sandbox: a small, readable media-mix-modeling toolkit.

Everything mathematical lives in this package. The Streamlit app in `app/`
only renders what these modules compute.
"""

from mmm_sandbox.transforms import geometric_adstock, hill_saturation

__all__ = ["geometric_adstock", "hill_saturation"]
