"""CROPGRO crop model framework — soybean and canola.

The CROPGRO framework supports any grain legume or oilseed crop that uses the
Boote–Jones–Hoogenboom photothermal phenology model.  This package provides:

- ``CropGROSoybean`` / ``SoybeanCultivar`` — soybean with N fixation
- ``CropGROCanola``  / ``CanolaCultivar``  — spring canola, no N fixation

Both models share the same CROPGRO base logic in :mod:`dssat.crop.cropgro.phenology`,
:mod:`dssat.crop.cropgro.growth`, and :mod:`dssat.crop.cropgro.nfix`.

Quick start::

    from dssat.crop.cropgro import CropGROSoybean, SoybeanCultivar
    from dssat.crop.cropgro import CropGROCanola, CanolaCultivar
"""

from dssat.crop.cropgro.soybean import SoybeanCultivar, CropGROSoybean
from dssat.crop.cropgro.canola import CanolaCultivar, CropGROCanola
from dssat.crop.cropgro.phenology import CropGROPhenology, CropGROPhenoState
from dssat.crop.cropgro.growth import CropGROGrowth, CropGROGrowthState
from dssat.crop.cropgro.nfix import CropGRONFix

__all__ = [
    "SoybeanCultivar",
    "CropGROSoybean",
    "CanolaCultivar",
    "CropGROCanola",
    "CropGROPhenology",
    "CropGROPhenoState",
    "CropGROGrowth",
    "CropGROGrowthState",
    "CropGRONFix",
]
