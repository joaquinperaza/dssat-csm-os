"""DSSAT crop models.

Available models
----------------

.. list-table::
   :header-rows: 1

   * - Class
     - Crop code
     - Source model
     - Status
   * - :class:`~dssat.crop.ceres_maize.CeresMaize`
     - ``MZ``
     - CERES-Maize (``Plant/CERES-Maize/``)
     - **Implemented**
   * - ``CeresWheat``
     - ``WH``
     - CERES-Wheat/Barley (``Plant/CERES-Wheat_Barley/``)
     - Stub
   * - ``CeresRice``
     - ``RI``
     - CERES-Rice (``Plant/CERES-Rice/``)
     - Stub
   * - ``CeresSorghum``
     - ``SG``
     - CERES-Sorghum (``Plant/CERES-Sorghum/``)
     - Stub
   * - ``CeresMillet``
     - ``ML``
     - CERES-Millet (``Plant/CERES-Millet/``)
     - Stub
   * - ``SubstorPotato``
     - ``PT``
     - SUBSTOR-Potato (``Plant/SUBSTOR-Potato/``)
     - Stub
   * - ``Cropgro``
     - ``SB``, ``PE``, ``BN``, ``TM``, ``CS``, …
     - CROPGRO (``Plant/CROPGRO/``)
     - Stub
   * - ``Canegro``
     - ``SC``
     - CANEGRO-Sugarcane (``Plant/CANEGRO-Sugarcane/``)
     - Stub
   * - ``Cscas``
     - ``CS``
     - CSCAS-Cassava (``Plant/CSCAS/``)
     - Stub
"""

from dssat.crop.base import CropModel
from dssat.crop.ceres_maize import CeresMaize, MaizeCultivar
from dssat.crop.ceres_sorghum import CeresSorghum, SorghumCultivar
from dssat.crop.ceres_millet import CeresMillet, MilletCultivar
from dssat.crop.ceres_wheat import CeresWheat, WheatCultivar

__all__ = [
    "CropModel",
    "CeresMaize", "MaizeCultivar",
    "CeresSorghum", "SorghumCultivar",
    "CeresMillet", "MilletCultivar",
    "CeresWheat", "WheatCultivar",
]


def __getattr__(name: str):
    """Lazy forward-declaration for not-yet-installed optional crop families.

    Raises :class:`ImportError` with a helpful message rather than a bare
    :exc:`AttributeError` when user code tries to import a crop family that
    is defined in the roadmap but not yet packaged.

    Example::

        from dssat.crop import cropgro   # → ImportError (not yet installed)
    """
    _roadmap = {
        "cropgro": (
            "CROPGRO (soybean, peanut, dry-bean, tomato, …) is not yet "
            "available in this release.  Check back in a future version or "
            "install the 'dssat-cropgro' extension package."
        ),
        "canegro": (
            "CANEGRO-Sugarcane is not yet available in this release."
        ),
        "substor": (
            "SUBSTOR-Potato is not yet available in this release."
        ),
        "cscas": (
            "CSCAS-Cassava is not yet available in this release."
        ),
    }
    key = name.lower()
    if key in _roadmap:
        raise ImportError(_roadmap[key])
    raise AttributeError(f"module 'dssat.crop' has no attribute {name!r}")
