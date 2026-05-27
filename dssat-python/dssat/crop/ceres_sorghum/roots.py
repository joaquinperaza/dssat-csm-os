"""CERES-Sorghum root growth and water uptake.

Root dynamics are identical to CERES-Maize; this module re-exports
the shared implementation to avoid duplication.

Reference: Ritchie, J.T. et al. (1998) DSSAT v3.
"""

from dssat.crop.ceres_maize.roots import grow_roots, root_water_uptake

__all__ = ["grow_roots", "root_water_uptake"]
