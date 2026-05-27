"""CERES-Millet root growth and water uptake.

Re-exports from the CERES-Maize root module since root dynamics are
essentially identical between millet, sorghum, and maize.
"""

from dssat.crop.ceres_maize.roots import grow_roots, root_water_uptake

__all__ = ["grow_roots", "root_water_uptake"]
