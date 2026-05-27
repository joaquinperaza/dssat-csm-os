"""CERES-Maize crop model — top-level coordinator.

This module assembles the phenology, growth, and root sub-models into a
single interface that the main :class:`~dssat.core.engine.CropModel` API
expects.

Usage::

    from dssat.crop.ceres_maize import CeresMaize, MaizeCultivar

    cv = MaizeCultivar(p1=300, p2=0.52, p5=860, g2=800, g3=8.5, phint=38.9)
    model = CeresMaize(cultivar=cv, pltpop=7.0, sdepth=5.0, yrsim=2020100)

    # inside the daily loop:
    model.run(control, iswitch, soilprop, weather, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dssat.core.constants import NL, RUNINIT, SEASINIT, RATE, INTEGR, OUTPUT
from dssat.core.types import (
    ControlType, SwitchType, SoilType, WeatherType, ResidueType,
)
from dssat.crop.base import CropModel
from dssat.crop.ceres_maize.phenology import MaizeCultivar, MaizePhenology, PhenologyState
from dssat.crop.ceres_maize.growth import MaizeGrowth, GrowthState
from dssat.crop.ceres_maize.roots import grow_roots, root_water_uptake


class CeresMaize(CropModel):
    """CERES-Maize crop model.

    Implements the full CERES-Maize simulation cycle: phenology,
    photosynthesis, organ growth, root growth, and N uptake.

    Args:
        cultivar: Cultivar and ecotype coefficients.
        pltpop: Plant population (plants m⁻²).
        sdepth: Sowing depth (cm).
        yrsim: Simulation start date (YYYYDDD).
        rowspc: Row spacing (cm).
        slpf: Soil fertility factor (0–1).
        yrplt: Planting date (YYYYDDD); if -99 uses first sowing trigger.
    """

    crop_code: str = "MZ"
    model_name: str = "MZCER048"

    def __init__(
        self,
        cultivar: MaizeCultivar,
        pltpop: float,
        sdepth: float,
        yrsim: int,
        rowspc: float = 75.0,
        slpf: float = 1.0,
        yrplt: int = -99,
    ) -> None:
        self.cultivar = cultivar
        self.pltpop = pltpop
        self.sdepth = sdepth
        self.yrsim = yrsim
        self.rowspc = rowspc
        self.slpf = slpf
        self.yrplt = yrplt

        self._pheno = MaizePhenology(cultivar, pltpop, sdepth, yrsim)
        self._growth = MaizeGrowth(cultivar, pltpop, rowspc, slpf)

        # Root state
        self.rlv: np.ndarray = np.zeros(NL)
        self.rtdep: float = 0.0
        self.trwup: float = 0.0
        self.unh4: np.ndarray = np.zeros(NL)
        self.uno3: np.ndarray = np.zeros(NL)
        self.fracfts: np.ndarray = np.zeros(NL)
        self.senesce: ResidueType = ResidueType()

        # N stress — can be overridden externally by the engine
        self._nstres_external: float = 1.0
        # Daily N demand (kg N ha⁻¹) — set in _compute_n_uptake, read by engine
        self._n_demand: float = 0.0

    # ------------------------------------------------------------------
    @property
    def pheno(self) -> PhenologyState:
        """Current phenology state."""
        return self._pheno.state

    @property
    def growth(self) -> GrowthState:
        """Current growth state."""
        return self._growth.state

    # ------------------------------------------------------------------
    def run(
        self,
        control: ControlType,
        iswitch: SwitchType,
        soilprop: SoilType,
        weather: WeatherType,
        sw: np.ndarray,
        nh4: np.ndarray,
        no3: np.ndarray,
        es: float,
        eop: float,
    ) -> None:
        """Run one simulation step.

        Args:
            control: Simulation control block.
            iswitch: Simulation switches.
            soilprop: Soil profile properties.
            weather: Today's weather data.
            sw: Soil water content by layer.
            nh4: Ammonium by layer (µg g⁻¹).
            no3: Nitrate by layer (µg g⁻¹).
            es: Potential soil evaporation (mm d⁻¹).
            eop: Potential transpiration (mm d⁻¹).
        """
        dynamic = control.dynamic
        yrdoy = control.yrdoy
        nlayr = soilprop.nlayr

        # ---- Phenology
        g = self._growth.state
        self._pheno.run(
            dynamic=dynamic,
            yrdoy=yrdoy,
            tmax=weather.tmax,
            tmin=weather.tmin,
            srad=weather.srad,
            dayl=weather.dayl,
            snow=0.0,  # passed from watbal
            sw=sw,
            ll=soilprop.ll,
            dlayr=soilprop.dlayr,
            nlayr=nlayr,
            iswwat=iswitch.iswwat,
            twilen=getattr(weather, "twilen", weather.dayl),
            xn=int(g.cumph + 1.0),   # XN = leaf number ≈ cumph+1 (MZ_GROSUB)
            sump=g.sump if hasattr(g, "sump") else 0.0,
        )

        if dynamic in (RUNINIT, SEASINIT):
            self._init_roots(soilprop, nlayr)
            self._growth.run(
                dynamic=dynamic,
                pheno=self.pheno,
                yrdoy=yrdoy,
                tmax=weather.tmax,
                tmin=weather.tmin,
                srad=weather.srad,
                co2=weather.co2,
                sw=sw,
                dul=soilprop.dul,
                sat=soilprop.sat,
                ll=soilprop.ll,
                dlayr=soilprop.dlayr,
                nlayr=nlayr,
                nh4=nh4,
                no3=no3,
                kg2ppm=soilprop.kg2ppm,
                shf=soilprop.shf,
                rlv=self.rlv,
                rtdep=self.rtdep,
                trwup=self.trwup,
                eop=eop,
                iswwat=iswitch.iswwat,
                iswnit=iswitch.iswnit,
            )
            return

        if self.pheno.istage in (7, 8):
            return

        # ---- Root water uptake
        _, self.trwup = root_water_uptake(
            rlv=self.rlv,
            sw=sw,
            ll=soilprop.ll,
            dlayr=soilprop.dlayr,
            nlayr=nlayr,
            rwumx=self._growth._rwumx,
            rwuep1=self._growth._rwuep1,
            eop=eop,
        )

        # ---- Growth
        self._growth.run(
            dynamic=dynamic,
            pheno=self.pheno,
            yrdoy=yrdoy,
            tmax=weather.tmax,
            tmin=weather.tmin,
            srad=weather.srad,
            co2=weather.co2,
            sw=sw,
            dul=soilprop.dul,
            sat=soilprop.sat,
            ll=soilprop.ll,
            dlayr=soilprop.dlayr,
            nlayr=nlayr,
            nh4=nh4,
            no3=no3,
            kg2ppm=soilprop.kg2ppm,
            shf=soilprop.shf,
            rlv=self.rlv,
            rtdep=self.rtdep,
            trwup=self.trwup,
            eop=eop,
            iswwat=iswitch.iswwat,
            iswnit=iswitch.iswnit,
        )

        # ---- N uptake (RATE phase only)
        if dynamic == RATE and iswitch.iswnit == "Y":
            self.uno3, self.unh4 = self._compute_n_uptake(soilprop, no3, nh4, sw)
            # Apply external nstres override (set by engine from N supply/demand)
            self._growth.state.nstres = min(
                self._growth.state.nstres, self._nstres_external
            )

        # ---- Root growth
        if dynamic == INTEGR:
            g = self.growth
            self.rlv, self.rtdep = grow_roots(
                grort=g.grort,
                dtt=self.pheno.dtt,
                pltpop=self.pltpop,
                sw=sw,
                ll=soilprop.ll,
                dul=soilprop.dul,
                sat=soilprop.sat,
                dlayr=soilprop.dlayr,
                ds=soilprop.ds,
                shf=soilprop.shf,
                nlayr=nlayr,
                rlwr=self._growth._rlwr,
                rlv=self.rlv,
                rtdep=self.rtdep,
                rwumx=self._growth._rwumx,
            )

    # ------------------------------------------------------------------
    def set_nstres(self, v: float) -> None:
        """Set external N stress factor (0–1) used next RATE phase.

        Called by the engine after computing N supply/demand ratio.

        Args:
            v: N stress factor (0 = full stress, 1 = no stress).
        """
        self._nstres_external = float(max(0.0, min(1.0, v)))

    # ------------------------------------------------------------------
    def _compute_n_uptake(
        self,
        soilprop: SoilType,
        no3: np.ndarray,
        nh4: np.ndarray,
        sw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute daily N uptake per layer (kg N/ha).

        Based on MZ_NUPTAK.for (Fortran DSSAT CSM).

        Args:
            soilprop: Soil profile properties.
            no3: Nitrate concentration by layer (µg g⁻¹).
            nh4: Ammonium concentration by layer (µg g⁻¹).
            sw: Soil water content by layer (cm³ cm⁻³).

        Returns:
            Tuple ``(uno3, unh4)`` — NO₃ and NH₄ uptake per layer
            (kg N ha⁻¹ d⁻¹).
        """
        g = self._growth.state
        nlayr = soilprop.nlayr
        kg2ppm = soilprop.kg2ppm
        dlayr = soilprop.dlayr
        rlv = self.rlv
        shf = soilprop.shf
        ll = soilprop.ll
        sat = soilprop.sat

        # N demand
        tanc = g.stovn / max(g.stovwt, 0.001)
        ranc = g.rootn / max(g.rtwt, 0.001)
        tcnp = max(0.006, 0.040 - 0.03 * min(6.0, self.pheno.xstage))  # critical N conc
        rcnp = 0.01

        tndem = g.stovwt * max(0.0, tcnp - tanc)
        rndem = g.rtwt * max(0.0, rcnp - ranc)
        ndem = tndem + rndem
        andem = ndem * self.pltpop * 10.0  # kg N/ha
        self._n_demand = andem  # expose to engine for N stress calculation

        uno3 = np.zeros(NL)
        unh4 = np.zeros(NL)

        if andem <= 0.0:
            return uno3, unh4

        # Potential N supply
        trnu = 0.0
        rno3u = np.zeros(NL)
        rnh4u = np.zeros(NL)

        for L in range(nlayr):
            if rlv[L] > 0.0 and kg2ppm[L] > 0.0:
                sw_frac = (sw[L] - ll[L]) / max(0.001, sat[L] - ll[L])
                smdfr = max(0.0, min(1.0, 1.5 - 6.0 * (sw_frac - 0.5) ** 2))
                rfac = 1.0 - np.exp(-8.0 * rlv[L])
                fno3 = shf[L] * 0.075
                fnh4 = shf[L] * 0.075
                rno3u[L] = max(0.0, rfac * smdfr * fno3 * no3[L] * dlayr[L])
                rnh4u[L] = max(0.0, rfac * smdfr * fnh4 * (nh4[L] - 0.5) * dlayr[L])
                trnu += rno3u[L] + rnh4u[L]

        if trnu < 1e-6:
            return uno3, unh4

        andem = min(andem, trnu)
        nuf = andem / trnu

        for L in range(nlayr):
            if kg2ppm[L] > 0.0:
                xmin_no3 = 0.25 / kg2ppm[L]
                xmin_nh4 = 0.5 / kg2ppm[L]
                sno3_L = no3[L] / kg2ppm[L]
                snh4_L = nh4[L] / kg2ppm[L]
                uno3[L] = min(rno3u[L] * nuf, max(0.0, sno3_L - xmin_no3))
                unh4[L] = min(rnh4u[L] * nuf, max(0.0, snh4_L - xmin_nh4))

        return uno3, unh4

    # ------------------------------------------------------------------
    def _init_roots(self, soilprop: SoilType, nlayr: int) -> None:
        """Initialise root state at start of season."""
        self.rlv[:] = 0.0
        self.rtdep = self.sdepth
        self.trwup = 0.0
        # Seed layer gets initial root entry
        cumdep = 0.0
        for L in range(nlayr):
            cumdep += soilprop.dlayr[L]
            if self.sdepth <= cumdep:
                self.rlv[L] = 0.01
                break

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Return a dictionary of end-of-season output variables.

        Returns:
            Dictionary with keys matching standard DSSAT Summary.OUT
            column names.
        """
        g = self.growth
        p = self.pheno
        from dssat.core.date_utils import yrdoy_to_date
        def _safe_date(yrdoy: int) -> str:
            try:
                return str(yrdoy_to_date(yrdoy)) if yrdoy > 0 else "NA"
            except Exception:
                return "NA"

        return {
            "crop": self.crop_code,
            "model": self.model_name,
            "cultivar": self.cultivar.varno,
            "plant_date": _safe_date(self.yrplt),
            "anthesis_date": _safe_date(p.isdate),
            "maturity_date": _safe_date(p.mdate),
            "yield_kg_ha": round(g.yield_, 1),
            "biomass_kg_ha": round(g.biomas * 10.0, 1),
            "leaf_area_index": round(g.lai, 3),
            "grain_number_m2": round(g.seedno * self.pltpop * 10.0, 0),
            "grain_wt_mg": round(g.grnwt / max(g.seedno, 0.001) * 1000.0, 2),
            "n_uptake_kg_ha": round(g.wtnup * self.pltpop * 10.0, 2),
            "harvest_index": round(
                g.gpsm / max(g.biomas, 0.001), 3
            ),
            "crop_status": p.crop_status,
            "istage_final": p.istage,
        }
