"""Main DSSAT simulation engine.

:class:`Simulation` is the top-level object.  It wires together all
sub-modules (weather, soil, crop, management) and runs the daily time loop.

This mirrors the ``LAND`` subroutine in ``CSM_Main/LAND.for`` and the outer
simulation loop in ``CSM_Main/CSM.for``.

Usage::

    from dssat import Simulation
    from dssat.crop.ceres_maize import CeresMaize, MaizeCultivar

    sim = Simulation.from_json("experiment.json")
    results = sim.run()
    print(results["summary"])
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from dssat.core.constants import (
    NL, RUNINIT, SEASINIT, RATE, INTEGR, OUTPUT, SEASEND, ENDRUN,
)
from dssat.core.types import (
    ControlType, SwitchType, SoilType, WeatherType,
    TillType, MulchType, FloodWatType, FloodNType,
    FertType, OrgMatAppType, ResidueType, FertilizerEvent,
)
from dssat.core.date_utils import incdat, timdif, date_to_yrdoy
from dssat.weather.weather import WeatherModule
from dssat.soil.water.watbal import WaterBalance
from dssat.crop.base import CropModel

# Soil N modules — optional, imported lazily so missing modules don't break
try:
    from dssat.soil.nitrogen import SoilNitrogenModule
    from dssat.soil.organic_matter import SoilOrganicMatterModule
    _SOIL_N_AVAILABLE = True
except ImportError:
    _SOIL_N_AVAILABLE = False

# Fertilizer placement
try:
    from dssat.management.fertilizer import place_fertilizer
    _FERT_AVAILABLE = True
except ImportError:
    _FERT_AVAILABLE = False


class Simulation:
    """DSSAT Cropping System Model simulation.

    This class orchestrates the full simulation: it initialises each
    sub-module, advances the daily time loop, and collects output.

    Args:
        control: Simulation control block.
        iswitch: Simulation switches.
        weather_module: Configured :class:`~dssat.weather.WeatherModule`.
        water_balance: Configured :class:`~dssat.soil.water.WaterBalance`.
        crop_model: A concrete :class:`~dssat.crop.base.CropModel` instance.
        soilprop: Initial soil profile.
        sw_init: Initial soil water content by layer (cm³ cm⁻³).
        no3_init: Initial NO₃ by layer (µg g⁻¹).
        nh4_init: Initial NH₄ by layer (µg g⁻¹).
        irrigation_events: List of :class:`~dssat.core.types.IrrigationEvent`.
        fertilizer_events: List of :class:`~dssat.core.types.FertilizerEvent`.
    """

    def __init__(
        self,
        control: ControlType,
        iswitch: SwitchType,
        weather_module: WeatherModule,
        water_balance: WaterBalance,
        crop_model: CropModel,
        soilprop: SoilType,
        sw_init: np.ndarray,
        no3_init: Optional[np.ndarray] = None,
        nh4_init: Optional[np.ndarray] = None,
        irrigation_events: Optional[list] = None,
        fertilizer_events: Optional[list] = None,
    ) -> None:
        self.control = control
        self.iswitch = iswitch
        self.weather_module = weather_module
        self.water_balance = water_balance
        self.crop_model = crop_model
        self.soilprop = soilprop
        self.sw_init = sw_init.copy()
        self.no3 = no3_init.copy() if no3_init is not None else np.zeros(NL)
        self.nh4 = nh4_init.copy() if nh4_init is not None else np.zeros(NL)
        self.irrig_events = irrigation_events or []
        self.fert_events = fertilizer_events or []

        # Soil N and organic matter modules
        if _SOIL_N_AVAILABLE and iswitch.iswnit == "Y":
            self.soil_n: Optional[SoilNitrogenModule] = SoilNitrogenModule()
            self.soil_om: Optional[SoilOrganicMatterModule] = SoilOrganicMatterModule()
            # Initialise N pools from initial conditions
            kg2ppm = soilprop.kg2ppm
            no3_kgha = (
                no3_init / kg2ppm
                if no3_init is not None
                else np.zeros(NL)
            )
            nh4_kgha = (
                nh4_init / kg2ppm
                if nh4_init is not None
                else np.zeros(NL)
            )
            # Avoid division by zero in layers with kg2ppm == 0
            no3_kgha = np.where(kg2ppm > 0.0, no3_kgha, 0.0)
            nh4_kgha = np.where(kg2ppm > 0.0, nh4_kgha, 0.0)
            self.soil_n.initialise_n(no3_kgha, nh4_kgha, soilprop.nlayr, kg2ppm)
        else:
            self.soil_n = None
            self.soil_om = None

        # Working arrays
        self._sw = self.sw_init.copy()
        self._swdeltx = np.zeros(NL)  # SW change from root extraction
        self._upflow = np.zeros(NL)
        self._mulch = MulchType()
        self._floodwat = FloodWatType()
        self._tillvals = TillType()

        # Daily output accumulator
        self._daily_rows: list[dict] = []

    # ------------------------------------------------------------------
    @classmethod
    def from_json(cls, experiment_path: str | Path) -> "Simulation":
        """Build a :class:`Simulation` from an experiment JSON file.

        This factory method reads the experiment, weather, and soil JSON
        files referenced in *experiment_path*, constructs all sub-modules,
        and returns a ready-to-run :class:`Simulation` instance.

        Args:
            experiment_path: Path to the experiment ``.json`` file.

        Returns:
            Configured :class:`Simulation` ready to call :meth:`run`.

        Raises:
            ValueError: If the crop model specified in the experiment is not
                yet implemented.
        """
        from dssat.io.readers import (
            load_experiment, load_weather, load_soil,
            build_initial_sw, build_initial_nutrients,
        )
        from dssat.crop.ceres_maize import CeresMaize, MaizeCultivar
        from dssat.crop.ceres_wheat import CeresWheat, WheatCultivar
        from dssat.crop.ceres_sorghum import CeresSorghum, SorghumCultivar
        from dssat.crop.ceres_millet import CeresMillet, MilletCultivar
        from dssat.crop.cropgro import CropGROSoybean, SoybeanCultivar
        from dssat.crop.cropgro import CropGROCanola, CanolaCultivar

        exp = load_experiment(experiment_path)
        base_dir = Path(experiment_path).parent

        # ---- weather
        wx_spec = exp["weather"]
        if "inline" in wx_spec:
            wx_data = wx_spec["inline"]
        else:
            wx_data = base_dir / wx_spec["source"]
        wx_records, wx_meta = load_weather(wx_data)

        # ---- soil
        soil_spec = exp["soil"]
        if "inline" in soil_spec:
            soil_data = soil_spec["inline"]
        else:
            soil_data = base_dir / soil_spec["source"]
        soilprop = load_soil(soil_data)

        # ---- initial conditions
        sw_init = build_initial_sw(exp, soilprop)
        no3_init, nh4_init = build_initial_nutrients(exp, soilprop)

        # ---- control
        sim_cfg = exp.get("simulation", {})
        control = ControlType()
        control.yrsim = int(sim_cfg["start_date"])
        control.yrdoy = control.yrsim
        control.nyrs = int(sim_cfg.get("nyrs", 1))
        control.crop = exp["crop"]["model"][:2].upper()
        control.model = exp["crop"]["model"].upper().ljust(8)[:8]
        control.dynamic = RUNINIT

        # ---- switches
        iswitch = SwitchType()
        iswitch.iswwat = sim_cfg.get("water", "Y")
        iswitch.iswnit = sim_cfg.get("nitrogen", "Y")
        iswitch.iswpho = sim_cfg.get("phosphorus", "N")
        iswitch.iswpot = sim_cfg.get("potassium", "N")
        iswitch.meevp = sim_cfg.get("et_method", "R")
        iswitch.mesom = sim_cfg.get("som_method", "G")

        # ---- weather module
        weather_mod = WeatherModule(
            daily_records=wx_records,
            xlat=float(wx_meta.get("lat", 0.0)),
            xlong=float(wx_meta.get("lon", 0.0)),
            xelev=float(wx_meta.get("elev", 0.0)),
            tav=float(wx_meta.get("tav", 20.0)),
            tamp=float(wx_meta.get("tamp", 10.0)),
            co2=float(sim_cfg.get("co2", 380.0)),
        )

        # ---- crop model
        crop_cfg = exp["crop"]
        model_code = crop_cfg["model"].upper()
        cv_data = crop_cfg["cultivar"]

        # Common planting parameters shared by all CERES crops.
        plt = exp.get("planting", {})
        yrplt = int(plt.get("date", -99))
        _pltpop_default = float(plt.get("pltpop", 7.0))
        _sdepth_default = float(plt.get("sdepth", 5.0))
        _rowspc_default = float(plt.get("rowspc", 75.0))

        # Helper: extract numeric cultivar kwargs, skipping identity fields.
        def _cv_floats(skip: tuple) -> dict:
            return {k: float(v) for k, v in cv_data.items() if k not in skip}

        if model_code.startswith("MZCER"):
            cv = MaizeCultivar(
                varno=cv_data.get("id", "GENERIC"),
                vrname=cv_data.get("name", "Generic Maize"),
                econo=cv_data.get("ecotype", "DFAULT"),
                **_cv_floats(("id", "name", "ecotype")),
            )
            crop_model: CropModel = CeresMaize(
                cultivar=cv,
                pltpop=float(plt.get("pltpop", 7.0)),
                sdepth=_sdepth_default,
                yrsim=control.yrsim,
                rowspc=_rowspc_default,
                slpf=float(soilprop.slpf),
                yrplt=yrplt,
            )

        elif model_code.startswith("WHCER"):
            # WheatCultivar uses 'id'/'name'/'ecotype' field names.
            cv_w = WheatCultivar(
                id=cv_data.get("id", "IB1500"),
                name=cv_data.get("name", "Generic Spring Wheat"),
                ecotype=cv_data.get("ecotype", "USWH01"),
                **_cv_floats(("id", "name", "ecotype")),
            )
            crop_model = CeresWheat(
                cultivar=cv_w,
                pltpop=float(plt.get("pltpop", 300.0)),
                sdepth=_sdepth_default,
                yrsim=control.yrsim,
                rowspc=float(plt.get("rowspc", 20.0)),
                slpf=float(soilprop.slpf),
                yrplt=yrplt,
            )

        elif model_code.startswith("SGCER"):
            cv_s = SorghumCultivar(
                varno=cv_data.get("id", "IS3301"),
                vrname=cv_data.get("name", "Generic Sorghum"),
                econo=cv_data.get("ecotype", "DFAULT"),
                **_cv_floats(("id", "name", "ecotype")),
            )
            crop_model = CeresSorghum(
                cultivar=cv_s,
                pltpop=_pltpop_default,
                sdepth=_sdepth_default,
                yrsim=control.yrsim,
                rowspc=_rowspc_default,
                slpf=float(soilprop.slpf),
                yrplt=yrplt,
            )

        elif model_code.startswith("MLCER"):
            cv_m = MilletCultivar(
                varno=cv_data.get("id", "LCSA94"),
                vrname=cv_data.get("name", "Generic Millet"),
                econo=cv_data.get("ecotype", "DFAULT"),
                **_cv_floats(("id", "name", "ecotype")),
            )
            crop_model = CeresMillet(
                cultivar=cv_m,
                pltpop=float(plt.get("pltpop", 25.0)),
                sdepth=_sdepth_default,
                yrsim=control.yrsim,
                rowspc=_rowspc_default,
                slpf=float(soilprop.slpf),
                yrplt=yrplt,
            )

        elif model_code.startswith(("SBGRO",)):
            cv_sb = SoybeanCultivar(
                varno=cv_data.get("varno", cv_data.get("id", "IB0011")),
                vrname=cv_data.get("vrname", cv_data.get("name", "Braxton")),
                econo=cv_data.get("econo", cv_data.get("ecotype", "IB0001")),
                **_cv_floats(("varno", "vrname", "econo", "id", "name", "ecotype", "fix_n")),
            )
            crop_model = CropGROSoybean(
                cultivar=cv_sb,
                pltpop=float(plt.get("pltpop", 30.0)),
                sdepth=_sdepth_default,
                yrsim=control.yrsim,
                rowspc=float(plt.get("rowspc", 50.0)),
                slpf=float(soilprop.slpf),
                yrplt=yrplt,
            )

        elif model_code.startswith(("CNGRO", "CRGRO")):
            cv_cn = CanolaCultivar(
                varno=cv_data.get("varno", cv_data.get("id", "CGEXPT")),
                vrname=cv_data.get("vrname", cv_data.get("name", "Generic Spring Canola")),
                econo=cv_data.get("econo", cv_data.get("ecotype", "CNSP01")),
                **_cv_floats(("varno", "vrname", "econo", "id", "name", "ecotype", "fix_n")),
            )
            crop_model = CropGROCanola(
                cultivar=cv_cn,
                pltpop=float(plt.get("pltpop", 100.0)),
                sdepth=_sdepth_default,
                yrsim=control.yrsim,
                rowspc=float(plt.get("rowspc", 25.0)),
                slpf=float(soilprop.slpf),
                yrplt=yrplt,
            )

        elif model_code.startswith(("CNCSS", "CASCY")):
            raise NotImplementedError(
                f"CROPGRO-Cassava ('{model_code}') is not yet implemented. "
                "Support is planned for a future release."
            )

        else:
            raise ValueError(
                f"Crop model '{model_code}' is not recognised. "
                "Currently available: MZCER, WHCER, SGCER, MLCER, SBGRO, CNGRO. "
                "See dssat.crop for the implementation roadmap."
            )

        # ---- water balance
        water_balance = WaterBalance()
        water_balance.initialise_sw(sw_init, soilprop.nlayr, soilprop.ll)

        # ---- irrigation & fertiliser events
        irrig = [
            type("IrrigEvent", (), e)()
            for e in exp.get("irrigation", [])
        ]
        fert_events = []
        for fe in exp.get("fertilizer", []):
            fert_events.append(FertilizerEvent(
                date=fe.get("date", 0),
                amount=fe.get("amount", 0.0),
                n_pct=fe.get("n_pct", 46.0),
                type=fe.get("type", "FE005"),
                depth=fe.get("depth", 0.0),
            ))
        fert = fert_events

        return cls(
            control=control,
            iswitch=iswitch,
            weather_module=weather_mod,
            water_balance=water_balance,
            crop_model=crop_model,
            soilprop=soilprop,
            sw_init=sw_init,
            no3_init=no3_init,
            nh4_init=nh4_init,
            irrigation_events=irrig,
            fertilizer_events=fert,
        )

    # ------------------------------------------------------------------
    def run(self) -> dict:
        """Execute the full simulation and return results.

        Returns:
            Dictionary with keys:

            - ``"summary"`` — end-of-season summary dict from the crop model.
            - ``"daily"`` — :class:`pandas.DataFrame` with one row per day.
        """
        ctrl = self.control
        sw = self._sw

        # ---- Run initialisation
        ctrl.dynamic = RUNINIT
        self._step_all(ctrl, sw)

        # ---- Seasonal initialisation
        ctrl.dynamic = SEASINIT
        self._step_all(ctrl, sw)

        # ---- Daily loop
        yrdoy = ctrl.yrsim
        end_yrdoy = incdat(ctrl.yrsim, 365 * ctrl.nyrs + 10)

        while yrdoy <= end_yrdoy:
            ctrl.yrdoy = yrdoy
            ctrl.das += 1

            # Rate
            ctrl.dynamic = RATE
            self._step_all(ctrl, sw)

            # Integration
            ctrl.dynamic = INTEGR
            self._step_all(ctrl, sw)

            # Output
            ctrl.dynamic = OUTPUT
            self._collect_daily(ctrl, sw)

            # Check for crop maturity
            if hasattr(self.crop_model, "pheno"):
                pheno = self.crop_model.pheno
                if pheno.mdate > 0 and pheno.mdate <= yrdoy:
                    break

            yrdoy = incdat(yrdoy, 1)

        # ---- Season end
        ctrl.dynamic = SEASEND
        self._step_all(ctrl, sw)

        return {
            "summary": self.crop_model.summary(),
            "daily": pd.DataFrame(self._daily_rows),
        }

    # ------------------------------------------------------------------
    def _step_all(self, ctrl: ControlType, sw: np.ndarray) -> None:
        """Call all sub-modules for the current dynamic phase."""
        # Weather
        self.weather_module.run(ctrl, self.iswitch)
        weather = self.weather_module.weather

        if ctrl.dynamic in (RATE, INTEGR):
            # Irrigation on today's date
            irramt = self._get_irrigation(ctrl.yrdoy)

            # Potential ET (simplified Priestley-Taylor)
            es, eop = _potential_et(weather, self.soilprop, sw, self.crop_model)

            # Water balance
            self.water_balance.run(
                control=ctrl,
                iswitch=self.iswitch,
                soilprop=self.soilprop,
                weather=weather,
                es=es,
                irramt=irramt,
                swdeltx=self._swdeltx,
                tillvals=self._tillvals,
                mulch=self._mulch,
                floodwat=self._floodwat,
            )

            if ctrl.dynamic == INTEGR:
                sw[:] = self.water_balance.sw

            # ---- Fertilizer placement (RATE phase)
            fert_sno3 = np.zeros(NL)
            fert_snh4 = np.zeros(NL)
            fert_urea = np.zeros(NL)
            apply_fert_today = False
            if ctrl.dynamic == RATE and _FERT_AVAILABLE:
                for fev in self.fert_events:
                    if getattr(fev, "date", 0) == ctrl.yrdoy:
                        placement = place_fertilizer(fev, self.soilprop)
                        fert_sno3 += placement["sno3"]
                        fert_snh4 += placement["snh4"]
                        fert_urea += placement["urea"]
                        apply_fert_today = True

            # ---- Organic matter module
            if self.soil_om is not None:
                self.soil_om.run(ctrl, self.iswitch, self.soilprop, weather, sw)
                ssom = self.soil_om.ssom
            else:
                ssom = np.zeros(NL)

            # ---- Crop model
            self.crop_model.run(
                control=ctrl,
                iswitch=self.iswitch,
                soilprop=self.soilprop,
                weather=weather,
                sw=sw,
                nh4=self.nh4,
                no3=self.no3,
                es=es,
                eop=eop,
            )

            # ---- Soil N module
            if self.soil_n is not None:
                uno3 = getattr(self.crop_model, "uno3", np.zeros(NL))
                unh4 = getattr(self.crop_model, "unh4", np.zeros(NL))
                drn = getattr(self.water_balance, "drn", np.zeros(NL))
                self.soil_n.run(
                    ctrl,
                    self.iswitch,
                    self.soilprop,
                    weather,
                    sw=sw,
                    drn=drn,
                    ssom=ssom,
                    uno3=uno3,
                    unh4=unh4,
                    fert_sno3=fert_sno3,
                    fert_snh4=fert_snh4,
                    fert_urea=fert_urea,
                    apply_fert_today=apply_fert_today,
                )
                # Sync concentrations back to engine arrays (ppm)
                self.no3[:] = self.soil_n.no3
                self.nh4[:] = self.soil_n.nh4

                # ---- N stress feedback to crop model (RATE only)
                if ctrl.dynamic == RATE:
                    total_n_supply = float(np.sum(uno3 + unh4))
                    total_n_demand = getattr(self.crop_model, "_n_demand", 0.0)
                    if total_n_demand > 0.001:
                        nstres_val = min(1.0, total_n_supply / total_n_demand)
                    else:
                        nstres_val = 1.0
                    if hasattr(self.crop_model, "set_nstres"):
                        self.crop_model.set_nstres(nstres_val)
        else:
            # Init phases — call both with dummy values
            self.water_balance.run(
                ctrl, self.iswitch, self.soilprop,
                weather, 0.0, 0.0,
                self._swdeltx, self._tillvals, self._mulch, self._floodwat,
            )
            # Organic matter and soil N init phases
            if self.soil_om is not None:
                self.soil_om.run(ctrl, self.iswitch, self.soilprop, weather, sw)
            if self.soil_n is not None:
                self.soil_n.run(
                    ctrl, self.iswitch, self.soilprop, weather,
                    sw=sw, drn=np.zeros(NL), ssom=np.zeros(NL),
                    uno3=np.zeros(NL), unh4=np.zeros(NL),
                )
            self.crop_model.run(
                ctrl, self.iswitch, self.soilprop, weather,
                sw, self.nh4, self.no3, 0.0, 0.0,
            )

    # ------------------------------------------------------------------
    def _get_irrigation(self, yrdoy: int) -> float:
        """Return irrigation amount (mm) for today."""
        total = 0.0
        for ev in self.irrig_events:
            if getattr(ev, "date", 0) == yrdoy:
                total += float(getattr(ev, "amount", 0.0))
        return total

    # ------------------------------------------------------------------
    def _collect_daily(self, ctrl: ControlType, sw: np.ndarray) -> None:
        """Append one row of daily output."""
        from dssat.core.date_utils import yrdoy_to_date
        weather = self.weather_module.weather
        row: dict = {
            "date": str(yrdoy_to_date(ctrl.yrdoy)),
            "yrdoy": ctrl.yrdoy,
            "das": ctrl.das,
            "tmax": weather.tmax,
            "tmin": weather.tmin,
            "srad": weather.srad,
            "rain": weather.rain,
            "tsw_cm": round(float(np.sum(sw[:self.soilprop.nlayr] * self.soilprop.dlayr[:self.soilprop.nlayr])), 3),
            "runoff_mm": round(self.water_balance.runoff, 2),
            "drain_mm": round(self.water_balance.drain, 2),
            "snow_mm": round(self.water_balance.snow, 2),
        }
        # Crop outputs
        if hasattr(self.crop_model, "pheno"):
            p = self.crop_model.pheno
            g = self.crop_model.growth
            row.update({
                "istage": p.istage,
                "xstage": round(p.xstage, 3),
                "lai": round(g.lai, 4),
                "biomas_g_m2": round(g.biomas, 2),
                "yield_kg_ha": round(g.yield_, 1),
                "swfac": round(g.swfac, 3),
                "nstres": round(g.nstres, 3),
            })
        # Soil N outputs (only when soil N module is active)
        if self.soil_n is not None:
            nlayr = self.soilprop.nlayr
            kg2ppm = self.soilprop.kg2ppm
            safe_kg2ppm = np.maximum(kg2ppm[:nlayr], 1e-6)
            row["total_no3_kg_ha"] = round(
                float(np.sum(self.no3[:nlayr] / safe_kg2ppm)), 2
            )
            row["total_nh4_kg_ha"] = round(self.soil_n.total_nh4, 2)
            row["nitrif_kg_ha"] = round(self.soil_n.nitrif_today, 3)
            row["leach_kg_ha"] = round(self.soil_n.leach_today, 3)
        else:
            row["total_no3_kg_ha"] = None
            row["total_nh4_kg_ha"] = None
            row["nitrif_kg_ha"] = None
            row["leach_kg_ha"] = None
        self._daily_rows.append(row)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _potential_et(
    weather: WeatherType,
    soilprop: SoilType,
    sw: np.ndarray,
    crop_model,
) -> tuple[float, float]:
    """Estimate potential soil evaporation and transpiration (Ritchie method).

    Returns:
        Tuple ``(es, eop)`` in mm d⁻¹.
    """
    # Priestley-Taylor ETP
    delta = 4098.0 * 0.6108 * (2.71828 ** (17.27 * weather.tavg / (weather.tavg + 237.3))) / (weather.tavg + 237.3) ** 2
    gamma = 0.067  # psychrometric constant kPa °C-1
    rn = weather.srad * 0.77 * (1.0 - soilprop.salb) - 4.9e-9 * ((weather.tmax + 273.0) ** 4 + (weather.tmin + 273.0) ** 4) / 2.0 * (0.34 - 0.14 * weather.vapr ** 0.5) * (1.35 * weather.srad / max(weather.srad + 0.01, 0.01) - 0.35)
    rn = max(rn, 0.0)
    etp = 1.28 * delta / (delta + gamma) * rn / 2.45  # mm d-1

    # Split into E and T based on LAI
    if hasattr(crop_model, "growth"):
        lai = crop_model.growth.lai
    else:
        lai = 0.0
    frac_t = min(1.0 - 1.0 * 2.71828 ** (-0.7 * lai), 1.0)
    eop = etp * frac_t
    es = etp * (1.0 - frac_t)
    return max(es, 0.0), max(eop, 0.0)
