import numpy as np
from pathlib import Path
from dssat.core.constants import NL, RUNINIT, SEASINIT, RATE, INTEGR
from dssat.core.types import SoilType, ControlType, SwitchType
from dssat.core.date_utils import incdat
from dssat.crop.ceres_maize import CeresMaize, MaizeCultivar
from dssat.weather.weather import WeatherModule
from dssat.io.converters import xfile, wth, sol, cul

base_dir = Path("tests/test_data/EJEMPLO1")
exp = xfile.read_xfile(base_dir / "EJEMPLO1.MZX")
wth_data = wth.read_wth(base_dir / "UYLE6601.WTH")
daily_records_dict = {r["date"]: r for r in wth_data["records"]}
sol_prof = sol.read_sol_profile(base_dir / "WI.SOL", "WI_LVUY018")
cultivars = cul.read_cul(base_dir / "MZIXM048.CUL")

def get_soil():
    soil = SoilType()
    soil.slno = sol_prof.get("id", "WI_LVUY018")
    soil.salb = float(sol_prof.get("salb", 0.13))
    soil.slpf = float(sol_prof.get("slpf", 1.0))
    phys_layers = [lyr for lyr in sol_prof.get("layers", []) if "ll" in lyr]
    soil.nlayr = min(len(phys_layers), NL)
    for i in range(soil.nlayr):
        lyr = phys_layers[i]
        depth_bottom = float(lyr.get("depth_bottom", 20.0 * (i+1)))
        soil.dlayr[i] = depth_bottom if i == 0 else depth_bottom - float(phys_layers[i-1].get("depth_bottom", 0))
        soil.ds[i] = depth_bottom
        soil.ll[i] = float(lyr.get("ll", 0.1))
        soil.dul[i] = float(lyr.get("dul", 0.2))
        soil.sat[i] = float(lyr.get("sat", 0.4))
        soil.swcn[i] = float(lyr.get("swcn", 0.5))
        soil.bd[i] = float(lyr.get("bd", 1.3))
        soil.shf[i] = float(lyr.get("shf", 1.0))
        soil.kg2ppm[i] = 10.0 * soil.bd[i] * soil.dlayr[i]
    return soil

treatments = exp["_tables"]["CULTIVARS"]
pltpop = exp.get("planting", {}).get("pltpop", 7.0)
sdepth = exp.get("planting", {}).get("sdepth", 5.0)
rowspc = exp.get("planting", {}).get("rowspc", 75.0)
yrsim = wth_data["records"][0]["date"]

for trt in treatments:
    trt_id = trt.get("C", "?")
    cname = trt.get("CNAME", "Unknown")
    cultivar_id = trt["INGENO"]
    
    cv_data = next((c for c in cultivars if c.get("VAR#", "") == cultivar_id), None)
    if not cv_data:
        print(f"Treatment {trt_id}: Cultivar {cultivar_id} not found!")
        continue
        
    cv = MaizeCultivar(
        varno=cultivar_id,
        vrname=cv_data.get("VRNAME", "Unknown"),
        econo=cv_data.get("ECO#", "IB0001"),
        p1=cv_data.get("P1", 200.0),
        p2=cv_data.get("P2", 0.4),
        p5=cv_data.get("P5", 700.0),
        g2=cv_data.get("G2", 700.0),
        g3=cv_data.get("G3", 8.0),
        phint=cv_data.get("PHINT", 38.9),
        tbase=8.0, topt=34.0, ropt=26.0, p2o=12.5, gdde=6.0, dsgft=150.0, rue=4.0, kcan=0.85
    )
    
    soil = get_soil()
    
    model = CeresMaize(
        cultivar=cv, pltpop=pltpop, sdepth=sdepth, yrsim=yrsim,
        rowspc=rowspc, slpf=soil.slpf, yrplt=yrsim,
    )
    
    weather_mod = WeatherModule(
        daily_records=daily_records_dict,
        xlat=wth_data.get("lat", -33.0), xlong=wth_data.get("lon", -58.0), xelev=wth_data.get("elev", 0.0),
        tav=wth_data.get("tav", 18.0), tamp=wth_data.get("tamp", 10.0), co2=380.0
    )
    
    iswitch = SwitchType()
    iswitch.iswwat = "Y"
    iswitch.iswnit = "Y"
    
    ctrl = ControlType()
    ctrl.yrdoy = yrsim
    ctrl.yrsim = yrsim
    ctrl.dynamic = RUNINIT
    
    sw = np.array(soil.dul)
    nh4 = np.zeros(NL)
    no3 = np.zeros(NL)
    
    weather_mod.run(ctrl, iswitch)
    model.run(ctrl, iswitch, soil, weather_mod.weather, sw, nh4, no3, 0.0, 0.0)
    
    ctrl.dynamic = SEASINIT
    weather_mod.run(ctrl, iswitch)
    model.run(ctrl, iswitch, soil, weather_mod.weather, sw, nh4, no3, 0.0, 0.0)
    
    matured = False
    for i in range(300):
        ctrl.yrdoy = incdat(yrsim, i)
        if ctrl.yrdoy not in daily_records_dict:
            break
            
        ctrl.dynamic = RATE
        weather_mod.run(ctrl, iswitch)
        model.run(ctrl, iswitch, soil, weather_mod.weather, sw, nh4, no3, 0.0, 0.0)
        
        ctrl.dynamic = INTEGR
        model.run(ctrl, iswitch, soil, weather_mod.weather, sw, nh4, no3, 0.0, 0.0)
        
        if model.pheno.istage == 6:
            matured = True
            break
            
    if matured:
        print(f"Treatment {trt_id} ({cname}): Yield = {model.growth.yield_:.1f} kg/ha (Matured on {ctrl.yrdoy})")
    else:
        print(f"Treatment {trt_id} ({cname}): Did not mature within weather records! Last stage = {model.pheno.istage}")

