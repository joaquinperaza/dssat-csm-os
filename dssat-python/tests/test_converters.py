"""Tests for DSSAT legacy file converters.

Run with::

    pytest tests/test_converters.py -v
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from dssat.io.converters import wth, sol, xfile


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

_WTH_TEXT = """\
$UFGA  GAINESVILLE, FLORIDA
@ INSI      LAT     LONG  ELEV   TAV   AMP REFHT WNDHT
  UFGA   29.630  -82.370    30  22.5   7.8   2.0   2.0
@DATE  SRAD  TMAX  TMIN  RAIN  DEWP  WIND  RHUM
82100  12.5  28.0  16.0   0.0  15.0 180.0  75.0
82101  14.2  30.2  18.3   5.2  14.0 150.0  80.0
82102   8.1  22.5  12.0  12.3  12.0 200.0  90.0
"""

_SOL_TEXT = """\
*SOILS: DSSAT Python test soil file

*IBMZ910014  IBSNAT      SIC    210 Millhopper Fine Sand
@SITE        COUNTRY          LAT     LONG SCS FAMILY
 Gainesville  USA            29.63  -82.37 Loamy, siliceous, hyperthermic Grossarenic
@SCOM  SALB  SLU1 SLDR SLRO SLNF SLPF SMHB SMPX SMKE
  -99   .18   6.0  .15  66. 1.00  .92 IB001 IB001 IB001
@  SLB  SLMH  SLLL  SDUL  SSAT  SRGF  SSKS  SBDM  SLOC  SLCL  SLSI  SLCF  SLNI  SLHW  SLHB  SCEC  SADC
    5   A11  .023  .086  .230  1.00  7.40  1.36  .900     2    16     0   .00   6.0  -99    .9  -99
   15   A12  .023  .086  .230  1.00  7.40  1.40  .690     2    16     0   .00   6.0  -99    .9  -99
   30   A21  .023  .086  .215   .65  3.68  1.47  .280     2    16     0   .00   6.0  -99    .5  -99

*UFIB000001  IBSNAT      SIL    150 Sandy Loam Test
@SITE        COUNTRY          LAT     LONG SCS FAMILY
 TestSite     USA            35.00  -90.00 Test family
@SCOM  SALB  SLU1 SLDR SLRO SLNF SLPF SMHB SMPX SMKE
  -99   .13   5.0  .20  73. 1.00 1.00 IB001 IB001 IB001
@  SLB  SLMH  SLLL  SDUL  SSAT  SRGF  SSKS  SBDM  SLOC  SLCL  SLSI  SLCF  SLNI  SLHW  SLHB  SCEC  SADC
   15   Ap   .080  .200  .380  1.00  1.20  1.45  1.200    10    25     0   .12   6.5  -99   12.0  -99
   30   A    .080  .200  .370   .60  0.90  1.50  0.800    10    25     0   .08   6.5  -99   10.0  -99
"""

_XFILE_TEXT = """\
*EXP.DETAILS: UFGA8201MZ MAIZE VALIDATION TEST

*GENERAL
@PEOPLE
 DSSAT Foundation
@ADDRESS
 -99
@NOTES
 Test experiment for unit testing

*TREATMENTS                        -------------FACTOR LEVELS------------
@N R O C TNAME.................... CU FL SA IC MP MI MF MR MC MT ME MH SM
 1 1 0 0 UFGA8201-1                 1  1  0  1  1  0  1  0  0  0  0  1  1

*CULTIVARS
@C CR INGENO CNAME
 1 MZ IB0001 DEKALB XL71A

*FIELDS
@L ID_FIELD WSTA....  FLSA  FLOB  FLDT  FLDD  FLDS  FLST SLTX  SLDP  ID_SOIL....    FLNAME
 1 UFGA8201 UFGA8201   0.0   0.0 DR000   0.0   0.0 00000 SIL    210  IBMZ910014     -99
@L ...XCRD ...YCRD .....ELEV .............AREA .SLEN .FLWR .SLAS FLHST FHDUR
 1 -82.370  29.630    30.000             1.000  100. 1.00  1.00   -99  -99

*INITIAL CONDITIONS
@C   PCR ICDAT  ICRT  ICND  ICRN  ICRE  ICWD ICRES ICREN ICREP ICRIP ICRID ICNAME
 1    MZ 82100   300.  100.   .80  1.00   100   -99   -99   -99   -99   -99 -99
@C   SLB  ICBL  SH2O  SNH4  SNO3
 1     5  .050   .086   .500  4.000
 1    15  .050   .086   .500  3.000
 1    30  .050   .086   .200  2.000

*PLANTING DETAILS
@P PDATE EDATE  PPOP  PPOE  PLME  PLDS  PLRS  PLRD  PLDP  PLWT  PAGE  PENV  PLPH  SPRL                        PLNAME
 1 82100   -99   7.2   7.2   S     R    76.0   0.   6.0   -99   -99   -99   -99   -99                         -99

*FERTILIZERS (INORGANIC)
@F FDATE  FMCD  FACD  FDEP  FAMN  FAMP  FAMK  FAMC  FAMO  FOCD FERNAME
 1 82100 FE005 AP001   0.0  46.0   0.0   0.0   0.0   0.0   -99 -99
 1 82130 FE005 AP001   0.0  80.0   0.0   0.0   0.0   0.0   -99 -99

*HARVEST DETAILS
@H HDATE  HSTG  HCOM HSIZE   HPC  HBPC  HNAME
 1   -99  GS000 HI1  S       100.   0.  -99

*SIMULATION CONTROLS
@N GENERAL     NYERS NREPS START SDATE RSEED SNAME.................... SMODEL
 1 GE              1     1     S 82100  2150 UFGA8201-1               MZCER
@N OPTIONS     WATER NITRO SYMBI PHOSP POTAS DISES  CHEM  TILL   CO2
 1 OP              Y     Y     N     N     N     N     N     N     M
@N METHODS     WTHER INCON LIGHT EVAPO INFIL PHOTO HYDRO NSWIT MESOM MESEV MESOL METMP MEHYD
 1 ME              M     M     E     R     S     R     R     1     G     S     2     R     D
"""


# ---------------------------------------------------------------------------
# WTH tests
# ---------------------------------------------------------------------------

class TestWTHConverter:
    def _write_tmp(self, suffix=".WTH") -> Path:
        tmp = Path(tempfile.mktemp(suffix=suffix))
        tmp.write_text(_WTH_TEXT, encoding="utf-8")
        return tmp

    def test_read_station_meta(self):
        with tempfile.NamedTemporaryFile(suffix=".WTH", mode="w",
                                         delete=False, encoding="utf-8") as f:
            f.write(_WTH_TEXT)
            p = Path(f.name)
        data = wth.read_wth(p)
        assert abs(data["lat"] - 29.63) < 0.01
        assert abs(data["lon"] - (-82.37)) < 0.01
        assert abs(data["tav"] - 22.5) < 0.01
        assert abs(data["tamp"] - 7.8) < 0.01
        assert data["insi"].strip() == "UFGA"

    def test_read_record_count(self):
        with tempfile.NamedTemporaryFile(suffix=".WTH", mode="w",
                                         delete=False, encoding="utf-8") as f:
            f.write(_WTH_TEXT)
            p = Path(f.name)
        data = wth.read_wth(p)
        assert len(data["records"]) == 3

    def test_read_date_expansion(self):
        with tempfile.NamedTemporaryFile(suffix=".WTH", mode="w",
                                         delete=False, encoding="utf-8") as f:
            f.write(_WTH_TEXT)
            p = Path(f.name)
        data = wth.read_wth(p)
        # 82100 → YYDDD → year=1982, doy=100 → 1982100
        assert data["records"][0]["date"] == 1982100

    def test_read_daily_values(self):
        with tempfile.NamedTemporaryFile(suffix=".WTH", mode="w",
                                         delete=False, encoding="utf-8") as f:
            f.write(_WTH_TEXT)
            p = Path(f.name)
        data = wth.read_wth(p)
        r = data["records"][0]
        assert abs(r["srad"] - 12.5) < 0.01
        assert abs(r["tmax"] - 28.0) < 0.01
        assert abs(r["tmin"] - 16.0) < 0.01
        assert r["rain"] == 0.0
        assert abs(r["windsp"] - 180.0) < 0.01

    def test_roundtrip(self):
        """Write then re-read should preserve key fields."""
        src_data = {
            "insi": "TEST",
            "lat": 35.0, "lon": -90.0, "elev": 100.0,
            "tav": 18.0, "tamp": 12.0,
            "refht": 2.0, "wndht": 2.0,
            "records": [
                {"date": 2020100, "srad": 15.0, "tmax": 25.0,
                 "tmin": 12.0, "rain": 3.5, "windsp": 120.0},
                {"date": 2020101, "srad": 18.0, "tmax": 27.0,
                 "tmin": 14.0, "rain": 0.0, "windsp": 90.0},
            ],
        }
        with tempfile.NamedTemporaryFile(suffix=".WTH", delete=False) as f:
            out_path = Path(f.name)
        wth.write_wth(src_data, out_path)
        back = wth.read_wth(out_path)
        assert abs(back["lat"] - 35.0) < 0.1
        assert abs(back["tav"] - 18.0) < 0.1
        assert len(back["records"]) == 2
        assert back["records"][0]["date"] == 2020100
        assert abs(back["records"][1]["tmax"] - 27.0) < 0.1

    def test_missing_value_not_in_record(self):
        """Missing values (``.`` tokens) should be omitted from output dict."""
        text = (
            "$TEST\n"
            "@ INSI      LAT     LONG  ELEV   TAV   AMP REFHT WNDHT\n"
            "  TEST   10.000   20.000   50  20.0   8.0   2.0   2.0\n"
            "@DATE  SRAD  TMAX  TMIN  RAIN  DEWP\n"
            "20100  15.0  30.0  18.0   .     .\n"
        )
        with tempfile.NamedTemporaryFile(suffix=".WTH", mode="w",
                                         delete=False, encoding="utf-8") as f:
            f.write(text)
            p = Path(f.name)
        data = wth.read_wth(p)
        r = data["records"][0]
        # "." in rain column → _val returns None → key omitted from record
        assert "rain" not in r or r["rain"] == 0.0
        assert "dewp" not in r  # . → omitted

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            wth.read_wth("/no/such/file.WTH")

    def test_4digit_year_output(self):
        data = {
            "insi": "AAAA", "lat": 0.0, "lon": 0.0, "elev": 0.0,
            "tav": 20.0, "tamp": 5.0,
            "records": [{"date": 2023050, "srad": 10.0, "tmax": 20.0,
                          "tmin": 10.0, "rain": 0.0}],
        }
        with tempfile.NamedTemporaryFile(suffix=".WTH", delete=False) as f:
            p = Path(f.name)
        wth.write_wth(data, p, use_4digit_year=True)
        content = p.read_text()
        assert "2023050" in content

    def test_2digit_year_output(self):
        data = {
            "insi": "AAAA", "lat": 0.0, "lon": 0.0, "elev": 0.0,
            "tav": 20.0, "tamp": 5.0,
            "records": [{"date": 1990120, "srad": 10.0, "tmax": 20.0,
                          "tmin": 10.0, "rain": 0.0}],
        }
        with tempfile.NamedTemporaryFile(suffix=".WTH", delete=False) as f:
            p = Path(f.name)
        wth.write_wth(data, p, use_4digit_year=False)
        content = p.read_text()
        assert "90120" in content  # 1990 → yy=90


# ---------------------------------------------------------------------------
# SOL tests
# ---------------------------------------------------------------------------

class TestSOLConverter:
    def _write_tmp(self) -> Path:
        tmp = Path(tempfile.mktemp(suffix=".SOL"))
        tmp.write_text(_SOL_TEXT, encoding="utf-8")
        return tmp

    def test_read_profile_count(self):
        p = self._write_tmp()
        profiles = sol.read_sol(p)
        assert len(profiles) == 2

    def test_read_profile_id(self):
        p = self._write_tmp()
        profiles = sol.read_sol(p)
        assert profiles[0]["id"] == "IBMZ910014"
        assert profiles[1]["id"] == "UFIB000001"

    def test_read_profile_meta(self):
        p = self._write_tmp()
        profiles = sol.read_sol(p)
        pr = profiles[0]
        assert abs(pr.get("salb", 0) - 0.18) < 0.01
        assert abs(pr.get("slpf", 0) - 0.92) < 0.01
        assert "Millhopper" in pr.get("description", "")

    def test_read_layer_count(self):
        p = self._write_tmp()
        profiles = sol.read_sol(p)
        assert len(profiles[0]["layers"]) == 3

    def test_read_layer_values(self):
        p = self._write_tmp()
        profiles = sol.read_sol(p)
        layer0 = profiles[0]["layers"][0]
        assert layer0["depth_bottom"] == 5.0
        assert abs(layer0["ll"] - 0.023) < 0.001
        assert abs(layer0["dul"] - 0.086) < 0.001
        assert abs(layer0["sat"] - 0.230) < 0.001
        assert abs(layer0["bd"] - 1.36) < 0.01

    def test_read_by_id(self):
        p = self._write_tmp()
        pr = sol.read_sol_profile(p, "IBMZ910014")
        assert pr is not None
        assert pr["id"] == "IBMZ910014"

    def test_read_by_id_not_found(self):
        p = self._write_tmp()
        pr = sol.read_sol_profile(p, "DOESNOTEXIST")
        assert pr is None

    def test_roundtrip(self):
        p = self._write_tmp()
        profiles = sol.read_sol(p)

        out = Path(tempfile.mktemp(suffix=".SOL"))
        sol.write_sol(profiles, out)
        back = sol.read_sol(out)

        assert len(back) == len(profiles)
        assert back[0]["id"].strip() == profiles[0]["id"].strip()
        assert len(back[0]["layers"]) == len(profiles[0]["layers"])
        assert abs(back[0]["layers"][0]["ll"] - profiles[0]["layers"][0]["ll"]) < 0.001

    def test_write_single_profile(self):
        p = self._write_tmp()
        profiles = sol.read_sol(p)
        out = Path(tempfile.mktemp(suffix=".SOL"))
        sol.write_sol(profiles[0], out)  # single dict, not list
        back = sol.read_sol(out)
        assert len(back) == 1

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            sol.read_sol("/no/such/file.SOL")

    def test_sol_to_json_file(self):
        p = self._write_tmp()
        json_out = Path(tempfile.mktemp(suffix=".json"))
        result = sol.sol_to_json_file(p, json_out)
        assert result.exists()
        with result.open() as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 2


# ---------------------------------------------------------------------------
# X-file tests
# ---------------------------------------------------------------------------

class TestXFileConverter:
    def _write_tmp(self) -> Path:
        tmp = Path(tempfile.mktemp(suffix=".MZX"))
        tmp.write_text(_XFILE_TEXT, encoding="utf-8")
        return tmp

    def test_read_experiment_id(self):
        p = self._write_tmp()
        exp = xfile.read_xfile(p)
        assert "UFGA8201MZ" in exp["experiment_id"]

    def test_read_crop_model(self):
        p = self._write_tmp()
        exp = xfile.read_xfile(p)
        assert exp["crop"]["model"] == "MZCER"

    def test_read_cultivar(self):
        p = self._write_tmp()
        exp = xfile.read_xfile(p)
        assert exp["crop"]["cultivar"]["id"] == "IB0001"
        assert "DEKALB" in exp["crop"]["cultivar"]["name"]

    def test_read_planting(self):
        p = self._write_tmp()
        exp = xfile.read_xfile(p)
        plt = exp.get("planting", {})
        assert plt["date"] == 1982100
        assert abs(plt["pltpop"] - 7.2) < 0.1
        assert abs(plt["sdepth"] - 6.0) < 0.1
        assert abs(plt["rowspc"] - 76.0) < 0.5

    def test_read_fertilizers(self):
        p = self._write_tmp()
        exp = xfile.read_xfile(p)
        ferts = exp.get("fertilizer", [])
        assert len(ferts) == 2
        assert ferts[0]["date"] == 1982100
        assert ferts[1]["date"] == 1982130
        assert abs(ferts[0]["amount"] - 46.0) < 0.1

    def test_read_simulation_controls(self):
        p = self._write_tmp()
        exp = xfile.read_xfile(p)
        sim = exp["simulation"]
        assert sim["start_date"] == 1982100
        assert sim["water"] == "Y"
        assert sim["nitrogen"] == "Y"

    def test_read_initial_conditions(self):
        p = self._write_tmp()
        exp = xfile.read_xfile(p)
        ic = exp.get("initial_conditions", {})
        assert len(ic["sw_layers"]) == 3
        assert abs(ic["sw_layers"][0] - 0.086) < 0.001
        assert abs(ic["no3_layers"][0] - 4.0) < 0.01

    def test_write_produces_valid_file(self):
        p = self._write_tmp()
        exp = xfile.read_xfile(p)
        out = Path(tempfile.mktemp(suffix=".MZX"))
        xfile.write_xfile(exp, out)
        content = out.read_text()
        assert "*EXP.DETAILS" in content
        assert "*TREATMENTS" in content
        assert "*CULTIVARS" in content
        assert "*PLANTING DETAILS" in content
        assert "*SIMULATION CONTROLS" in content

    def test_roundtrip_experiment_id(self):
        p = self._write_tmp()
        exp = xfile.read_xfile(p)
        out = Path(tempfile.mktemp(suffix=".MZX"))
        xfile.write_xfile(exp, out)
        back = xfile.read_xfile(out)
        assert back["experiment_id"] == exp["experiment_id"]

    def test_roundtrip_cultivar(self):
        p = self._write_tmp()
        exp = xfile.read_xfile(p)
        out = Path(tempfile.mktemp(suffix=".MZX"))
        xfile.write_xfile(exp, out)
        back = xfile.read_xfile(out)
        assert back["crop"]["cultivar"]["id"] == exp["crop"]["cultivar"]["id"]

    def test_xfile_to_json_file(self):
        p = self._write_tmp()
        json_out = Path(tempfile.mktemp(suffix=".json"))
        result = xfile.xfile_to_json_file(p, json_out)
        assert result.exists()
        with result.open() as f:
            data = json.load(f)
        assert "experiment_id" in data
        assert "_tables" not in data  # internal key stripped

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            xfile.read_xfile("/no/such/file.MZX")
