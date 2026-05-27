"""Tests for ECO and SPE file converters.

Tests round-trip conversion (file → JSON → file), value correctness,
and compatibility with actual DSSAT genotype files.

Run with::

    pytest tests/test_eco_spe_converters.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from dssat.io.converters import eco, spe


# ---------------------------------------------------------------------------
# Sample ECO text fixture
# ---------------------------------------------------------------------------

_ECO_TEXT = """\
*MAIZE ECOTYPE COEFFICIENTS: MZCER048 MODEL
!
! COEFF   DEFINITIONS
! =====   ===========
! ECO#    Code for the ecotype
! ECONAME Name of the ecotype
! TBASE   Base temperature (°C)
! TOPT    Optimal vegetative temperature (°C)
!
@ECO#  ECONAME.........  TBASE  TOPT ROPT   P20  DJTI  GDDE  DSGFT  RUE   KCAN  TSEN  CDAY
IB0001 GENERIC MIDWEST1    8.0 34.0  34.0  12.5   4.0   6.0   170.  4.2   0.85   6.0  15.0
IB0002 GENERIC MIDWEST2    8.0 34.0  34.0  12.5   4.0   6.0   170.  4.5   0.85   6.0  15.0
DFAULT DEFAULT             8.0 34.0  34.0  12.5   4.0   6.0   170.  4.2   0.85   6.0  15.0
"""

# ---------------------------------------------------------------------------
# Sample SPE text fixture
# ---------------------------------------------------------------------------

_SPE_TEXT = """\
*MAIZE SPECIES COEFFICIENTS: MZCER048 MODEL

*TEMPERATURE EFFECTS
!       TBASE TOP1  TOP2  TMAX
  PRFTC  6.2  16.5  33.0  44.0     !Effect of temperature on photosynthesis
  RGFIL  5.5  16.0  27.0  35.0     !Effect of temperature on relative grain filling rate

*PHOTOSYNTHESIS PARAMETERS
  PARSR   0.50      !Conversion of solar radiation to PAR
  CO2X     0   220   280   330   400   490   570   750   990  9999
  CO2Y  0.00  0.85  0.95  1.00  1.02  1.04  1.05  1.06  1.07  1.08

*STRESS RESPONSE
  FSLFW   0.050     !Fraction of leaf area senesced under 100% water stress, 1/day
  FSLFN   0.050     !Fraction of leaf area senesced under 100% nitrogen stress, 1/day

*SEED GROWTH PARAMETERS
  DSGT    21.0      !Maximum days from sowing to germination before seed dies.
  DGET   150.0      !Growing degree days between germination and emergence
  SWCG    0.02      !Minimum available soil water required for seed germination
"""


# ---------------------------------------------------------------------------
# ECO converter tests
# ---------------------------------------------------------------------------

class TestEcoRead:
    def test_read_returns_list(self, tmp_path):
        """read_eco should return a list of dicts."""
        f = tmp_path / "TEST.ECO"
        f.write_text(_ECO_TEXT)
        records = eco.read_eco(f)
        assert isinstance(records, list)
        assert len(records) == 3

    def test_eco_id_and_name_parsed(self, tmp_path):
        """ECO# and ECONAME should be correctly extracted."""
        f = tmp_path / "TEST.ECO"
        f.write_text(_ECO_TEXT)
        records = eco.read_eco(f)
        assert records[0]["ECO#"] == "IB0001"
        assert records[0]["ECONAME"] == "GENERIC MIDWEST1"
        assert records[2]["ECO#"] == "DFAULT"

    def test_numeric_values_parsed(self, tmp_path):
        """Numeric columns should be parsed as floats."""
        f = tmp_path / "TEST.ECO"
        f.write_text(_ECO_TEXT)
        records = eco.read_eco(f)
        assert records[0]["TBASE"] == pytest.approx(8.0)
        assert records[0]["TOPT"] == pytest.approx(34.0)
        assert records[0]["RUE"] == pytest.approx(4.2)
        assert records[0]["KCAN"] == pytest.approx(0.85)

    def test_comments_ignored(self, tmp_path):
        """Lines starting with '!' should be skipped."""
        f = tmp_path / "TEST.ECO"
        f.write_text(_ECO_TEXT)
        records = eco.read_eco(f)
        # Should have exactly 3 records (not counting comment lines)
        assert len(records) == 3


class TestEcoWrite:
    def test_roundtrip(self, tmp_path):
        """Read → write → read should yield same records."""
        f_in = tmp_path / "IN.ECO"
        f_in.write_text(_ECO_TEXT)
        records_in = eco.read_eco(f_in)

        f_out = tmp_path / "OUT.ECO"
        eco.write_eco(records_in, f_out, file_header="*MAIZE ECOTYPE COEFFICIENTS")

        records_out = eco.read_eco(f_out)
        assert len(records_out) == len(records_in)
        for r_in, r_out in zip(records_in, records_out):
            assert r_out["ECO#"] == r_in["ECO#"]
            assert r_out["TBASE"] == pytest.approx(r_in["TBASE"])
            assert r_out["RUE"] == pytest.approx(r_in["RUE"])

    def test_write_preserves_econame(self, tmp_path):
        """ECONAME (can contain spaces) should survive a round-trip."""
        f_in = tmp_path / "IN.ECO"
        f_in.write_text(_ECO_TEXT)
        records = eco.read_eco(f_in)

        f_out = tmp_path / "OUT.ECO"
        eco.write_eco(records, f_out, file_header="*MAIZE ECOTYPE COEFFICIENTS")
        records2 = eco.read_eco(f_out)

        assert records2[0]["ECONAME"] == "GENERIC MIDWEST1"
        assert records2[1]["ECONAME"] == "GENERIC MIDWEST2"


class TestEcoJson:
    def test_eco_to_json_produces_valid_json(self, tmp_path):
        """eco_to_json should return a valid JSON string."""
        f = tmp_path / "TEST.ECO"
        f.write_text(_ECO_TEXT)
        json_str = eco.eco_to_json(f)
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)
        assert len(parsed) == 3

    def test_eco_to_json_file(self, tmp_path):
        """eco_to_json_file should create a .json file."""
        f = tmp_path / "TEST.ECO"
        f.write_text(_ECO_TEXT)
        json_path = eco.eco_to_json_file(f)
        assert json_path.exists()
        assert json_path.suffix == ".json"
        parsed = json.loads(json_path.read_text())
        assert len(parsed) == 3

    def test_json_to_eco_roundtrip(self, tmp_path):
        """json_to_eco should reconstruct a readable ECO file."""
        f_in = tmp_path / "TEST.ECO"
        f_in.write_text(_ECO_TEXT)

        json_str = eco.eco_to_json(f_in)
        f_out = tmp_path / "OUT.ECO"
        eco.json_to_eco(json_str, f_out, file_header="*MAIZE ECOTYPE COEFFICIENTS")

        records = eco.read_eco(f_out)
        assert records[0]["ECO#"] == "IB0001"
        assert records[0]["TBASE"] == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# SPE converter tests
# ---------------------------------------------------------------------------

class TestSpeRead:
    def test_read_returns_dict_with_sections(self, tmp_path):
        """read_spe should return a dict with 'sections' key."""
        f = tmp_path / "TEST.SPE"
        f.write_text(_SPE_TEXT)
        data = spe.read_spe(f)
        assert isinstance(data, dict)
        assert "sections" in data
        assert isinstance(data["sections"], dict)

    def test_sections_identified(self, tmp_path):
        """Section names from '*'-prefixed lines should be keys."""
        f = tmp_path / "TEST.SPE"
        f.write_text(_SPE_TEXT)
        data = spe.read_spe(f)
        sections = data["sections"]
        assert "TEMPERATURE EFFECTS" in sections
        assert "PHOTOSYNTHESIS PARAMETERS" in sections
        assert "STRESS RESPONSE" in sections
        assert "SEED GROWTH PARAMETERS" in sections

    def test_scalar_values(self, tmp_path):
        """Scalar parameter values should be parsed as float/int."""
        f = tmp_path / "TEST.SPE"
        f.write_text(_SPE_TEXT)
        data = spe.read_spe(f)
        assert data["sections"]["PHOTOSYNTHESIS PARAMETERS"]["PARSR"] == pytest.approx(0.50)
        assert data["sections"]["SEED GROWTH PARAMETERS"]["DSGT"] == pytest.approx(21.0)
        assert data["sections"]["SEED GROWTH PARAMETERS"]["DGET"] == pytest.approx(150.0)
        assert data["sections"]["SEED GROWTH PARAMETERS"]["SWCG"] == pytest.approx(0.02)

    def test_array_values(self, tmp_path):
        """Multi-value lines should be parsed as lists."""
        f = tmp_path / "TEST.SPE"
        f.write_text(_SPE_TEXT)
        data = spe.read_spe(f)
        prftc = data["sections"]["TEMPERATURE EFFECTS"]["PRFTC"]
        assert isinstance(prftc, list)
        assert len(prftc) == 4
        assert prftc[0] == pytest.approx(6.2)
        assert prftc[-1] == pytest.approx(44.0)

        co2x = data["sections"]["PHOTOSYNTHESIS PARAMETERS"]["CO2X"]
        assert isinstance(co2x, list)
        assert len(co2x) == 10
        assert co2x[0] == 0
        assert co2x[-1] == 9999

    def test_comments_stripped(self, tmp_path):
        """Inline comments after '!' should not appear in values."""
        f = tmp_path / "TEST.SPE"
        f.write_text(_SPE_TEXT)
        data = spe.read_spe(f)
        # PARSR value should be 0.50, not include the comment text
        parsr = data["sections"]["PHOTOSYNTHESIS PARAMETERS"]["PARSR"]
        assert isinstance(parsr, (int, float))
        assert parsr == pytest.approx(0.50)

    def test_file_header_captured(self, tmp_path):
        """The '*' file header line should be stored."""
        f = tmp_path / "TEST.SPE"
        f.write_text(_SPE_TEXT)
        data = spe.read_spe(f)
        assert "MAIZE SPECIES COEFFICIENTS" in data["_file_header"]


class TestSpeWrite:
    def test_roundtrip(self, tmp_path):
        """Read → write → read should yield same sections and values."""
        f_in = tmp_path / "IN.SPE"
        f_in.write_text(_SPE_TEXT)
        data_in = spe.read_spe(f_in)

        f_out = tmp_path / "OUT.SPE"
        spe.write_spe(data_in, f_out)

        data_out = spe.read_spe(f_out)
        assert set(data_out["sections"].keys()) == set(data_in["sections"].keys())

        for sect, params in data_in["sections"].items():
            for k, v in params.items():
                if isinstance(v, list):
                    for a, b in zip(v, data_out["sections"][sect][k]):
                        assert pytest.approx(a) == b
                else:
                    assert pytest.approx(v) == data_out["sections"][sect][k]


class TestSpeJson:
    def test_spe_to_json_valid(self, tmp_path):
        """spe_to_json should return valid JSON."""
        f = tmp_path / "TEST.SPE"
        f.write_text(_SPE_TEXT)
        json_str = spe.spe_to_json(f)
        parsed = json.loads(json_str)
        assert "sections" in parsed

    def test_spe_to_json_file(self, tmp_path):
        """spe_to_json_file should create a .json file."""
        f = tmp_path / "TEST.SPE"
        f.write_text(_SPE_TEXT)
        json_path = spe.spe_to_json_file(f)
        assert json_path.exists()
        assert json_path.suffix == ".json"

    def test_json_to_spe_roundtrip(self, tmp_path):
        """json_to_spe should reconstruct a readable SPE file."""
        f_in = tmp_path / "TEST.SPE"
        f_in.write_text(_SPE_TEXT)
        json_str = spe.spe_to_json(f_in)

        f_out = tmp_path / "OUT.SPE"
        spe.json_to_spe(json_str, f_out)

        data_out = spe.read_spe(f_out)
        assert "TEMPERATURE EFFECTS" in data_out["sections"]
        parsr = data_out["sections"]["PHOTOSYNTHESIS PARAMETERS"]["PARSR"]
        assert parsr == pytest.approx(0.50)

    def test_get_spe_param(self, tmp_path):
        """get_spe_param should find a value by keyword."""
        f = tmp_path / "TEST.SPE"
        f.write_text(_SPE_TEXT)
        val = spe.get_spe_param(f, "DSGT")
        assert val == pytest.approx(21.0)

    def test_get_spe_param_not_found(self, tmp_path):
        """get_spe_param should return None if keyword is absent."""
        f = tmp_path / "TEST.SPE"
        f.write_text(_SPE_TEXT)
        val = spe.get_spe_param(f, "NONEXISTENT_PARAM_XYZ")
        assert val is None

    def test_get_spe_param_with_section(self, tmp_path):
        """get_spe_param with section should restrict search."""
        f = tmp_path / "TEST.SPE"
        f.write_text(_SPE_TEXT)
        # Found in correct section
        val = spe.get_spe_param(f, "DSGT", section="SEED GROWTH PARAMETERS")
        assert val == pytest.approx(21.0)
        # Not in wrong section
        val_wrong = spe.get_spe_param(f, "DSGT", section="TEMPERATURE EFFECTS")
        assert val_wrong is None


# ---------------------------------------------------------------------------
# Integration: real DSSAT files (skipped if not present)
# ---------------------------------------------------------------------------

_GENOTYPE_DIR = Path(__file__).parent.parent.parent / "Data" / "Genotype"


@pytest.mark.skipif(
    not (_GENOTYPE_DIR / "MZCER048.ECO").exists(),
    reason="DSSAT genotype files not found",
)
class TestRealEcoFiles:
    def test_maize_eco_round_trip(self, tmp_path):
        """MZCER048.ECO should survive a read→write→read round-trip."""
        orig = _GENOTYPE_DIR / "MZCER048.ECO"
        records = eco.read_eco(orig)
        assert len(records) > 0

        out = tmp_path / "MZCER048_out.ECO"
        eco.write_eco(records, out, file_header=f"*MAIZE ECOTYPE COEFFICIENTS: MZCER048 MODEL")
        records2 = eco.read_eco(out)

        assert len(records2) == len(records)
        for r1, r2 in zip(records, records2):
            assert r1["ECO#"] == r2["ECO#"]

    def test_maize_eco_json_round_trip(self, tmp_path):
        """MZCER048.ECO JSON round-trip should preserve values."""
        orig = _GENOTYPE_DIR / "MZCER048.ECO"
        json_path = eco.eco_to_json_file(orig, tmp_path / "MZCER048.json")
        out = tmp_path / "MZCER048_from_json.ECO"
        eco.eco_from_json_file(json_path, out,
                                file_header="*MAIZE ECOTYPE COEFFICIENTS: MZCER048 MODEL")
        records = eco.read_eco(out)
        assert records[0]["ECO#"] == "IB0001"

    def test_sorghum_eco_readable(self):
        """SGCER048.ECO should be readable."""
        orig = _GENOTYPE_DIR / "SGCER048.ECO"
        if not orig.exists():
            pytest.skip("SGCER048.ECO not found")
        records = eco.read_eco(orig)
        assert len(records) > 0

    def test_millet_eco_readable(self):
        """MLCER048.ECO should be readable."""
        orig = _GENOTYPE_DIR / "MLCER048.ECO"
        if not orig.exists():
            pytest.skip("MLCER048.ECO not found")
        records = eco.read_eco(orig)
        assert len(records) > 0


@pytest.mark.skipif(
    not (_GENOTYPE_DIR / "MZCER048.SPE").exists(),
    reason="DSSAT genotype files not found",
)
class TestRealSpeFiles:
    def test_maize_spe_readable(self):
        """MZCER048.SPE should be readable without errors."""
        orig = _GENOTYPE_DIR / "MZCER048.SPE"
        data = spe.read_spe(orig)
        assert "sections" in data
        assert len(data["sections"]) > 0

    def test_maize_spe_seed_params(self):
        """MZCER048.SPE SEED section should have DSGT=21, DGET=150, SWCG=0.02."""
        orig = _GENOTYPE_DIR / "MZCER048.SPE"
        data = spe.read_spe(orig)
        seed_section = None
        for s_name, params in data["sections"].items():
            if "DSGT" in params:
                seed_section = params
                break
        assert seed_section is not None, "DSGT not found in any section"
        assert seed_section["DSGT"] == pytest.approx(21.0)
        assert seed_section["DGET"] == pytest.approx(150.0)
        assert seed_section["SWCG"] == pytest.approx(0.02)

    def test_maize_spe_prftc_array(self):
        """MZCER048.SPE PRFTC should be a 4-element array."""
        orig = _GENOTYPE_DIR / "MZCER048.SPE"
        data = spe.read_spe(orig)
        prftc = None
        for params in data["sections"].values():
            if "PRFTC" in params:
                prftc = params["PRFTC"]
                break
        assert prftc is not None
        assert isinstance(prftc, list)
        assert len(prftc) == 4

    def test_maize_spe_round_trip(self, tmp_path):
        """MZCER048.SPE should survive a read→write→read round-trip."""
        orig = _GENOTYPE_DIR / "MZCER048.SPE"
        data_in = spe.read_spe(orig)

        out = tmp_path / "MZCER048_out.SPE"
        spe.write_spe(data_in, out)
        data_out = spe.read_spe(out)

        # Same sections
        assert set(data_out["sections"].keys()) == set(data_in["sections"].keys())

    def test_maize_spe_json_round_trip(self, tmp_path):
        """MZCER048.SPE JSON round-trip should preserve DSGT value."""
        orig = _GENOTYPE_DIR / "MZCER048.SPE"
        json_path = spe.spe_to_json_file(orig, tmp_path / "MZCER048.json")
        out = tmp_path / "MZCER048_from_json.SPE"
        spe.spe_from_json_file(json_path, out)
        dsgt = spe.get_spe_param(out, "DSGT")
        assert dsgt == pytest.approx(21.0)

    def test_sorghum_spe_readable(self):
        """SGCER048.SPE should be readable."""
        orig = _GENOTYPE_DIR / "SGCER048.SPE"
        if not orig.exists():
            pytest.skip("SGCER048.SPE not found")
        data = spe.read_spe(orig)
        assert len(data["sections"]) > 0

    def test_wheat_spe_readable(self):
        """WHCER048.SPE should be readable without errors.

        Note: the WHCER048 (Cropsim-CERES) SPE file uses a different
        format (``$SPECIES`` header with ``@``-prefixed data tables) that
        differs from the standard keyword-value CERES-Maize/Sorghum/Millet
        format.  The reader should not crash; section count may be 0 for
        this format variant.
        """
        orig = _GENOTYPE_DIR / "WHCER048.SPE"
        if not orig.exists():
            pytest.skip("WHCER048.SPE not found")
        data = spe.read_spe(orig)
        # Just check the reader completes without error and returns a dict
        assert isinstance(data, dict)
        assert "sections" in data
