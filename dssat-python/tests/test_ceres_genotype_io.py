"""Tests for CERES genotype file I/O (CUL / ECO / SPE readers).

Verifies that parameters parsed from the actual genotype files in
``Data/Genotype/`` match the expected values from those files.
No values are invented — all expected values are read from the files themselves.
"""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate Data/Genotype
# ---------------------------------------------------------------------------

def _find_genotype_dir() -> Path:
    """Walk up from this file to find Data/Genotype."""
    p = Path(__file__).resolve().parent
    for _ in range(10):
        candidate = p / "Data" / "Genotype"
        if candidate.is_dir():
            return candidate
        p = p.parent
    pytest.skip("Data/Genotype not found — skipping genotype IO tests")


# ---------------------------------------------------------------------------
# Low-level parser tests (genotype_io.py)
# ---------------------------------------------------------------------------

class TestGenotypeIOLowLevel:
    """Tests for the generic read_genotype_row / parse_spe_inline parsers."""

    def test_maize_cul_ibm0001(self):
        from dssat.io.converters.genotype_io import read_genotype_row
        gdir = _find_genotype_dir()
        row = read_genotype_row(gdir / "MZCER048.CUL", "IB0001",
                                vrname_col="VRNAME", econame_col=None)
        assert row["VAR#"] == "IB0001"
        assert row["ECO#"] == "IB0001"
        assert row["P1"] == pytest.approx(110.0)
        assert row["P2"] == pytest.approx(0.3)
        assert row["P5"] == pytest.approx(685.0)
        assert row["G2"] == pytest.approx(907.9)
        assert row["G3"] == pytest.approx(6.6)
        assert row["PHINT"] == pytest.approx(38.9)

    def test_maize_eco_ibm0001(self):
        from dssat.io.converters.genotype_io import read_genotype_row
        gdir = _find_genotype_dir()
        row = read_genotype_row(gdir / "MZCER048.ECO", "IB0001",
                                id_col="ECO#", vrname_col=None,
                                econame_col="ECONAME")
        assert row["ECO#"] == "IB0001"
        assert row["TBASE"] == pytest.approx(8.0)
        assert row["TOPT"] == pytest.approx(34.0)
        assert row["RUE"] == pytest.approx(4.2)
        assert row["KCAN"] == pytest.approx(0.85)

    def test_maize_spe_prftc(self):
        from dssat.io.converters.genotype_io import parse_spe_inline
        gdir = _find_genotype_dir()
        spe = parse_spe_inline(gdir / "MZCER048.SPE")
        assert spe["PRFTC"] == pytest.approx([6.2, 16.5, 33.0, 44.0])
        assert spe["CO2X"] == pytest.approx(
            [0, 220, 280, 330, 400, 490, 570, 750, 990, 9999])
        assert spe["CO2Y"] == pytest.approx(
            [0.0, 0.85, 0.95, 1.0, 1.02, 1.04, 1.05, 1.06, 1.07, 1.08])

    def test_wheat_cul_ib1500(self):
        from dssat.io.converters.genotype_io import read_genotype_row
        gdir = _find_genotype_dir()
        row = read_genotype_row(gdir / "WHCER048.CUL", "IB1500",
                                vrname_col="VAR-NAME", econame_col=None)
        assert row["VAR#"] == "IB1500"
        assert row["VAR-NAME"] == "MANITOU"
        assert row["ECO#"] == "CAWH01"
        assert row["P1V"] == pytest.approx(9.333)
        assert row["P1D"] == pytest.approx(13.12)
        assert row["P5"] == pytest.approx(331.4)

    def test_wheat_eco_cawh01(self):
        from dssat.io.converters.genotype_io import read_genotype_row
        gdir = _find_genotype_dir()
        row = read_genotype_row(gdir / "WHCER048.ECO", "CAWH01",
                                id_col="ECO#", vrname_col=None,
                                econame_col=None)
        assert row["ECO#"] == "CAWH01"
        assert row["P1"] == pytest.approx(362.0)
        assert row["PARUE"] == pytest.approx(2.7)
        assert row["KCAN"] == pytest.approx(0.85)

    def test_wheat_spe_trphs(self):
        from dssat.io.converters.genotype_io import parse_spe_csreads
        gdir = _find_genotype_dir()
        spe = parse_spe_csreads(gdir / "WHCER048.SPE")
        assert spe["TRPHS"] == pytest.approx([0.0, 5.0, 25.0, 35.0])
        assert spe["CO2RF"][2] == pytest.approx(330.0)

    def test_millet_cul_var_name_column(self):
        """Verify 16-char VAR-NAME field is parsed correctly."""
        from dssat.io.converters.genotype_io import read_genotype_row
        gdir = _find_genotype_dir()
        row = read_genotype_row(gdir / "MLCER048.CUL", "990001",
                                vrname_col="VAR-NAME", econame_col=None)
        assert row["VAR#"] == "990001"
        assert row["VAR-NAME"] == "NORTH VARIETY"
        assert row["ECO#"] == "IB0001"
        assert row["P1"] == pytest.approx(120.0)

    def test_sorghum_cul_var_name_column(self):
        from dssat.io.converters.genotype_io import read_genotype_row
        gdir = _find_genotype_dir()
        row = read_genotype_row(gdir / "SGCER048.CUL", "IB0001",
                                vrname_col="VAR-NAME", econame_col=None)
        assert row["VAR#"] == "IB0001"
        assert row["VAR-NAME"] == "RIO"
        assert row["ECO#"] == "IB0001"
        assert row["P1"] == pytest.approx(430.0)
        assert row["PANTH"] == pytest.approx(617.5)


# ---------------------------------------------------------------------------
# High-level fileio loader tests
# ---------------------------------------------------------------------------

class TestMaizeFileio:
    def test_load_ceres_maize_params(self):
        from dssat.crop.ceres_maize.fileio import load_ceres_maize_params
        p = load_ceres_maize_params("IB0001")
        # CUL
        assert p.cul.varno == "IB0001"
        assert p.cul.p1 == pytest.approx(110.0)
        assert p.cul.p2 == pytest.approx(0.3)
        assert p.cul.p5 == pytest.approx(685.0)
        assert p.cul.g2 == pytest.approx(907.9)
        assert p.cul.g3 == pytest.approx(6.6)
        assert p.cul.phint == pytest.approx(38.9)
        assert p.cul.econo == "IB0001"
        # ECO
        assert p.eco.tbase == pytest.approx(8.0)
        assert p.eco.topt == pytest.approx(34.0)
        assert p.eco.rue == pytest.approx(4.2)
        assert p.eco.kcan == pytest.approx(0.85)
        assert p.eco.dsgft == pytest.approx(170.0)
        # SPE
        assert p.spe.prftc == pytest.approx((6.2, 16.5, 33.0, 44.0))
        assert p.spe.co2x[3] == pytest.approx(330.0)
        assert p.spe.co2y[3] == pytest.approx(1.0)

    def test_maize_dfault_fallback(self):
        """Unknown cultivar should fall back to DFAULT row."""
        from dssat.crop.ceres_maize.fileio import load_ceres_maize_params
        p = load_ceres_maize_params("UNKNOWN99")
        # Should return something (DFAULT row), not raise
        assert p.cul is not None


class TestWheatFileio:
    def test_load_ceres_wheat_params(self):
        from dssat.crop.ceres_wheat.fileio import load_ceres_wheat_params
        p = load_ceres_wheat_params("IB1500")
        # CUL
        assert p.cul.varno == "IB1500"
        assert p.cul.vrname == "MANITOU"
        assert p.cul.p1v == pytest.approx(9.333)
        assert p.cul.p1d == pytest.approx(13.12)
        assert p.cul.p5 == pytest.approx(331.4)
        assert p.cul.econo == "CAWH01"
        # ECO
        assert p.eco.p1 == pytest.approx(362.0)
        assert p.eco.parue == pytest.approx(2.7)
        assert p.eco.kcan == pytest.approx(0.85)
        # Resolved (CUL overrides ECO)
        assert p.p1 == pytest.approx(362.0)   # from ECO (CUL has 0)
        assert p.parue == pytest.approx(2.7)   # from ECO
        assert p.kcan == pytest.approx(0.85)   # always from ECO
        # SPE
        assert p.spe.trphs == pytest.approx((0.0, 5.0, 25.0, 35.0))
        assert p.spe.co2rf[2] == pytest.approx(330.0)


class TestMilletFileio:
    def test_load_ceres_millet_params(self):
        from dssat.crop.ceres_millet.fileio import load_ceres_millet_params
        p = load_ceres_millet_params("990001")
        # CUL
        assert p.cul.varno == "990001"
        assert p.cul.vrname == "NORTH VARIETY"
        assert p.cul.econo == "IB0001"
        assert p.cul.p1 == pytest.approx(120.0)
        assert p.cul.p2o == pytest.approx(12.0)
        assert p.cul.p2r == pytest.approx(125.0)
        assert p.cul.p5 == pytest.approx(360.0)
        # ECO
        assert p.eco.tbase == pytest.approx(10.0)
        assert p.eco.topt == pytest.approx(34.0)
        assert p.eco.rue == pytest.approx(4.0)
        assert p.eco.djti == pytest.approx(68.0)
        # SPE
        assert p.spe.prftc == pytest.approx((11.0, 22.0, 35.0, 48.0))
        assert p.spe.co2x[3] == pytest.approx(330.0)


class TestSorghumFileio:
    def test_load_ceres_sorghum_params(self):
        from dssat.crop.ceres_sorghum.fileio import load_ceres_sorghum_params
        p = load_ceres_sorghum_params("IB0001")
        # CUL
        assert p.cul.varno == "IB0001"
        assert p.cul.vrname == "RIO"
        assert p.cul.econo == "IB0001"
        assert p.cul.p1 == pytest.approx(430.0)
        assert p.cul.p2 == pytest.approx(102.0)
        assert p.cul.p2o == pytest.approx(11.6)
        assert p.cul.p2r == pytest.approx(24.0)
        assert p.cul.panth == pytest.approx(617.5)
        assert p.cul.p5 == pytest.approx(540.0)
        # ECO
        assert p.eco.tbase == pytest.approx(8.0)
        assert p.eco.rue == pytest.approx(3.2)
        assert p.eco.kcan == pytest.approx(0.85)
        assert p.eco.stpc == pytest.approx(0.1)
        assert p.eco.rtpc == pytest.approx(0.25)
        # SPE
        assert p.spe.prftc == pytest.approx((8.0, 20.0, 40.0, 44.0))
        assert p.spe.co2x[3] == pytest.approx(330.0)


# ---------------------------------------------------------------------------
# from_file classmethod tests
# ---------------------------------------------------------------------------

class TestFromFile:
    def test_maize_from_file(self):
        from dssat.crop.ceres_maize.phenology import MaizeCultivar
        cv = MaizeCultivar.from_file("IB0001")
        assert cv.varno == "IB0001"
        assert cv.p1 == pytest.approx(110.0)
        assert cv.tbase == pytest.approx(8.0)
        assert cv.rue == pytest.approx(4.2)
        # SPE accessible via _params
        assert cv._params.spe.prftc == pytest.approx((6.2, 16.5, 33.0, 44.0))

    def test_wheat_from_file(self):
        from dssat.crop.ceres_wheat.phenology import WheatCultivar
        cv = WheatCultivar.from_file("IB1500")
        assert cv.varno == "IB1500"
        assert cv.p1v == pytest.approx(9.333)
        assert cv.p1 == pytest.approx(362.0)      # resolved from ECO
        assert cv.parue == pytest.approx(2.7)      # resolved from ECO
        assert cv._params.spe.trphs == pytest.approx((0.0, 5.0, 25.0, 35.0))

    def test_millet_from_file(self):
        from dssat.crop.ceres_millet.phenology import MilletCultivar
        cv = MilletCultivar.from_file("990001")
        assert cv.varno == "990001"
        assert cv.p1 == pytest.approx(120.0)
        assert cv.p2o == pytest.approx(12.0)
        assert cv.djti == pytest.approx(68.0)
        assert cv.tbase == pytest.approx(10.0)
        assert cv._params.spe.prftc == pytest.approx((11.0, 22.0, 35.0, 48.0))

    def test_sorghum_from_file(self):
        from dssat.crop.ceres_sorghum.phenology import SorghumCultivar
        cv = SorghumCultivar.from_file("IB0001")
        assert cv.varno == "IB0001"
        assert cv.p1 == pytest.approx(430.0)
        assert cv.p2 == pytest.approx(102.0)       # photoperiod-sensitive base
        assert cv.panth == pytest.approx(617.5)
        assert cv.tbase == pytest.approx(8.0)
        assert cv._params.spe.prftc == pytest.approx((8.0, 20.0, 40.0, 44.0))


# ---------------------------------------------------------------------------
# Growth module SPE integration tests
# ---------------------------------------------------------------------------

class TestGrowthModuleSPEParams:
    """Growth modules should load SPE parameters from the cultivar._params."""

    def test_maize_growth_spe_params(self):
        from dssat.crop.ceres_maize.phenology import MaizeCultivar
        from dssat.crop.ceres_maize.growth import MaizeGrowth
        cv = MaizeCultivar.from_file("IB0001")
        g = MaizeGrowth(cv, pltpop=6.0)
        assert g._prftc == pytest.approx((6.2, 16.5, 33.0, 44.0))
        assert g._co2x[3] == pytest.approx(330.0)
        assert g._co2y[3] == pytest.approx(1.0)
        assert g._co2x[-1] == pytest.approx(9999.0)

    def test_wheat_growth_spe_params(self):
        from dssat.crop.ceres_wheat.phenology import WheatCultivar
        from dssat.crop.ceres_wheat.growth import WheatGrowth
        cv = WheatCultivar.from_file("IB1500")
        g = WheatGrowth(cv, pltpop=200.0)
        assert g._prftc == pytest.approx((0.0, 5.0, 25.0, 35.0))
        assert g._co2x[2] == pytest.approx(330.0)

    def test_millet_growth_spe_params(self):
        from dssat.crop.ceres_millet.phenology import MilletCultivar
        from dssat.crop.ceres_millet.growth import MilletGrowth
        cv = MilletCultivar.from_file("990001")
        g = MilletGrowth(cv, pltpop=10.0)
        assert g._prftc == pytest.approx((11.0, 22.0, 35.0, 48.0))
        assert g._rgfilc == pytest.approx((7.0, 22.0, 27.0, 60.0))
        assert g._co2x[3] == pytest.approx(330.0)
        assert g._co2x[-1] == pytest.approx(9999.0)

    def test_sorghum_growth_spe_params(self):
        from dssat.crop.ceres_sorghum.phenology import SorghumCultivar
        from dssat.crop.ceres_sorghum.growth import SorghumGrowth
        cv = SorghumCultivar.from_file("IB0001")
        g = SorghumGrowth(cv, pltpop=10.0)
        assert g._prftc == pytest.approx((8.0, 20.0, 40.0, 44.0))
        assert g._rgfilc == pytest.approx((7.0, 22.0, 27.0, 35.0))
        assert g._co2x[3] == pytest.approx(330.0)
        assert g._co2x[-1] == pytest.approx(9999.0)
        # Partitioning from ECO
        assert g._stpc == pytest.approx(0.1)
        assert g._rtpc == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Sorghum phenology P2 fix
# ---------------------------------------------------------------------------

class TestSorghumPhenologyP2:
    """Verify the sorghum P2 field is correctly set from the CUL file."""

    def test_sorghum_p2_initialized_from_cul(self):
        from dssat.crop.ceres_sorghum.phenology import SorghumCultivar, SorghumPhenology
        cv = SorghumCultivar.from_file("IB0001")
        # P2 from CUL is 102 for IB0001
        assert cv.p2 == pytest.approx(102.0)
        pheno = SorghumPhenology(cv, pltpop=10.0, sdepth=4.0, yrsim=2020001)
        # After init, state.p2 should be cv.p2 (102), not cv.p1 (430)
        pheno._init(2020001)
        assert pheno.state.p2 == pytest.approx(102.0)


# ---------------------------------------------------------------------------
# Millet phenology DJTI fix
# ---------------------------------------------------------------------------

class TestMilletPhenologyDJTI:
    """Verify the millet phenology uses DJTI from ECO for stage 2."""

    def test_millet_djti_initialized_from_eco(self):
        from dssat.crop.ceres_millet.phenology import MilletCultivar, MilletPhenology
        cv = MilletCultivar.from_file("990001")
        # DJTI from ECO IB0001 is 68
        assert cv.djti == pytest.approx(68.0)
        pheno = MilletPhenology(cv, pltpop=10.0, sdepth=4.0, yrsim=2020001)
        pheno._init(2020001)
        # state.p2 should be cv.djti (68), not cv.p1 (120)
        assert pheno.state.p2 == pytest.approx(68.0)
