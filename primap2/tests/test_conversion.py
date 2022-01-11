import pytest

import primap2.pm2io as pm2io
import primap2.pm2io._conversion  # noqa: F401


class TestIPCCCodePrimapToPrimap2:
    @pytest.mark.parametrize(
        "code_in, expected_code_out",
        [
            ("IPC1A", "1.A"),
            ("CATM0EL", "M.0.EL"),
            ("IPC1A1B23", "1.A.1.b.ii.3"),
            ("1A1Bii3", "1.A.1.b.ii.3"),
            ("IPC_1.A.1.B.ii.3", "1.A.1.b.ii.3"),
            ("IPCM1B1C", "M.1.B.1.c"),
            ("M.1.B.1.C", "M.1.B.1.c"),
            ("M.1.B.1.C.", "M.1.B.1.c"),
            ("M1B1C", "M.1.B.1.c"),
        ],
    )
    def test_working(self, code_in, expected_code_out):
        assert (
            pm2io._conversion.convert_ipcc_code_primap_to_primap2(code_in)
            == expected_code_out
        )

    def test_too_short(self, caplog):
        assert (
            pm2io._conversion.convert_ipcc_code_primap_to_primap2("IPC") == "error_IPC"
        )
        assert "WARNING" in caplog.text
        assert (
            "Too short to be a PRIMAP IPCC code after "
            "removal of prefix." in caplog.text
        )

    def test_wrong_format(self, caplog):
        assert (
            pm2io._conversion.convert_ipcc_code_primap_to_primap2("IPD1A")
            == "error_IPD1A"
        )
        assert "WARNING" in caplog.text
        # assert (
        #    "Prefix is missing or unknown, known codes are 'IPC' and 'CAT'. "
        #    "Assuming no code is present." in caplog.text
        # )
        assert "No digit found on first level." in caplog.text

    def test_end_after_m(self, caplog):
        assert (
            pm2io._conversion.convert_ipcc_code_primap_to_primap2("IPCM")
            == "error_IPCM"
        )
        assert "WARNING" in caplog.text
        assert "Nothing follows the 'M' for an 'M'-code." in caplog.text

    def test_first_lvl(self, caplog):
        assert (
            pm2io._conversion.convert_ipcc_code_primap_to_primap2("IPCA1")
            == "error_IPCA1"
        )
        assert "WARNING" in caplog.text
        assert "No digit found on first level." in caplog.text

    def test_second_lvl(self, caplog):
        assert (
            pm2io._conversion.convert_ipcc_code_primap_to_primap2("IPC123")
            == "error_IPC123"
        )
        assert "WARNING" in caplog.text
        assert "No letter found on second level." in caplog.text

    def test_third_lvl(self, caplog):
        assert (
            pm2io._conversion.convert_ipcc_code_primap_to_primap2("IPC1AC")
            == "error_IPC1AC"
        )
        assert "WARNING" in caplog.text
        assert "No number found on third level." in caplog.text

    def test_fourth_lvl(self, caplog):
        assert (
            pm2io._conversion.convert_ipcc_code_primap_to_primap2("IPC1A2_")
            == "error_IPC1A2_"
        )
        assert "WARNING" in caplog.text
        assert "No letter found on fourth level." in caplog.text

    def test_fifth_lvl(self, caplog):
        assert (
            pm2io._conversion.convert_ipcc_code_primap_to_primap2("IPC1A2BB")
            == "error_IPC1A2BB"
        )
        assert "WARNING" in caplog.text
        assert "No digit or roman numeral found on fifth level." in caplog.text

    def test_sixth_lvl(self, caplog):
        assert (
            pm2io._conversion.convert_ipcc_code_primap_to_primap2("IPC1A2B3X")
            == "error_IPC1A2B3X"
        )
        assert "WARNING" in caplog.text
        assert "No number found on sixth level." in caplog.text

    def test_after_sixth_lvl(self, caplog):
        assert (
            pm2io._conversion.convert_ipcc_code_primap_to_primap2("IPC1A2B33A")
            == "error_IPC1A2B33A"
        )
        assert "WARNING" in caplog.text
        assert "Chars left after sixth level." in caplog.text


class TestUnitPrimapToPrimap2:
    @pytest.mark.parametrize(
        "unit_in, entity_in, expected_unit_out",
        [
            ("GgCO2eq", "KYOTOGHG", "Gg CO2 / yr"),
            ("MtC", "CO", "Mt C / yr"),
            ("GgN2ON", "N2O", "Gg N / yr"),
            ("t", "CH4", "t CH4 / yr"),
        ],
    )
    def test_working(self, unit_in, entity_in, expected_unit_out):
        assert (
            pm2io._conversion.convert_unit_to_primap2(unit_in, entity_in)
            == expected_unit_out
        )

    def test_no_prefix(self, caplog):
        assert (
            pm2io._conversion.convert_unit_to_primap2("CO2eq", "FGASES")
            == "error_CO2eq_FGASES"
        )
        assert "WARNING" in caplog.text
        assert "No unit prefix matched for unit." in caplog.text

    def test_unit_empty(self, caplog):
        assert (
            pm2io._conversion.convert_unit_to_primap2("", "FGASES") == "error__FGASES"
        )
        assert "WARNING" in caplog.text
        assert "Input unit is empty. Nothing converted." in caplog.text

    def test_entity_empty(self, caplog):
        assert (
            pm2io._conversion.convert_unit_to_primap2("GgCO2eq", "") == "error_GgCO2eq_"
        )
        assert "WARNING" in caplog.text
        assert "Input entity is empty. Nothing converted." in caplog.text


@pytest.mark.parametrize(
    "entity_pm1, entity_pm2",
    [
        ("CO2", "CO2"),
        ("KYOTOGHG", "KYOTOGHG (SARGWP100)"),
        ("KYOTOGHGAR4", "KYOTOGHG (AR4GWP100)"),
    ],
)
def test_convert_entity_gwp_primap_to_primap2(entity_pm1, entity_pm2):
    assert (
        pm2io._conversion.convert_entity_gwp_primap_to_primap2(entity_pm1) == entity_pm2
    )
