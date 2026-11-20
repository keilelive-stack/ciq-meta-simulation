"""New independent ENGINE tests; not a replay of all historical CIQ tests."""
import unittest
from decimal import Decimal as D

from ciq_ir_no_fit import audit, rs_from_delta


class NoFitAudit(unittest.TestCase):
    def test_small_delta_survives(self):
        a = audit({"delta_k": "1e-19"})
        self.assertEqual(D(a["K_IR"]) - 1, D("1e-19"))
        self.assertEqual(D(a["delta_lambda_reference_m"]), D("1e-24"))

    def test_no_const_no_g(self):
        a = audit({"delta_k": "9.8e-20", "sigma_delta_k": "2.3e-20"})
        self.assertIsNone(a["G_conditional_m3_per_kg_s2"])
        self.assertIsNone(a["GM_m3_per_s2"])

    def test_synthetic_radius_round_trip(self):
        # Pure geometric sample; no G or c used to generate shift.
        r_a, r_b, rs = D("100000"), D("200000"), D("2953.34")
        q = ((1 - rs / r_a) / (1 - rs / r_b)).sqrt()
        delta = 1 / q - 1
        self.assertLess(abs(rs_from_delta(delta, r_a, r_b) - rs), D("1e-18"))

    def test_reject_mass_provenance_from_g(self):
        payload = {"delta_k": "0.01", "r_a_m": "100", "r_b_m": "200",
                   "mass_kg": "10", "mass_provenance": "derived_from_g",
                   "calibration_lambda_m": "1", "calibration_frequency_hz": "3"}
        result = audit(payload)
        self.assertIsNotNone(result["GM_m3_per_s2"])
        self.assertIsNone(result["G_conditional_m3_per_kg_s2"])

    def test_g_computed_only_with_provenance(self):
        payload = {"delta_k": "0.01", "r_a_m": "100", "r_b_m": "200",
                   "mass_kg": "10", "mass_provenance": "non_gravitational",
                   "calibration_lambda_m": "1", "calibration_frequency_hz": "3"}
        g = audit(payload)
        self.assertIsNotNone(g["G_conditional_m3_per_kg_s2"])
        self.assertEqual(D(g["c_from_independent_channels_m_per_s"]), 3)

    def test_comparison_cannot_fit_result(self):
        payload = {"delta_k": "0.01", "r_a_m": "100", "r_b_m": "200",
                   "mass_kg": "10", "mass_provenance": "non_gravitational",
                   "calibration_lambda_m": "1", "calibration_frequency_hz": "3"}
        a = audit(payload)
        b = audit(dict(payload, comparison_g_only="1e-9"))
        self.assertEqual(a["G_conditional_m3_per_kg_s2"], b["G_conditional_m3_per_kg_s2"])

    def test_vacuum_is_not_inferred_from_residual(self):
        a = audit({"delta_k": "1e-19", "vacuum_provenance": "unknown"})
        self.assertEqual(a["vacuum_status"], "unknown")
        self.assertFalse(a["independently_measured_G"])

    def test_reject_invalid_exterior(self):
        a = audit({"delta_k": "-0.1", "r_a_m": "100", "r_b_m": "200"})
        self.assertIsNone(a["r_s_m"])

    def test_uncertainty_is_partial(self):
        a = audit({"delta_k": "0.01", "sigma_delta_k": "0.001",
                   "r_a_m": "100", "r_b_m": "200"})
        self.assertGreater(D(a["sigma_rs_delta_only_m"]), 0)

    def test_3c186_double_ratio(self):
        ratio = D("2.0529") / D("2.0256")
        self.assertLess(abs(ratio - D("1.01347748815165876777")), D("1e-18"))


if __name__ == "__main__":
    unittest.main()
