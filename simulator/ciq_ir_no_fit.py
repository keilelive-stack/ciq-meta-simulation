#!/usr/bin/env python3
"""CIQ-IR: high-precision, no-hard-coded-physical-constants measurement audit.

The one fixed 10 µm value is an *explicit chosen reference*, not a measured
fundamental constant. No G, c, h, or ħ enters the calculation from source code.
Only the Python standard library is required.
"""
from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal, InvalidOperation, localcontext

IR_REFERENCE_UM = Decimal("10.000000")  # Explicit user-chosen scale, not a physical constant.
UM_IN_M = Decimal("1e-6")              # SI unit conversion, not a physical constant.


def dec(value, name):
    """Parse a decimal string without passing through binary floating point."""
    if value is None or str(value).strip() == "":
        return None
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{name}: keine Dezimalzahl") from exc
    if not result.is_finite():
        raise ValueError(f"{name}: muss endlich sein")
    return result


def positive(x, field):
    if x is not None and x <= 0:
        raise ValueError(f"{field}: muss positiv sein")


def rs_from_delta(delta, r_a, r_b):
    """Invert static Schwarzschild redshift; no G, mass, or c is needed."""
    k = 1 + delta
    if k <= 0:
        raise ValueError("K_IR = 1 + delta muss positiv sein")
    den = k * k / r_a - 1 / r_b
    if den == 0:
        raise ValueError("Inversion unbestimmt: Nenner ist null")
    return (k * k - 1) / den


def derivative_rs(delta, r_a, r_b):
    """Exact derivative d(r_s)/d(delta) for partial uncertainty budget."""
    k = 1 + delta
    den = k * k / r_a - 1 / r_b
    return 2 * k * (1 / r_a - 1 / r_b) / (den * den)


def audit(payload: dict) -> dict:
    """Recompute on every call. Any missing independent input is reported, not fitted."""
    with localcontext() as ctx:
        ctx.prec = 80
        delta = dec(payload.get("delta_k"), "delta_k")
        sigma = dec(payload.get("sigma_delta_k"), "sigma_delta_k")
        r_a = dec(payload.get("r_a_m"), "r_a_m")
        r_b = dec(payload.get("r_b_m"), "r_b_m")
        m = dec(payload.get("mass_kg"), "mass_kg")
        lam_c = dec(payload.get("calibration_lambda_m"), "calibration_lambda_m")
        freq_c = dec(payload.get("calibration_frequency_hz"), "calibration_frequency_hz")
        g_reference = dec(payload.get("comparison_g_only"), "comparison_g_only")
        for name, value in [("r_a_m", r_a), ("r_b_m", r_b),
                            ("mass_kg", m), ("calibration_lambda_m", lam_c),
                            ("calibration_frequency_hz", freq_c)]:
            positive(value, name)
        if sigma is not None and sigma < 0:
            raise ValueError("sigma_delta_k: darf nicht negativ sein")
        source = str(payload.get("mass_provenance", "unknown"))
        vac = str(payload.get("vacuum_provenance", "unknown"))
        if source not in ("non_gravitational", "derived_from_g", "unknown"):
            raise ValueError("mass_provenance: ungültig")
        if vac not in ("documented", "unverified", "nonvacuum", "unknown"):
            raise ValueError("vacuum_provenance: ungültig")

        def s(number):
            return None if number is None else str(number)

        out = {
            "model": "static Schwarzschild exterior; same coherent signal, A below B",
            "fixed_reference_um": str(IR_REFERENCE_UM),
            "independently_measured_G": False,
            "vacuum_status": vac,
            "status": [],
            "K_IR": None,
            "q": None,
            "lambda_B_reference_m": None,
            "c_from_independent_channels_m_per_s": None,
            "r_s_m": None,
            "GM_m3_per_s2": None,
            "G_conditional_m3_per_kg_s2": None,
            "G_comparison_residual": None,
            "sigma_G_delta_only": None,
        }
        if vac != "documented":
            out["status"].append("Vakuum nicht dokumentiert oder nicht gegeben: Ursache einer Verschiebung offen")
        else:
            out["status"].append("Vakuum als dokumentiert angegeben; selbst nicht experimentell verifiziert")
        if delta is None:
            out["status"].append("delta_k fehlt; kein Phasentransfer bestimmbar")
        else:
            k = 1 + delta
            if k <= 0:
                raise ValueError("K_IR muss positiv sein")
            out["K_IR"] = s(k)
            out["q"] = s(1 / k)
            out["lambda_B_reference_m"] = s(IR_REFERENCE_UM * UM_IN_M * k)
            out["delta_lambda_reference_m"] = s(IR_REFERENCE_UM * UM_IN_M * delta)
            if sigma is not None:
                out["sigma_delta_lambda_reference_m"] = s(IR_REFERENCE_UM * UM_IN_M * sigma)

        c_m = None
        if lam_c is not None and freq_c is not None:
            c_m = lam_c * freq_c
            out["c_from_independent_channels_m_per_s"] = s(c_m)
            out["status"].append("c aus zwei deklarierten Messkanälen; deren Unabhängigkeit separat prüfen")
        else:
            out["status"].append("Unabhängige Wellenlänge und Frequenz zur Ermittlung von c fehlen")

        rs = None
        if delta is not None and r_a is not None and r_b is not None:
            if r_b <= r_a:
                out["status"].append("Für diese Orientierung muss r_B > r_A sein")
            else:
                rs = rs_from_delta(delta, r_a, r_b)
                if rs < 0 or rs >= r_a:
                    out["status"].append("Inversion außerhalb zulässiger statischer Schwarzschild-Außenmetrik")
                    rs = None
                else:
                    out["r_s_m"] = s(rs)
                    if sigma is not None:
                        out["sigma_rs_delta_only_m"] = s(abs(derivative_rs(delta, r_a, r_b)) * sigma)
        else:
            out["status"].append("Für r_s fehlen Phasenmessung oder beide unabhängigen Radien")

        gm = None
        if rs is not None and c_m is not None:
            gm = c_m * c_m * rs / 2
            out["GM_m3_per_s2"] = s(gm)
        if gm is not None and m is not None and source == "non_gravitational":
            g = gm / m
            out["G_conditional_m3_per_kg_s2"] = s(g)
            out["status"].append("G BEDINGT bestimmt: M als unabhängig deklariert, nicht unabhängig nachgeprüft")
            if sigma is not None:
                out["sigma_G_delta_only"] = s(abs(derivative_rs(delta, r_a, r_b)) * sigma * c_m * c_m / (2 * m))
            if g_reference is not None and g_reference > 0:
                out["G_comparison_residual"] = s(g / g_reference - 1)
        else:
            out["status"].append("G NICHT BESTIMMBAR: unabhängig gemessene c, M oder gültige Geometrie fehlt")
        if g_reference is not None:
            out["status"].append("comparison_g_only wird ausschließlich NACH der Inversion benutzt, niemals als Eingang")
        out["status"].append("Unsicherheit aus delta allein; M-, c-, Radius- und Modellfehler separat erforderlich")
        return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="JSON-Datei; sonst JSON auf stdin")
    args = parser.parse_args(argv)
    try:
        with open(args.input, encoding="utf-8") if args.input else sys.stdin as inp:
            data = json.load(inp)
        result = audit(data)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
