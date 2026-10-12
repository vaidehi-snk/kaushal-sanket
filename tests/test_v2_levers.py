from engine.v2 import build, ipcc


def _rows():
    return [r for r in build(write=False)["rows"] if r["level"] == "district_trade" and r["status"] != "insufficient_evidence"]


def test_calls_are_at_least_very_likely():
    for r in _rows():
        if r["status"] in ("shortage", "saturation"):
            p = r["p_shortage"] if r["status"] == "shortage" else r["p_saturation"]
            assert p >= 0.89, (r["district"], r["trade"], p)   # whole 80% range beyond threshold == one-sided 90%


def test_planted_levers_are_found():
    lv = {(r["district"], r["trade"]): r for r in _rows()}
    assert lv[("Thane", "Solar PV Installer")]["lever"] == "fix_uptake"
    assert lv[("Thane", "Solar PV Installer")]["change"] == 0 and all(c["change"] == 0 for c in lv[("Thane", "Solar PV Installer")]["centres"])
    assert lv[("Pune", "Automotive Body Repair Technician")]["lever"] == "review_course"


def test_deactivated_qp_is_flagged():
    for r in _rows():
        if r["trade"] == "Two-wheeler Service Technician":
            assert r["qp_status"] == "deactivated" and "deactivated" in r["action"]


def test_ipcc_words():
    assert [ipcc(p) for p in (.995, .95, .7, .5, .1)] == ["virtually certain", "very likely", "likely", "about as likely as not", "unlikely"]
