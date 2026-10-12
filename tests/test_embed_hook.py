import numpy as np, hashlib
from engine.coding.nco_coder import NCOCoder
import calibrate_embeddings as cal


def fake(xs):
    out = []
    for x in xs:
        v = np.zeros(64)
        for w in x.split(): v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        out.append(v / (np.linalg.norm(v) or 1))
    return np.array(out)


def test_encoder_hook_and_calibration_run():
    assert NCOCoder(encoder=fake, th=(.8, .68, .55)).predict("plumber")["top_k"]
    res, t = cal.run(fake)
    assert set(res) == {"en", "hi", "mr"} and 0.3 <= t <= 0.99


def test_pilot_titles_loaded_and_junk_abstains():
    from engine.coding.nco_coder import load_pilot
    assert len(load_pilot()) == 90
    c = NCOCoder()
    assert c.predict("Solar Panel Installation Technician")["status"] == "yes"      # code confirmed in official files
    r = c.predict("Electrician, General"); assert r["code"].startswith("7411") and r["status"] == "pending"
    assert c.predict("qqzz xxvv")["level"] == 0
