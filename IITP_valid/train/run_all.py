"""Step 1 training evaluation: model check, memory calc, accuracy, measured memory, verdict.

    python run_all.py --device cuda:0
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from core.common import BATCH, LABEL_FRAC, NAME, TRAIN_TARGET as TARGET, MiB  # noqa: E402

CALC_MODES = ["fp32", "nipq_hot", "nipq_hot_naive"]


def _w(s):  # display width (CJK = 2)
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def ljust(s, n):
    return s + " " * max(0, n - _w(s))


def rjust(s, n):
    return " " * max(0, n - _w(s)) + s


def render(s):
    ok = lambda v: "PASS" if v >= TARGET else "FAIL"  # noqa: E731
    mib = lambda v: f"{v / MiB:.1f}"  # noqa: E731
    acc, f, q = s["accuracy"], s["memory"]["fp32"], s["memory"]["nipq_hot"]
    fa, ha = acc["fp32"]["acc"], acc["nipq_hot"]["acc"]
    cr, mr = q["calc"]["reduction_pct"], q["measured"]["reduction_pct"]
    pk = "peak_training_tensors_bytes"
    L, C = 38, 16
    line = lambda lab, *cs: ljust(lab, L) + "".join(rjust(c, C) for c in cs)  # noqa: E731
    lines = [f"Target: training memory reduction >= {TARGET:.0f}% vs original FP32", "",
             line("[Step 1 training]", "Original FP32", "NIPQ+HOT", "Reduction", "Verdict"),
             line(f"  Fine-tuning accuracy (1/{round(1 / LABEL_FRAC)} labels)", f"{fa:.2f}%", f"{ha:.2f}%",
                  f"{ha - fa:+.2f}%p", "-"),
             line("  Training memory calc (MiB)", mib(f["calc"]["total_bytes"]), mib(q["calc"]["total_bytes"]),
                  f"{cr:.1f}%", ok(cr)),
             line("  Training memory meas. (MiB)", mib(f["measured"][pk]), mib(q["measured"][pk]), f"{mr:.1f}%", ok(mr)),
             ""]
    if not s["model_check"]["ok"]:
        lines.append("Model check: FAIL (see model_check in summary.json)")
    lines.append(f"Final verdict: {'PASS' if s['all_gates_passed'] else 'FAIL'}")
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data", default=None, help="CIFAR-10 dir (downloaded to IITP_valid/data if omitted)")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results")
    a = p.parse_args()
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":0:0",
               PYTHONWARNINGS="ignore")
    data = ["--data", a.data] if a.data else []
    dev = ["--device", a.device]
    commands = []

    with tempfile.TemporaryDirectory(prefix="run_all_train_") as tmp:
        work = Path(tmp)

        def run(script, *opts):
            commands.append(["python", script, *map(str, opts)])
            label = " ".join([Path(script).stem] + [str(o) for o in opts if not str(o).startswith("--")][:1])
            print(f"\n▶ {label}", flush=True)
            cmd = [sys.executable, str(ROOT / script), *map(str, opts), "--output-dir", str(work)]
            if subprocess.run(cmd, cwd=ROOT, env=env).returncode:
                raise RuntimeError(f"{label} failed (see output above)")

        run("check_model.py", *dev, *data)
        run("memory_calc.py", *dev, *data)
        run("evaluate.py", *dev, *data)
        for m in CALC_MODES:
            run("memory_profile.py", "--mode", m, *dev, *data)
        run("memory_profile.py", "--combine")

        J = lambda name: json.loads((work / name).read_text())  # noqa: E731
        ck, calc, acc = J("check.json"), J("memory_calc.json"), J("acc.json")
        meas = {m: J(f"{m}_memory.json") for m in CALC_MODES}
        shutil.copy2(work / f"{NAME}_train_memory_comparison.png", out)

    pct = lambda q, f: 100 * (1 - q / f)  # noqa: E731
    pk = {m: meas[m]["peak_training_tensors_bytes"] for m in CALC_MODES}
    meas_pct = {m: pct(pk[m], pk["fp32"]) for m in CALC_MODES[1:]}
    gates = {"model_check_ok": ck["ok"], "memory_calc_ge_target": calc["nipq_hot"]["reduction_pct"] >= TARGET,
             "memory_measured_ge_target": meas_pct["nipq_hot"] >= TARGET}
    keys = ["weight_bytes", "gradient_bytes", "optimizer_state_bytes", "activation_bytes", "total_bytes"]
    summary = {"target_reduction_pct": TARGET, "model": "SimCLR + ResNet18 (CIFAR-10, 32x32)",
               "lightweight": "NIPQ forward (W4A4) + HOT INT8 backward",
               "batch": f"{BATCH} images x 2 views", "device": a.device, "gpu": meas["fp32"]["gpu"],
               "torch": meas["fp32"]["torch"], "accuracy": acc,
               "memory": {m: {"calc": {k: calc[m][k] for k in keys}, "measured": meas[m]} for m in CALC_MODES},
               "model_check": ck, "gates": gates, "all_gates_passed": all(gates.values())}
    for m in CALC_MODES[1:]:
        summary["memory"][m]["calc"]["reduction_pct"] = calc[m]["reduction_pct"]
        summary["memory"][m]["measured"]["reduction_pct"] = meas_pct[m]
    lines = render(summary)
    summary["commands"] = commands
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "=" * 100 + "\n" + "\n".join(lines) + f"\n\nResults: {out}/summary.txt, summary.json, "
          f"{NAME}_train_memory_comparison.png", flush=True)
    if not summary["all_gates_passed"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
