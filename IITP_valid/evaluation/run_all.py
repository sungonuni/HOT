"""Step 2 inference evaluation: model check, memory calc, accuracy, measured memory, verdict.

Inputs: ../models/simclr_resnet18_fp32.pt, ../models/simclr_resnet18_w4a4.bin

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
sys.path.insert(0, str(ROOT))
from common import A_BITS, QLABEL, TARGET, TITLE, W_BITS, MiB  # noqa: E402


def _w(s):  # display width (CJK = 2 columns)
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def ljust(s, n):
    return s + " " * max(0, n - _w(s))


def rjust(s, n):
    return " " * max(0, n - _w(s)) + s


def render(s):
    ok = lambda v: "PASS" if v >= TARGET else "FAIL"  # noqa: E731
    mib = lambda v: f"{v / MiB:.3f}"  # noqa: E731
    f, q = s["fp32"], s["w4a4"]
    cr, mr = q["memory_calc"]["reduction_pct"], q["memory_measured"]["reduction_pct"]
    pk = "peak_model_tensors_bytes"
    L, C = 38, 16
    line = lambda lab, *cs: ljust(lab, L) + "".join(rjust(c, C) for c in cs)  # noqa: E731
    lines = [f"Target: inference memory reduction >= {TARGET:.0f}% vs original FP32", "",
             line("[Step 2 inference]", "Original FP32", QLABEL, "Reduction", "Verdict"),
             line("  Accuracy", f"{f['accuracy']:.2f}%", f"{q['accuracy']:.2f}%",
                  f"{q['accuracy'] - f['accuracy']:+.2f}%p", "-"),
             line("  Inference memory calc (MiB)", mib(f["memory_calc"]["total_bytes"]),
                  mib(q["memory_calc"]["total_bytes"]), f"{cr:.1f}%", ok(cr)),
             line("  Inference memory meas. (MiB)", mib(f["memory_measured"][pk]), mib(q["memory_measured"][pk]),
                  f"{mr:.1f}%", ok(mr)), ""]
    if not s["gates"]["model_check_ok"]:
        lines.append("Model check: FAIL (see model_check in summary.json)")
    lines.append(f"Final verdict: {'PASS' if s['all_gates_passed'] else 'FAIL'}")
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data", default=None, help="CIFAR-10 dir (downloaded to IITP_valid/data if missing)")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results")
    p.add_argument("--chunk", type=int, default=16, help="W4A4 runtime: channels dequantized per chunk (0 = whole layer)")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"], help="W4A4 runtime: dequantization dtype")
    a = p.parse_args()
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":0:0",
               PYTHONWARNINGS="ignore")
    data = ["--data", a.data] if a.data else []
    rt = ["--chunk", str(a.chunk), "--dtype", a.dtype]
    commands = []

    with tempfile.TemporaryDirectory(prefix="run_all_infer_") as tmp:
        work = Path(tmp)

        def run(script, *opts):
            commands.append(["python", script, *map(str, opts)])
            label = " ".join([Path(script).stem] + [str(o) for o in opts if not str(o).startswith("--")][:1])
            print(f"\n▶ {label}", flush=True)
            cmd = [sys.executable, str(ROOT / script), *map(str, opts), "--output-dir", str(work)]
            if subprocess.run(cmd, cwd=ROOT, env=env).returncode:
                raise RuntimeError(f"{label} failed (see output above)")

        run("check_model.py")
        run("memory_calc.py")
        run("evaluate.py", "--mode", "fp32", "--device", a.device, *data)
        run("evaluate.py", "--mode", "w4a4", "--device", a.device, *data, *rt)
        run("memory_profile.py", "--mode", "fp32", "--device", a.device, *data)
        run("memory_profile.py", "--mode", "w4a4", "--device", a.device, *data, *rt)
        run("memory_profile.py", "--combine")

        J = lambda name: json.loads((work / name).read_text())  # noqa: E731
        ck, calc = J("check.json"), J("memory_calc.json")
        af, aq = J("fp32_acc.json"), J("w4a4_acc.json")
        mf, mq = J("fp32_memory.json"), J("w4a4_memory.json")
        shutil.copy2(work / "simclr_resnet18_infer_memory_comparison.png", out)

    cf, cq = calc["fp32"], calc["w4a4"]
    meas = 100 * (1 - mq["peak_model_tensors_bytes"] / mf["peak_model_tensors_bytes"])
    gates = {"model_check_ok": ck["ok"], "memory_calc_ge_target": cq["reduction_pct"] >= TARGET,
             "memory_measured_ge_target": meas >= TARGET}
    mem = lambda c: {k: c[k] for k in ["weight_bytes", "peak_activation_bytes", "total_bytes"]}  # noqa: E731
    meas_keys = ["weights_resident_bytes", "peak_model_tensors_bytes"]
    summary = {"target_reduction_pct": TARGET, "model": TITLE, "quant": {"weight_bits": W_BITS, "act_bits": A_BITS},
               "input": "1x3x32x32 (1 image)", "device": a.device, "gpu": mf["gpu"], "torch": mf["torch"],
               "runtime": {"chunk": a.chunk or "layer", "dtype": a.dtype},
               "fp32": {"accuracy": af["acc"], "memory_calc": mem(cf), "memory_measured": {k: mf[k] for k in meas_keys},
                        "structure_check": {**ck["structure"], "ok": ck["structure_ok"]}},
               "w4a4": {"accuracy": aq["acc"], "memory_calc": {**mem(cq), "reduction_pct": cq["reduction_pct"]},
                        "memory_measured": {**{k: mq[k] for k in meas_keys}, "reduction_pct": meas},
                        "model_check": {**ck["quant"]["bits"], **ck["quant"]["packed_file"], "ok": ck["quant"]["ok"]}},
               "gates": gates, "all_gates_passed": all(gates.values())}
    lines = render(summary)
    summary["commands"] = commands
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "=" * 100 + "\n" + "\n".join(lines) + f"\n\nResults: {out}/summary.txt, summary.json, "
          "simclr_resnet18_infer_memory_comparison.png", flush=True)
    if not summary["all_gates_passed"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
