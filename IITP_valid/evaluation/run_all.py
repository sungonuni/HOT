"""Step 2 추론 평가 전체 실행: 모델 확인 → 메모리 계산 → 정확도 → 실제 메모리 측정·그림 → 판정 (models/ 의 파일만 사용).
Step 1 에서 경량 학습(NIPQ+HOT)으로 만든 W4A4 배포 파일을 원본 FP32 모델과 비교한다.

    python run_all.py --device cuda:0 --data <CIFAR-10 폴더>

입력 (Step 1 의 train/finetune.py 가 만든다)
  ../models/simclr_resnet18_fp32.pt   원본 (SimCLR FP32 사전학습 → FP32 fine-tuning)
  ../models/simclr_resnet18_w4a4.bin  경량 배포 파일 (SimCLR NIPQ+HOT 사전학습 → NIPQ+HOT fine-tuning → 4bit)
결과 (--output-dir, 기본 ./results):
  summary.txt, summary.json                      최종 결과·판정 (json 에 세부 수치, 실행 명령)
  simclr_resnet18_infer_memory_comparison.png    원본 FP32 / W4A4 메모리 타임라인 (같은 Y 축)
단계별 중간 파일은 임시 폴더에서 주고받고 끝나면 지운다.
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


def _w(s):  # 터미널 표시 폭 (한글 2칸)
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def ljust(s, n):
    return s + " " * max(0, n - _w(s))


def rjust(s, n):
    return " " * max(0, n - _w(s)) + s


def render(s):
    """summary.json 내용 → summary.txt 줄 목록 (README 결과 표와 같은 항목만)."""
    ok = lambda v: "PASS" if v >= TARGET else "FAIL"  # noqa: E731
    mib = lambda v: f"{v / MiB:.3f}"  # noqa: E731
    f, q = s["fp32"], s["w4a4"]
    cr, mr = q["memory_calc"]["reduction_pct"], q["memory_measured"]["reduction_pct"]
    pk = "peak_model_tensors_bytes"
    L, C = 26, 16
    line = lambda lab, *cs: ljust(lab, L) + "".join(rjust(c, C) for c in cs)  # noqa: E731
    lines = [f"목표: 원본 FP32 대비 추론 메모리 {TARGET:.0f}% 이상 절감", "",
             line("[Step 2 추론 평가]", "원본 FP32", f"경량 {QLABEL}", "절감률", "판정"),
             line("  정확도", f"{f['accuracy']:.2f}%", f"{q['accuracy']:.2f}%",
                  f"{q['accuracy'] - f['accuracy']:+.2f}%p", "-"),
             line("  추론 메모리 계산값 (MiB)", mib(f["memory_calc"]["total_bytes"]),
                  mib(q["memory_calc"]["total_bytes"]), f"{cr:.1f}%", ok(cr)),
             line("  추론 메모리 실측 (MiB)", mib(f["memory_measured"][pk]), mib(q["memory_measured"][pk]),
                  f"{mr:.1f}%", ok(mr)), ""]
    if not s["gates"]["model_check_ok"]:
        lines.append("모델 확인: FAIL (summary.json 의 model_check 참고)")
    lines.append(f"최종 판정: {'PASS' if s['all_gates_passed'] else 'FAIL'}")
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data", default=None, help="CIFAR-10 폴더 (없으면 IITP_valid/data 에 자동 다운로드)")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results")
    p.add_argument("--chunk", type=int, default=16, help="경량 모델 런타임: 채널 몇 개씩 복원 (0 = 층 전체)")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"], help="경량 모델 런타임: 복원 정밀도")
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
                raise RuntimeError(f"{label} 실패 (위 출력 참고)")

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
               "input": "1x3x32x32 (이미지 1장)", "device": a.device, "gpu": mf["gpu"], "torch": mf["torch"],
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
    print("\n" + "=" * 100 + "\n" + "\n".join(lines) + f"\n\n결과: {out}/summary.txt, summary.json, "
          "simclr_resnet18_infer_memory_comparison.png", flush=True)
    if not summary["all_gates_passed"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
