"""Step 1 학습 평가 전체 실행: 모델 확인 → 학습 메모리 계산 → 정확도 → 실제 학습 메모리 측정·그림 → 판정.
경량 NIPQ+HOT 으로 학습한 SimCLR + ResNet18 을 원본 FP32 로 학습한 SimCLR + ResNet18 과 비교한다.

    python run_all.py --device cuda:0 --data <CIFAR-10 폴더>

입력 (pretrain_simclr.py, finetune.py 로 만든다)
  ckpt/simclr_resnet18_pretrain_{fp32,nipq_hot}.pt    SimCLR 사전학습
  models/simclr_resnet18_fp32.pt                       원본 fine-tuning 결과
  ckpt/simclr_resnet18_finetune_nipq_hot.pt            경량 fine-tuning 결과
결과 (--output-dir, 기본 ./results):
  summary.txt, summary.json                       최종 결과·판정 (json 에 세부 수치, 실행 명령)
  simclr_resnet18_train_memory_comparison.png     학습 1 step 메모리 타임라인 (원본 / 경량 / 참고, 같은 Y 축)
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
sys.path.insert(0, str(ROOT.parent))
from core.common import BATCH, LABEL_FRAC, NAME, TRAIN_TARGET as TARGET, MiB  # noqa: E402

CALC_MODES = ["fp32", "nipq_hot", "nipq_hot_naive"]


def _w(s):  # 터미널 표시 폭 (한글 2칸)
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def ljust(s, n):
    return s + " " * max(0, n - _w(s))


def rjust(s, n):
    return " " * max(0, n - _w(s)) + s


def render(s):
    """summary.json 내용 → summary.txt 줄 목록 (README 결과 표와 같은 항목만)."""
    ok = lambda v: "PASS" if v >= TARGET else "FAIL"  # noqa: E731
    mib = lambda v: f"{v / MiB:.1f}"  # noqa: E731
    acc, f, q = s["accuracy"], s["memory"]["fp32"], s["memory"]["nipq_hot"]
    fa, ha = acc["fp32"]["acc"], acc["nipq_hot"]["acc"]
    cr, mr = q["calc"]["reduction_pct"], q["measured"]["reduction_pct"]
    pk = "peak_training_tensors_bytes"
    L, C = 34, 16
    line = lambda lab, *cs: ljust(lab, L) + "".join(rjust(c, C) for c in cs)  # noqa: E731
    lines = [f"목표: 원본 FP32 대비 학습 메모리 {TARGET:.0f}% 이상 절감", "",
             line("[Step 1 학습 평가]", "원본 FP32", "경량 NIPQ+HOT", "절감률", "판정"),
             line(f"  fine-tuning 정확도 (라벨 1/{round(1 / LABEL_FRAC)})", f"{fa:.2f}%", f"{ha:.2f}%",
                  f"{ha - fa:+.2f}%p", "-"),
             line("  학습 메모리 계산값 (MiB)", mib(f["calc"]["total_bytes"]), mib(q["calc"]["total_bytes"]),
                  f"{cr:.1f}%", ok(cr)),
             line("  학습 메모리 실측 (MiB)", mib(f["measured"][pk]), mib(q["measured"][pk]), f"{mr:.1f}%", ok(mr)),
             ""]
    if not s["model_check"]["ok"]:
        lines.append("모델 확인: FAIL (summary.json 의 model_check 참고)")
    lines.append(f"최종 판정: {'PASS' if s['all_gates_passed'] else 'FAIL'}")
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data", default=None, help="CIFAR-10 폴더 (없으면 IITP_valid/data 에 자동 다운로드)")
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
                raise RuntimeError(f"{label} 실패 (위 출력 참고)")

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
    print("\n" + "=" * 100 + "\n" + "\n".join(lines) + f"\n\n결과: {out}/summary.txt, summary.json, "
          f"{NAME}_train_memory_comparison.png", flush=True)
    if not summary["all_gates_passed"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
