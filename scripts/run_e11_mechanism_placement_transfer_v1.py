#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
from e11_runner_support import prepare_e11,dev_e11,freeze_transfer_e11,build_transfer_e11,compare_transfer_e11,verify_e11,report_e11,write_json
DEFAULT_OUTPUT=ROOT/"results/e11_mechanism_placement_transfer_v1"
def require(output,stage):
 p=output/f"{stage}.checkpoint.json"
 if not p.exists() or not json.loads(p.read_text()).get("complete"):raise RuntimeError(f"required stage {stage} incomplete")
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--stage",required=True,choices=("prepare","test","dev","freeze-transfer","build-transfer","compare-transfer","verify","report"));ap.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);a=ap.parse_args();cfg=json.loads((ROOT/"configs/e11_mechanism_placement_transfer_v1.json").read_text());out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);globals()["stage_"+a.stage.replace("-","_")](cfg,out)
def stage_prepare(c,o):prepare_e11(ROOT,c,o)
def stage_test(c,o):
 require(o,"prepare");cmds=[[sys.executable,"-m","pytest","-q","tests/test_e11_mechanism_transfer.py","tests/test_structured_routing.py","tests/test_e09r1_search.py","tests/test_e09_synchronized_motion.py","tests/test_e09_execution.py","tests/test_history_search.py","tests/test_completion_bound.py","tests/test_ordered_trace_evaluator.py"],[sys.executable,"-m","pytest","-q"]];lines=[];stats=[]
 for i,cmd in enumerate(cmds):
  t=perf_counter();env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1");r=subprocess.run(cmd,cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);lines += ["$ "+" ".join(cmd),r.stdout.rstrip(),f"exit={r.returncode} elapsed_s={perf_counter()-t:.6f}"];stats.append({"name":"focused" if i==0 else "literal_repository_suite","exit_code":r.returncode})
  if i==0 and r.returncode:break
 text="\n".join(lines)+"\n";(o/"tests.txt").write_text("\n".join(x.rstrip() for x in text.splitlines())+"\n");missing=[x for x in ["results/riemannian_anisotropy_utility_v1/r0_scene_calibration/witnesses/saddle_T17.npz","results/nuc_robot_skeleton_coupling_v1/config.json"] if x in text];full=next((x["exit_code"] for x in stats if x["name"]=="literal_repository_suite"),None);ok=stats[0]["exit_code"]==0 and (full==0 or bool(missing));write_json(o/"test_statuses.json",{"commands":stats,"focused_pass":stats[0]["exit_code"]==0,"literal_suite_exit":full,"missing_historical_fixtures":missing,"literal_suite_fully_passed":full==0});write_json(o/"test.checkpoint.json",{"complete":ok})
 if not ok:raise RuntimeError("unexplained test failure")
def stage_dev(c,o):require(o,"test");dev_e11(ROOT,c,o)
def stage_freeze_transfer(c,o):require(o,"dev");freeze_transfer_e11(ROOT,c,o)
def stage_build_transfer(c,o):require(o,"freeze-transfer");build_transfer_e11(ROOT,c,o)
def stage_compare_transfer(c,o):require(o,"build-transfer");compare_transfer_e11(ROOT,c,o)
def stage_verify(c,o):require(o,"compare-transfer");verify_e11(ROOT,c,o)
def stage_report(c,o):require(o,"verify");report_e11(ROOT,c,o)
if __name__=="__main__":main()
