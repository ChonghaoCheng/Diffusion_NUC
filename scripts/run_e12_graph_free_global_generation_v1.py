#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"scripts"))
from e12_runner_support import collect_train_val,encode,evaluate_frozen_candidates,freeze_generated_candidates,merge_teacher_shards,oracle_relift,prepare,repair_e11_diagnostics,report_e12,run_p_lazy,test_graph_reference

def main():
 p=argparse.ArgumentParser();p.add_argument("--stage",required=True,choices=("prepare","repair-diagnostics","encode","oracle-relift","collect-train-val","merge-teacher-shards","freeze-validation","validate-pilot","freeze-test","p-lazy","test-graph-free","test-graph-reference","report"));p.add_argument("--output",type=Path,default=ROOT/"results/e12_graph_free_global_generation_v1");p.add_argument("--scene-ids",default=None,help="comma-separated deterministic teacher shard; only valid for collect-train-val");a=p.parse_args();c=json.loads((ROOT/"configs/e12_graph_free_global_generation_v1.json").read_text());o=a.output.resolve();o.mkdir(parents=True,exist_ok=True)
 if a.stage=="freeze-validation":freeze_generated_candidates(ROOT,c,o,"VALIDATION")
 elif a.stage=="validate-pilot":evaluate_frozen_candidates(ROOT,c,o,"VALIDATION")
 elif a.stage=="freeze-test":freeze_generated_candidates(ROOT,c,o,"SEALED_TEST")
 elif a.stage=="p-lazy":run_p_lazy(ROOT,c,o)
 elif a.stage=="test-graph-free":evaluate_frozen_candidates(ROOT,c,o,"SEALED_TEST")
 elif a.stage=="test-graph-reference":test_graph_reference(ROOT,c,o)
 elif a.stage=="report":report_e12(ROOT,c,o)
 elif a.stage=="collect-train-val":collect_train_val(ROOT,c,o,None if a.scene_ids is None else set(a.scene_ids.split(",")))
 elif a.stage=="merge-teacher-shards":merge_teacher_shards(ROOT,c,o)
 else:{"prepare":prepare,"repair-diagnostics":repair_e11_diagnostics,"encode":encode,"oracle-relift":oracle_relift}[a.stage](ROOT,c,o)
if __name__=="__main__":main()
