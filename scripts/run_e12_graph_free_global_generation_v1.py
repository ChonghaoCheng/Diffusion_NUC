#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"scripts"))
from e12_runner_support import collect_train_val,encode,oracle_relift,prepare,repair_e11_diagnostics

def main():
 p=argparse.ArgumentParser();p.add_argument("--stage",required=True,choices=("prepare","repair-diagnostics","encode","oracle-relift","collect-train-val"));p.add_argument("--output",type=Path,default=ROOT/"results/e12_graph_free_global_generation_v1");a=p.parse_args();c=json.loads((ROOT/"configs/e12_graph_free_global_generation_v1.json").read_text());o=a.output.resolve();o.mkdir(parents=True,exist_ok=True)
 {"prepare":prepare,"repair-diagnostics":repair_e11_diagnostics,"encode":encode,"oracle-relift":oracle_relift,"collect-train-val":collect_train_val}[a.stage](ROOT,c,o)
if __name__=="__main__":main()
