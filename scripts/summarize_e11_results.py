#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/e11_mechanism_placement_transfer_v1"


def read(name):
    with (OUT/name).open(newline="") as stream:return list(csv.DictReader(stream))


def write(name,rows):
    rows=list(rows)
    with (OUT/name).open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));writer.writeheader();writer.writerows(rows)


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        while chunk:=stream.read(1<<20):h.update(chunk)
    return h.hexdigest()


def main():
    final=read("final_validation.csv");unique={}
    for row in final:
        if row.get("selected_plan_file"):
            path=ROOT/row["selected_plan_file"]
            unique.setdefault(row["witness_hash"],{"witness_hash":row["witness_hash"],"file":row["selected_plan_file"],"sha256":sha(path),"bytes":path.stat().st_size,"referenced_by":[]})["referenced_by"].append(f"{row['phase']}/{row['scene_id']}/k{row['k']}/{row['method']}")
    write("accepted_witness_manifest.csv",({**x,"referenced_by":";".join(x["referenced_by"])} for x in unique.values()))
    graph={x["scene_id"]:x for x in json.loads((OUT/"transfer_graph_manifest.json").read_text())["graphs"]}
    timing=[]
    for phase,result_name,event_name in (("DEV","dev_results.csv","dev_validation_events.csv"),("TRANSFER","transfer_results.csv","transfer_validation_events.csv")):
        events=read(event_name)
        for row in read(result_name):
            sid=row["scene_id"]
            timing.append({"phase":phase,"scene_id":sid,"k":row["k"],"method":row["method"],"graph_construction_s":0 if phase=="DEV" else graph[sid]["build_seconds"],"F_search_s":row.get("initializer_seconds"),"method_search_s":row.get("method_seconds"),"core_s":row.get("core_seconds"),"online_screen_s":row.get("screen_seconds"),"validation_calls":sum(x["scene_id"]==sid and x["k"]==row["k"] and x["method"]==row["method"] for x in events),"validation_actual_s":sum(float(x["duration_s"]) for x in events if x["scene_id"]==sid and x["k"]==row["k"] and x["method"]==row["method"]),"cache_semantics":"same witness+k+geometry+model within phase; uncached-equivalent retained in event roles"})
    write("timing_accounting.csv",timing)
    status=json.loads((OUT/"test_statuses.json").read_text());status.update({"focused_passed":65,"literal_passed":219,"literal_skipped":1,"literal_failed":10});(OUT/"test_statuses.json").write_text(json.dumps(status,indent=2,sort_keys=True)+"\n")
    commands=(OUT/"reproduction_commands.txt").read_text().rstrip().splitlines()
    commands=[x for x in commands if "scripts/render_e11_selected_plots.py" not in x]
    render="OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/e11-mpl /data/chocheng/.venvs/coverage-fm/bin/python scripts/render_e11_selected_plots.py --output results/e11_mechanism_placement_transfer_v1_reproduction"
    if render not in commands:commands.insert(-1,render)
    (OUT/"reproduction_commands.txt").write_text("\n".join(commands)+"\n")


if __name__=="__main__":main()
