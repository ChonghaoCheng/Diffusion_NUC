#!/usr/bin/env python3
from __future__ import annotations

import argparse,csv,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from diffusion_coverage.learning.e12_program_dataset import ProgramDataset
from diffusion_coverage.models.e12_generator import BOS, PAD, ContinuousProgramHead, GeneratorConfig, ProgramDecoder, parameter_count
from diffusion_coverage.planning.e12_programs import load_geometry_library


def atomic_json(path:Path,value):
    temporary=path.with_suffix(path.suffix+".tmp");temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n");temporary.replace(path)


def sample_indices(dataset:ProgramDataset,rng:np.random.Generator,batch:int,split:str="TRAIN"):
    tasks=sorted({x["scene_id"] for x in dataset.records if x["split"]==split});by_task={task:[i for i,x in enumerate(dataset.records) if x["split"]==split and x["scene_id"]==task] for task in tasks}
    chosen=[]
    for _ in range(batch):
        task=tasks[int(rng.integers(len(tasks)))];options=by_task[task];chosen.append(options[int(rng.integers(len(options)))])
    return np.asarray(chosen,np.int64)


def tensors(dataset,indices,device):
    values=dataset.batch(indices);return {key:torch.as_tensor(value,device=device) for key,value in values.items()}


def masked_mse(prediction,target,mask):
    weights=mask.unsqueeze(-1);return ((prediction-target).square()*weights).sum()/weights.sum().clamp_min(1)


MODEL_ORDER=("categorical","reg","fm")


def _gpu_inventory():
    try:return subprocess.run(["nvidia-smi","--query-gpu=name,uuid,memory.total,driver_version","--format=csv,noheader"],capture_output=True,text=True,check=True).stdout.strip().splitlines()
    except Exception as exc:return [f"nvidia-smi unavailable: {exc}"]


def finalize(out:Path,labels:list[dict],models:dict[str,nn.Module]|None=None):
    model_dir=out/"models";missing=[name for name in MODEL_ORDER if not (model_dir/f"{name}_final.pt").exists()]
    if missing:raise RuntimeError(f"cannot finalize missing checkpoints: {missing}")
    logs=[];workers={}
    for name in MODEL_ORDER:
        path=out/f"training_curve_{name}.csv"
        with path.open(newline="") as stream:logs.extend(csv.DictReader(stream))
        workers[name]=json.loads((out/f"training_worker_{name}.json").read_text())
    with (out/"training_curves.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(logs[0]));writer.writeheader();writer.writerows(logs)
    counts={name:(parameter_count(models[name]) if models is not None else int(workers[name]["parameter_count"])) for name in MODEL_ORDER}
    manifest={"status":"complete","parallel_model_training":True,"root_training_seed":0,"model_stream_seeds":{name:int(workers[name]["model_stream_seed"]) for name in MODEL_ORDER},"torch":torch.__version__,"cuda":torch.version.cuda,"gpu_inventory":_gpu_inventory(),"workers":workers,"elapsed_s_sum":sum(float(x["elapsed_s"]) for x in workers.values()),"elapsed_s_critical_path":max(float(x["elapsed_s"]) for x in workers.values()),"qualified_labels":len(labels),"qualified_train_tasks":len({x['scene_id'] for x in labels if x['split']=='TRAIN'}),"qualified_validation_tasks":len({x['scene_id'] for x in labels if x['split']=='VALIDATION'}),"parameter_counts":counts,"checkpoints":{name:{"path":str((model_dir/f'{name}_final.pt').relative_to(ROOT)),"sha256":hashlib.sha256((model_dir/f'{name}_final.pt').read_bytes()).hexdigest()} for name in MODEL_ORDER}}
    atomic_json(out/"training_manifest.json",manifest);atomic_json(out/"train.checkpoint.json",{"complete":True,"manifest":"training_manifest.json"});print(json.dumps(manifest,indent=2))


def train(args):
    config=json.loads((ROOT/"configs/e12_graph_free_global_generation_v1.json").read_text());out=args.output.resolve();model_dir=out/"models";model_dir.mkdir(parents=True,exist_ok=True)
    labels=json.loads((out/"qualified_program_dataset.json").read_text())["labels"]
    library=load_geometry_library(out/"geometry_only_library.npz",out/"geometry_only_library.json",radius=float(config["surface"]["radius_m"]));dataset=ProgramDataset.from_labels(labels,library,maximum_tokens=int(config["program"]["maximum_tokens"]));metadata=dataset.metadata();metadata_path=out/"dataset_normalization_and_prototypes.json"
    if args.prepare_only:
        atomic_json(metadata_path,metadata);print(json.dumps({"status":"prepared","labels":len(labels),"metadata_sha256":metadata["sha256"]},indent=2));return
    if not metadata_path.exists() or json.loads(metadata_path.read_text())!=metadata:raise RuntimeError("training metadata was not prepared or does not match the frozen corpus")
    if args.finalize:
        finalize(out,labels);return
    learning=config["learning"];torch.manual_seed(int(learning["seed"]));torch.cuda.manual_seed_all(int(learning["seed"]));np.random.seed(int(learning["seed"]));device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type!="cuda":raise RuntimeError("E12 registered GPU training but CUDA is unavailable")
    gcfg=GeneratorConfig(condition_dim=len(dataset.condition_mean),width=int(learning["width"]),layers=int(learning["layers"]),heads=int(learning["heads"]),ffn=int(learning["ffn"]),dropout=float(learning["dropout"]),maximum_tokens=int(config["program"]["maximum_tokens"]))
    decoder=ProgramDecoder(gcfg).to(device);reg=ContinuousProgramHead(gcfg).to(device);fm=ContinuousProgramHead(gcfg).to(device)
    models={"categorical":decoder,"reg":reg,"fm":fm};requested=tuple(x.strip() for x in args.models.split(",") if x.strip())
    if not requested or any(x not in MODEL_ORDER for x in requested):raise ValueError(f"invalid --models {args.models!r}")
    for name in requested:
        model=models[name];model_index=MODEL_ORDER.index(name);stream_seed=int(np.random.SeedSequence([int(learning["seed"]),model_index]).generate_state(1,dtype=np.uint32)[0]);rng=np.random.default_rng(stream_seed);torch.manual_seed(stream_seed);torch.cuda.manual_seed_all(stream_seed);logs=[]
        optimizer=torch.optim.AdamW(model.parameters(),lr=float(learning["learning_rate"]),weight_decay=float(learning["weight_decay"]));began=time.perf_counter()
        for update in range(1,int(learning["updates"])+1):
            batch=tensors(dataset,sample_indices(dataset,rng,int(learning["batch_size"])),device);symbols=batch["symbols"].long();condition=batch["condition"].float()
            if name=="categorical":
                inputs=torch.full_like(symbols,PAD);inputs[:,0]=BOS;inputs[:,1:]=symbols[:,:-1];logits=model(inputs,condition);loss=nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),symbols.reshape(-1),ignore_index=PAD)
            elif name=="reg":
                zero=torch.zeros_like(batch["residuals"]);prediction=model(symbols,condition,zero,torch.zeros(symbols.shape[0],device=device));loss=masked_mse(prediction,batch["residuals"].float(),batch["continuous_mask"])
            else:
                target=batch["residuals"].float();noise=torch.randn_like(target);t=torch.rand(symbols.shape[0],device=device);mixed=(1-t[:,None,None])*noise+t[:,None,None]*target;prediction=model(symbols,condition,mixed,t);loss=masked_mse(prediction,target-noise,batch["continuous_mask"])
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),float(learning["gradient_clip"]));optimizer.step()
            if update==1 or update%100==0 or update==int(learning["updates"]):
                validation=[i for i,x in enumerate(dataset.records) if x["split"]=="VALIDATION"]
                validation_loss=None
                if validation:
                    with torch.no_grad():
                        vb=tensors(dataset,np.asarray(validation,np.int64),device);vs=vb["symbols"].long();vc=vb["condition"].float()
                        if name=="categorical":
                            vi=torch.full_like(vs,PAD);vi[:,0]=BOS;vi[:,1:]=vs[:,:-1];validation_loss=float(nn.functional.cross_entropy(model(vi,vc).reshape(-1,gcfg.vocabulary_size),vs.reshape(-1),ignore_index=PAD).cpu())
                        elif name=="reg":validation_loss=float(masked_mse(model(vs,vc,torch.zeros_like(vb["residuals"]),torch.zeros(vs.shape[0],device=device)),vb["residuals"].float(),vb["continuous_mask"]).cpu())
                        else:
                            # Fixed validation noise/time isolates training changes.
                            generator=torch.Generator(device=device).manual_seed(20260917);target=vb["residuals"].float();noise=torch.randn(target.shape,generator=generator,device=device);t=torch.full((vs.shape[0],),.5,device=device);validation_loss=float(masked_mse(model(vs,vc,.5*(noise+target),t),target-noise,vb["continuous_mask"]).cpu())
                logs.append({"model":name,"update":update,"train_loss":float(loss.detach().cpu()),"validation_loss":validation_loss,"elapsed_s":time.perf_counter()-began})
        elapsed=time.perf_counter()-began;torch.save({"state_dict":model.state_dict(),"generator_config":gcfg.__dict__,"dataset_metadata":dataset.metadata(),"training_config":learning,"root_training_seed":int(learning["seed"]),"model_stream_seed":stream_seed,"final_update":int(learning["updates"])},model_dir/f"{name}_final.pt")
        with (out/f"training_curve_{name}.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(logs[0]));writer.writeheader();writer.writerows(logs)
        atomic_json(out/f"training_worker_{name}.json",{"status":"complete","model":name,"device":str(device),"visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"gpu_name":torch.cuda.get_device_name(device),"root_training_seed":int(learning["seed"]),"model_stream_seed":stream_seed,"elapsed_s":elapsed,"parameter_count":parameter_count(model),"checkpoint_sha256":hashlib.sha256((model_dir/f'{name}_final.pt').read_bytes()).hexdigest()})
    if set(requested)==set(MODEL_ORDER):finalize(out,labels,models)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=ROOT/"results/e12_graph_free_global_generation_v1");parser.add_argument("--models",default=",".join(MODEL_ORDER));parser.add_argument("--prepare-only",action="store_true");parser.add_argument("--finalize",action="store_true");train(parser.parse_args())
