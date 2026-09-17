from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

from diffusion_coverage.models.e12_generator import END, PAD, VIA, scan_symbol, sequence_key
from diffusion_coverage.planning.e12_programs import GeometryLibrary, GeometryProgram


FIXED_TASK_SCALARS = np.asarray([0.14, 0.008, 0.02, 0.10], dtype=np.float64)


def condition_vector(transform: np.ndarray, q0: np.ndarray) -> np.ndarray:
    transform=np.asarray(transform,dtype=np.float64);q0=np.asarray(q0,dtype=np.float64)
    if transform.shape!=(4,4) or q0.shape[0]<6:
        raise ValueError("invalid task condition")
    return np.concatenate((transform[:3,3],transform[:3,:3].reshape(-1),q0[:5],FIXED_TASK_SCALARS))


def program_arrays(program: GeometryProgram, library: GeometryLibrary) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    symbols=[];controls=[];scales=[]
    family_index={name:index for index,name in enumerate(library.families)}
    for token in program.tokens:
        if token.kind=="SCAN":
            symbols.append(scan_symbol(family_index[str(token.family)],int(token.direction)))
            controls.append((float(token.a),float(token.b)))
            points=library.family_points[str(token.family)]/library.radius
            length=float(library.radius*np.sum(np.arctan2(np.linalg.norm(np.cross(points[:-1],points[1:]),axis=1),np.sum(points[:-1]*points[1:],axis=1))))
            scales.append((0.008/max(length,1e-12),)*2)
        elif token.kind=="VIA":
            symbols.append(VIA);controls.append((float(token.d1),float(token.d2)));scales.append((0.008/library.radius,)*2)
        elif token.kind=="END":
            symbols.append(END);controls.append((0.,0.));scales.append((1.,1.))
        else:
            raise ValueError(f"unsupported token {token.kind}")
    return np.asarray(symbols,np.int64),np.asarray(controls,np.float64),np.asarray(scales,np.float64)


def symbol_scales(symbols: np.ndarray, library: GeometryLibrary) -> np.ndarray:
    scales=np.ones((len(symbols),2),dtype=np.float64)
    for index,symbol in enumerate(symbols):
        value=int(symbol)
        if 3<=value<=8:
            family=(value-3)//2;points=library.family_points[library.families[family]]/library.radius
            length=float(library.radius*np.sum(np.arctan2(np.linalg.norm(np.cross(points[:-1],points[1:]),axis=1),np.sum(points[:-1]*points[1:],axis=1))))
            scales[index]=0.008/max(length,1e-12)
        elif value==VIA:scales[index]=0.008/library.radius
    return scales


@dataclass(frozen=True)
class ProgramDataset:
    records: tuple[dict[str,Any],...]
    condition_mean: np.ndarray
    condition_std: np.ndarray
    prototypes: dict[str,np.ndarray]
    maximum_tokens: int

    @classmethod
    def from_labels(cls, labels: list[dict[str,Any]], library: GeometryLibrary, *, maximum_tokens: int=64) -> "ProgramDataset":
        records=[]
        for label in labels:
            program=GeometryProgram.from_json(label["program_json"]);symbols,controls,scales=program_arrays(program,library)
            if len(symbols)>maximum_tokens:raise ValueError("teacher exceeds maximum token count")
            records.append({**label,"symbols":symbols,"controls":controls,"scales":scales,"condition":condition_vector(np.asarray(label["transform"]),np.asarray(label["q0"]))})
        if not records:raise ValueError("no qualified labels")
        train=[x for x in records if x["split"]=="TRAIN"]
        if not train:raise ValueError("no TRAIN labels")
        conditions=np.stack([x["condition"] for x in train]);mean=conditions.mean(0);std=conditions.std(0);std=np.where(std<1e-8,1.,std)
        grouped:dict[str,list[np.ndarray]]={}
        for item in train:grouped.setdefault(sequence_key(item["symbols"]),[]).append(item["controls"])
        prototypes={key:np.stack(values).mean(0) for key,values in grouped.items()}
        return cls(tuple(records),mean,std,prototypes,maximum_tokens)

    def prototype_for(self, symbols: np.ndarray) -> np.ndarray:
        key=sequence_key(symbols)
        if key in self.prototypes:return self.prototypes[key].copy()
        controls=np.zeros((len(symbols),2),dtype=np.float64)
        for i,symbol in enumerate(symbols):
            if 3<=int(symbol)<=8:controls[i]=[0.,1.]
        return controls

    def batch(self, indices: np.ndarray) -> dict[str,np.ndarray]:
        length=self.maximum_tokens;batch=len(indices);symbols=np.full((batch,length),PAD,np.int64);controls=np.zeros((batch,length,2),np.float32);residuals=np.zeros_like(controls);mask=np.zeros((batch,length),bool);conditions=[]
        for row,index in enumerate(indices):
            item=self.records[int(index)];n=len(item["symbols"]);symbols[row,:n]=item["symbols"];controls[row,:n]=item["controls"];prototype=self.prototype_for(item["symbols"]);residuals[row,:n]=(item["controls"]-prototype)/item["scales"];mask[row,:n]=np.isin(item["symbols"],[VIA,3,4,5,6,7,8]);conditions.append((item["condition"]-self.condition_mean)/self.condition_std)
        return {"symbols":symbols,"controls":controls,"residuals":residuals,"continuous_mask":mask,"condition":np.asarray(conditions,np.float32)}

    def metadata(self) -> dict[str,Any]:
        proto={key:value.tolist() for key,value in self.prototypes.items()}
        payload={"condition_mean":self.condition_mean.tolist(),"condition_std":self.condition_std.tolist(),"prototypes":proto,"maximum_tokens":self.maximum_tokens}
        payload["sha256"]=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest();return payload
