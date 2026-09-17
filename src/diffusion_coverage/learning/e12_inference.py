from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diffusion_coverage.learning.e12_program_dataset import condition_vector, symbol_scales
from diffusion_coverage.models.e12_generator import END, PAD, VIA, ContinuousProgramHead, GeneratorConfig, ProgramDecoder, heun_generate, sequence_key, unpack_scan_symbol
from diffusion_coverage.planning.e12_programs import GeometryLibrary, GeometryProgram, ProgramToken


def load_models(model_directory: Path, device: torch.device):
    loaded={}
    for name,kind in (("categorical",ProgramDecoder),("reg",ContinuousProgramHead),("fm",ContinuousProgramHead)):
        checkpoint=torch.load(model_directory/f"{name}_final.pt",map_location=device,weights_only=False);config=GeneratorConfig(**checkpoint["generator_config"]);model=kind(config).to(device);model.load_state_dict(checkpoint["state_dict"]);model.eval();loaded[name]=(model,checkpoint)
    return loaded


def normalized_condition(transform,q0,metadata,device):
    raw=condition_vector(np.asarray(transform),np.asarray(q0));value=(raw-np.asarray(metadata["condition_mean"]))/np.asarray(metadata["condition_std"]);return torch.as_tensor(value,dtype=torch.float32,device=device).unsqueeze(0)


def generate_symbol_slots(decoder:ProgramDecoder,condition:torch.Tensor,*,seed:int)->list[np.ndarray]:
    slots=[]
    for slot in range(8):
        generator=torch.Generator(device=condition.device).manual_seed(seed+slot)
        tokens=decoder.generate(condition,generator=generator,temperature=None if slot==0 else 1.0)[0].cpu().numpy().astype(np.int64)
        slots.append(tokens)
    return slots


def prototype_for_symbols(symbols:np.ndarray,metadata:dict[str,Any])->np.ndarray:
    key=sequence_key(symbols)
    if key in metadata["prototypes"]:return np.asarray(metadata["prototypes"][key],dtype=np.float64)
    result=np.zeros((len(symbols),2),dtype=np.float64)
    result[(symbols>=3)&(symbols<=8)]=[0.,1.]
    return result


def controls_to_program(symbols:np.ndarray,controls:np.ndarray,library:GeometryLibrary,maximum_tokens:int)->tuple[GeometryProgram|None,list[dict[str,Any]],str|None]:
    tokens=[];corrections=[];ended=False
    if len(symbols)>maximum_tokens:return None,corrections,"overlength"
    for index,(symbol,control) in enumerate(zip(symbols,controls,strict=True)):
        value=int(symbol)
        if value==PAD:continue
        if value==END:
            tokens.append(ProgramToken("END"));ended=True;break
        if 3<=value<=8:
            family,direction=unpack_scan_symbol(value);raw=np.asarray(control,dtype=np.float64);clipped=np.clip(raw,0.,1.);a,b=sorted(clipped.tolist())
            if np.max(np.abs(raw-clipped))>0:corrections.append({"token":index,"kind":"scan_clamp","magnitude":float(np.max(np.abs(raw-clipped)))})
            if b-a<=1e-12:return None,corrections,"zero_span_SCAN"
            tokens.append(ProgramToken("SCAN",library.families[family],direction,a,b))
        elif value==VIA:
            raw=np.asarray(control,dtype=np.float64);norm=float(np.linalg.norm(raw));value2=raw/max(1.,norm)
            if norm>1:corrections.append({"token":index,"kind":"via_radial_projection","magnitude":float(np.linalg.norm(raw-value2))})
            tokens.append(ProgramToken("VIA",d1=float(value2[0]),d2=float(value2[1])))
        else:return None,corrections,"invalid_symbol"
    if not ended:return None,corrections,"END_not_generated"
    if not any(x.kind=="SCAN" for x in tokens):return None,corrections,"no_SCAN"
    return GeometryProgram(tuple(tokens)),corrections,None


@torch.no_grad()
def continuous_program(method:str,model:ContinuousProgramHead,symbols:np.ndarray,condition:torch.Tensor,metadata:dict[str,Any],library:GeometryLibrary,*,seed:int,maximum_tokens:int,fm_steps:int=32):
    usable=[]
    for symbol in symbols:
        usable.append(int(symbol))
        if int(symbol)==END:break
    symbols=np.asarray(usable,np.int64);tokens=torch.as_tensor(symbols,device=condition.device).unsqueeze(0);prototype=prototype_for_symbols(symbols,metadata);scales=symbol_scales(symbols,library)
    if method=="REG":residual=model(tokens,condition,torch.zeros((1,len(symbols),2),device=condition.device),torch.zeros(1,device=condition.device))[0].cpu().numpy();nfe=1
    elif method=="FM":
        generator=torch.Generator(device=condition.device).manual_seed(seed);noise=torch.randn((1,len(symbols),2),generator=generator,device=condition.device);sample,nfe=heun_generate(model,tokens,condition,noise,steps=fm_steps);residual=sample[0].cpu().numpy()
    else:raise ValueError(method)
    controls=prototype+residual*scales;program,corrections,error=controls_to_program(symbols,controls,library,maximum_tokens);return program,{"symbolic_key":sequence_key(symbols),"prototype_known":sequence_key(symbols) in metadata["prototypes"],"residual_norm":float(np.linalg.norm(residual)),"nfe":nfe,"corrections":corrections,"error":error,"controls":controls.tolist()}
