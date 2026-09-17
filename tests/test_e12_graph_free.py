from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from diffusion_coverage.planning.e09_geometry import shortest_sphere_arc
from diffusion_coverage.planning.e12_programs import (
    GeometryProgram, ProgramToken, decode_program, encode_edge_sequence,
    load_geometry_library, sphere_to_stereographic, stereographic_to_sphere,
    validate_program,
)
from diffusion_coverage.robot.e12_program_execution import lift_program
from diffusion_coverage.models.e12_generator import (
    END, VIA, ContinuousProgramHead, GeneratorConfig, ProgramDecoder,
    heun_generate, scan_symbol, sequence_key, unpack_scan_symbol,
)
from diffusion_coverage.learning.e12_inference import controls_to_program


ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def library():
    return load_geometry_library(
        ROOT/"results/e11_mechanism_placement_transfer_v1/geometry_bank.npz",
        ROOT/"results/e11_mechanism_placement_transfer_v1/geometry_bank.json",radius=.14,
    )


def test_geometry_identity(library):
    assert library.semantic_hash=="01050368f57eca9a904a04106dc705aec89cab65248c508462ad232328976047"
    assert len(library.ports)==238


def test_forward_source_encodes_scan(library):
    p=encode_edge_sequence([{"kind":"source","geom_arc_id":0,"start_port":0,"end_port":1}],library)
    assert [x.kind for x in p.tokens]==["SCAN","END"] and p.tokens[0].direction==1


def test_reverse_source_preserves_direction(library):
    p=encode_edge_sequence([{"kind":"source","geom_arc_id":1,"start_port":1,"end_port":0}],library)
    assert p.tokens[0].direction==-1
    d=decode_program(p,library,library.ports[1],maximum_step=.001)
    assert np.linalg.norm(d.surface_points[-1]-library.ports[0])<1e-12


def test_consecutive_sources_merge(library):
    p=encode_edge_sequence([{"kind":"source","geom_arc_id":0},{"kind":"source","geom_arc_id":2}],library)
    assert sum(x.kind=="SCAN" for x in p.tokens)==1


def test_cross_port_becomes_via(library):
    arc=next(i for i,k in enumerate(library.arc_kind) if k=="cross_port")
    p=encode_edge_sequence([{"kind":"source","geom_arc_id":0},{"kind":"cross_port","geom_arc_id":arc,"end_port":int(library.arc_end[arc])},{"kind":"source","geom_arc_id":2}],library)
    assert any(x.kind=="VIA" for x in p.tokens)


@pytest.mark.parametrize("point",[np.array([0.,0.,.14]),np.array([.14,0.,0.]),np.array([0.,.14,0.])])
def test_stereographic_pole_and_equator(point):
    d=sphere_to_stereographic(point,.14);restored,_=stereographic_to_sphere(*d,.14)
    assert np.linalg.norm(restored-point)<1e-14


def test_via_projection_logged():
    point,correction=stereographic_to_sphere(2.,0.,.14)
    assert correction and correction["magnitude"]==pytest.approx(1.)
    assert point[2]==pytest.approx(0.,abs=1e-14)


def test_identity_connector_is_not_episode_error(library):
    p=GeometryProgram((ProgramToken("SCAN","raster_u_phase_0.00",1,0.,.01),ProgramToken("END")))
    start=library.family_points["raster_u_phase_0.00"][0]
    d=decode_program(p,library,start,maximum_step=.001)
    assert np.linalg.norm(d.surface_points[0]-start)<1e-15
    assert np.max(np.linalg.norm(d.surface_points-start,axis=1))>0


def test_antipodal_rejected():
    with pytest.raises(ValueError,match="antipodal"):
        shortest_sphere_arc(np.array([.14,0,0]),np.array([-.14,0,0]),.14,.001)


@pytest.mark.parametrize("program",[
    GeometryProgram((ProgramToken("END"),)),
    GeometryProgram((ProgramToken("SCAN","x",1,0.,1.),ProgramToken("END"))),
    GeometryProgram((ProgramToken("SCAN","raster_u_phase_0.00",1,.2,.2),ProgramToken("END"))),
    GeometryProgram((ProgramToken("SCAN","raster_u_phase_0.00",1,0.,1.),)),
])
def test_malformed_programs_rejected(program,library):
    with pytest.raises(ValueError):validate_program(program,library.families)


def test_overlength_rejected(library):
    p=GeometryProgram(tuple([ProgramToken("VIA",d1=0.,d2=0.)]*64+[ProgramToken("SCAN",library.families[0],1,0.,1.),ProgramToken("END")]))
    with pytest.raises(ValueError,match="cap"):validate_program(p,library.families,64)


def test_off_witness_outside_primary_scope(library):
    with pytest.raises(ValueError,match="OFF"):
        encode_edge_sequence([{"kind":"off_reconfiguration"}],library)


def test_program_json_roundtrip(library):
    p=GeometryProgram((ProgramToken("SCAN",library.families[0],-1,.1,.9),ProgramToken("END")))
    assert GeometryProgram.from_json(p.to_json())==p
    assert len(p.content_hash)==64


def test_lifter_module_has_no_graph_loader_or_teacher_q_dependency():
    source=inspect.getsource(lift_program)
    assert "load_robot_graph" not in source
    assert "teacher" not in source.lower()
    assert list(inspect.signature(lift_program).parameters)[:5]==["program","library","transform_base_from_surface","q0","robot"]


def test_e12_generator_symbols_and_shapes():
    torch=pytest.importorskip("torch")
    for family in range(3):
        for direction in (-1,1):
            assert unpack_scan_symbol(scan_symbol(family,direction))==(family,direction)
    config=GeneratorConfig(width=16,layers=1,heads=2,ffn=32,maximum_tokens=8)
    decoder=ProgramDecoder(config);head=ContinuousProgramHead(config)
    condition=torch.zeros(2,config.condition_dim)
    tokens=torch.tensor([[scan_symbol(0,1),VIA,END],[scan_symbol(2,-1),END,0]])
    decoder_input=torch.cat((torch.ones(2,1,dtype=torch.long),tokens[:,:-1]),1)
    assert decoder(decoder_input,condition).shape==(2,3,10)
    controls=torch.zeros(2,3,2)
    assert head(tokens,condition,controls,torch.zeros(2)).shape==controls.shape
    sample,nfe=heun_generate(head,tokens,condition,controls,steps=3)
    assert sample.shape==controls.shape and nfe==6
    assert sequence_key([0,scan_symbol(0,1),END,VIA])==f"{scan_symbol(0,1)},{END}"


def test_generated_controls_are_decoded_without_oracle_structure(library):
    symbols=np.asarray([scan_symbol(1,-1),VIA,END])
    program,corrections,error=controls_to_program(symbols,np.asarray([[1.2,-.1],[2.,0.],[0.,0.]]),library,64)
    assert error is None and program is not None
    assert program.tokens[0].direction==-1 and program.tokens[0].a==0. and program.tokens[0].b==1.
    assert {x["kind"] for x in corrections}=={"scan_clamp","via_radial_projection"}


def test_missing_end_is_rejected(library):
    program,_,error=controls_to_program(np.asarray([scan_symbol(0,1)]),np.asarray([[0.,1.]]),library,64)
    assert program is None and error=="END_not_generated"
