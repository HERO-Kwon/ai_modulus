import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn as nn
from torch import Tensor
from typing import Dict

from sympy import Symbol, Eq, Abs, sin, cos, Or, And

import modulus
from modulus.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.utils.io import csv_to_dict
from modulus.solver import SequentialSolver
from modulus.domain import Domain


from modulus.models.fully_connected import FullyConnectedArch
from modulus.models.moving_time_window import MovingTimeWindowArch
from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from modulus.domain.constraint import Constraint
from modulus.domain.inferencer import PointVTKInferencer

from modulus.domain.monitor import PointwiseMonitor
from modulus.domain.validator import PointwiseValidator
from modulus.utils.io import (
    VTKUniformGrid,
)
from modulus.key import Key
from modulus.node import Node
from modulus.graph import Graph
from modulus.eq.pde import PDE

from a_params_v4 import *
from a_navier_stokes_vof_2d_v4 import NavierStokes_VOF, Curl
from a_slurry_viscosity_eq_v4 import SlurryViscosity
from a_HC_geo_v5 import *

'''
v0: 개발중
v1: 형상 완료. 0.01 초 간격 구동
v1_1: colloc pnt x5 증가
v1_2: parameter 변경, density, viscosity, vin, uw
v1_3: inferencer 해상도 축소, timestep 0.1, tension변경
v1_4: highres ic, initial filling, add outlet, noslip 범위조정
v1_5: param 1 2 변경
v1_6: interface 추가, 
v1_7: L_ref=Lu, 시간 001
v1_8: no inlet vel
v1_9: initial weighting x100
v1_10: time / l_ref, initial a weight 1
v1_11: initial ic air 1 
v1_12: test_twophase parameters
v1_13: test_vof change
v1_14: test_vof change, ic 0 1 change
v1_15: test_vof serface tension
v2: test_vof initial coating
v2_1: test_vof initial coating interface change
v2_2: air-slurry change
v2_3: no norm
v2_4: norm 0.0005 gradagg 2
v2_5: highres cond change
v2_6: timewindow 0.0001
v2_7: slurry 0 sdf loss
v2_7_1: slurry 물성
v2_8: slurry 물성 001 sec
v3: 물성 변경, hires 영역 변경
v3_1: lowres 영역 변경
v3_2: inlet 부분 lowres 추가

v5: v3_2, all eq
 - gcp: no eq norm
v5_1: v5_gcp + norm
 - gcp: v5_gcp + intecon(x) imp spl
 v5_2: v5_1_gcp + dyn vis(x)
 - gcp: a_v4 params
v5_3: v5_2_gcp + dyn vis
 - gcp: v5_3 + intecon
v5_4: v5_3_gcp + real 0001 sec
 - gcp: norm .002 .1
v5_5: tanh act
v5_6: tanh + norm .002 .01
v5_7: tanh + norm .002 .1
v5_8: silu + norm .002 .01 + bf
v5_9: silu + norm .0002 .1 + ts 001 sec
 -gcp: norm .002 .01
v5_10L v5_9 + stan
 -gcp: 5_10+silu+homoscedastic l
v5_11: v_5_10g + mu norm
v5_12: v_5_10g + no int sdf
 -gcp: norm .002 .01
v5_13: sdf + ts 0.0001
v5_14: norm .0002 1. ts 0.0001 + interior->interface
v5_15: norm .002 .01 ts 0.0001
-gcp: v eq 1.0
v5_16: norm .002 .1
v5_17: norm .0002 1.
- gcp: veq 1.0
v5_18: v eq 1.0
- gcp: free surf
v5_19: penalty a
- gcp: free surf
v5_21: ts 0.0005 tanh free interf
v5_22: ts 0.0001
- gcp: norm .0002 .
v5_23: initial interf(error cond.) a weight x10 ts .0005
v5_24: initial interf a weight x10 ts .0005
- gcp: free surf.
v5_25: silu, initial interf a weight x10 ts .0005, window vis
- gcp: free surf.
v5_26: v2_25 + stan, outleta1, no intecon
- gcp: free surf.
v5_27: v2_26 + silu + setted surf.
- gcp: uw weight 1, free surf(x)
v5_28:  + uw weight 1, density intecon
v5_29:  lr e-5
v5_30:  lr e-4
v5_31:  lr e-2
 - gcp: free surf
v5_32: lr default ,u weight 1 all, step 5000, gradagg 3
 - gcp: free surf
v5_33: 0.0003 sec
 - gcp: free surf
v5_34: 0.0001 sec
- gcp: free surf
v5_35: scale l 0.0002 v 10 no intecon weight u v 10 a 1
- gcp: scale l 0.002 v 10
v5_36: scale l 0.002 v 10 + intecon
- gcp: no intecon

v6: bubble conf + 5_36
-gcp: free surf
v6_2: timestep 20000 scale 0.0002 10
-gcp: vin weight 100 pdev 100
v6_3: gcp+timestep 5k
-gcp: free surf
v6_4: output p weight 100,
-gcp: free surf
v6_5: default v weight, ts 0.00001
-gcp: free surf
v6_6: pressure penalty. ts 0.0001
-gcp: free surf
v6_7: ts 0.001, max step 3000, v norm 10 
-1 : ts 0.0005
-2 : ts 0.0002
v6_8: ts 0.0001
v6_9: outlet change

v7_1: bubble setting doe, inlet pressure penalty(x), L_ref 0.002 network size 512
-gcp: free surf ts 0.001
v7_2: ts 0.0001 , inlet pressure penalty
v7_3: ts 0.00005, 
v7_4: ts 0.0001, inlet pressure 20, v vin
v7_5: v vin(x), pressure penalty
-gcp: free surf

v8_1: pressure doe, inlet pressure 1
-gcp: inlet pressure 10
v8_2: pressure doe l_ref 0.002 ts 0.0005, 3000 step, p 1
-gcp: p0.1
v8_3: l_ref 0.0002, p 20
-gcp: p 2
v8_4: l_ref 0.002, p 0.01
-gcp: p 0.001

-v9_1: 7_2_2 setting, inlet p 20, no intecon inlet, step 3000
-gcp: p 2 1
-v9_2: intecon x100 
-gcp: p 2 1
-v9_3: 25000 step
-gcp: no pressure penalty
-v9_4: from 9_3_gcp 3000 step, intecon inlet p x , w 1000
-v9_5: from 9_3 5k step, outlet a 1, impmes +10, size 256, v boundary x100
-gcp: no pressure penalty
-v9_6: impmes *10, 8k step, outlet a (x)
-gcp: no pressure penalty
-v9_7: impmes *100, pre interior +
-gcp : no pressure penalty
-V9_8: small ts
-1: many pnt, timestep 0.00001
-gcp : no pressure penalty

v10_1: impmes a 0.5, inlet vel x1000, inlet intercon
v10_2: initial pressure, prev diff p --> ts 0.001
- gcp: free surf, minimal pnt
v10_3: lref 0.002, ts0.0005, initial vin
- gcp: psdiff ini weight uv 1
v10_4: lref 0.0002, ts 0.0001, init vin, p prev diff, interior sdf, lr 1e-3, ini pnt 1000
- home: lr lre-4, diff uvpa, inlet p removed
- gcp: default

v11_1: conf_v3
v11_2: no sdf pressure, added monitor
v11_3: 10_4 setting conf v11
-gcp: conf v10_1
v11_4: reduced conf point interface focus, 
-gcp: Lref 0.002
v11_5: 10_1 conf interface focus
-gcp: Lref 0.002
v11_6: 10_4 setting

v12_1: running cond.
v12_2: Lref 0.002
v12_2_1: p inlet x
v12_2_2_gcp: monitor 100000
v12_2_3: lref 0.0002, p inlet x,
v12_2_4: no intecon
v12_2_5: default setting
v12_2_6: a p(x) output act. 
v12_2_7: tanh
v12_2_8: p output act
v12_2_8_gcp: free surf
''' 


class AlphaConverter(nn.Module):
    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {"alpha": torch.heaviside(in_vars["a"]-0.5,torch.tensor([0.0], device=in_vars["a"].device))}
class OutputA(nn.Module):
    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {"a": torch.sigmoid(in_vars["net_a"])}
class OutputP(nn.Module):
    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {"p": torch.exponential(in_vars["net_p"])}
class MuCalc(nn.Module):
    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {"mu": (in_vars["mu2"] + (mu1 - in_vars["mu2"]) * in_vars["a"] )}

class NormalDotVec(PDE):
    name = "NormalDotVec"
    def __init__(self, vec=["u", "v"]):
        # normal
        normal = [Symbol("normal_x"), Symbol("normal_y")]
        a = Symbol("a")
        # make input variables
        self.equations = {}
        self.equations["normal_dot_vel"] = 0
        for v, n in zip(vec, normal):
            self.equations["normal_dot_vel"] += Abs(1-a)*Symbol(v) * n


@modulus.main(config_path="conf", config_name="config_coating_v12_1")
def run(cfg: ModulusConfig) -> None:

    # make navier stokes equations
    slurry_viscosity = SlurryViscosity(dim=2, time=True)
    ns = NavierStokes_VOF(mu1=mu1,mu2=slurry_viscosity.equations["mu2"], rhos=(rho1,rho2), sigma=sigma, g=g, U_ref=U_ref, L_ref=L_ref, dim=2, time=True)
    normal_dot_vel = NormalDotVec(["u", "v"])

    # define sympy variables to parametrize domain curves
    x, y = Symbol("x"), Symbol("y")
    
    # make network for current step and previous step
    flow_net = FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[Key("u"), Key("v"), Key("net_p"),Key("net_a")],
        layer_size=256,
    )
    time_window_net = MovingTimeWindowArch(flow_net, time_window_size)

    # make nodes to unroll graph on
    nodes = (ns.make_nodes() + slurry_viscosity.make_nodes()
             + normal_dot_vel.make_nodes() 
             + [Node(['a'], ['alpha'], AlphaConverter())] 
             + [Node(['net_a'], ['a'], OutputA())] 
             + [Node(['net_p'], ['p'], OutputP())] 
             + [Node(['mu2','a'], ['mu'], MuCalc())] 
             + [time_window_net.make_node(name="time_window_network")])    
    # make importance model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    importance_model_graph = Graph(
        nodes,
        invar=[Key("x"), Key("y"), Key("t")],
        req_names=[Key("a")],
    ).to(device)

    def importance_measure(invar):
        outvar = importance_model_graph(
            Constraint._set_device(invar, device=device, requires_grad=False)
        )
        importance = 10* 2*(0.5-torch.abs(0.5-outvar["a"]))
        return importance.cpu().detach().numpy()
    
    # make initial condition domain
    ic_domain = Domain("initial_conditions")

    # make moving window domain
    window_domain = Domain("window")

    # make initial condition
    ic_air = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo_uncoating,
        outvar={
            "u": 0,
            "v": 0,
            "p": 0,
            "a": 1,
        },
        batch_size=cfg.batch_size.initial_condition,
        lambda_weighting={"u": Symbol("sdf"), "v": Symbol("sdf"), "p": Symbol("sdf"), "a": 100},
        #criteria=Or((x < 0.0), (x > Lf),(y<H0)),
        parameterization={t_symbol: 0},
    )
    ic_domain.add_constraint(ic_air, name="ic_air")

    ic_slurry_up = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo_coating,
        outvar={
            "u": 0,
            "v": -1*v_in,
            "p": 0.6662*y/H0+7.5513,
            "a": 0,
        },
        batch_size=cfg.batch_size.initial_condition,
        lambda_weighting={"u": Symbol("sdf"), "v": Symbol("sdf"), "p": Symbol("sdf"), "a": 100},
        criteria=And((x > 0.0), (x < Lf), (y>H0)),
        parameterization={t_symbol: 0},
    )
    ic_domain.add_constraint(ic_slurry_up, name="ic_slurry_up")       

    ic_slurry_down = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo_coating,
        outvar={
            "u": 0,
            "v": 0,
            "p": 4.0162*y*y/H0/H0-4.4586*y/H0+8.6604,
            "a": 0,
        },
        batch_size=cfg.batch_size.initial_condition,
        lambda_weighting={"u": Symbol("sdf"), "v": Symbol("sdf"), "p": Symbol("sdf"), "a": 100},
        criteria=And((x > 0.0), (x < Lf), (y<H0)),
        parameterization={t_symbol: 0},
    )
    ic_domain.add_constraint(ic_slurry_down, name="ic_slurry_down")       

    ic_slurry_right = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo_coating,
        outvar={
            "u": Uw,
            "v": 0,
            "p": 0,
            "a": 0,
        },
        batch_size=cfg.batch_size.initial_condition,
        lambda_weighting={"u": Symbol("sdf"), "v": Symbol("sdf"), "p": Symbol("sdf"), "a": 100},
        criteria=(x>Lf),
        parameterization={t_symbol: 0},
    )
    ic_domain.add_constraint(ic_slurry_right, name="ic_slurry_right")    

    # make constraint for matching previous windows initial condition
    ic_highres = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"p_prev_step_diff": 0, "a_prev_step_diff": 0},
        batch_size=cfg.batch_size.highres_interior,
        lambda_weighting={
            #"u_prev_step_diff": 100,
            #"v_prev_step_diff": 100,
            "p_prev_step_diff": 100,
            "a_prev_step_diff": 100,
        },
        parameterization={t_symbol: 0},
    )
    window_domain.add_constraint(ic_highres, name="ic_highres")

    # boundary condition
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0, "v": 0,},
        batch_size=cfg.batch_size.no_slip,
        criteria=And((y>0.0), Or(And((-1*left_tri_height-1*Lu<x),(x<=0.0)),And((Lf<=x),(x<Lf+Ld+right_tri_height)))),
        parameterization=time_range,
    )
    ic_domain.add_constraint(no_slip, "no_slip")
    window_domain.add_constraint(no_slip, "no_slip")

    # inlet boundary
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0, "v": -1*v_in, "a": 0},# "p": 18.45},
        batch_size=cfg.batch_size.inlet,
        lambda_weighting={"u": 1.0, "v": 1.0, "a":1.0},# "p":1.0},
        criteria=And((x>0),(x<Lf),(y>0)),
        parameterization=time_range,
    )
    ic_domain.add_constraint(inlet, "inlet")
    window_domain.add_constraint(inlet, "inlet")

    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"p": 0},
        batch_size=cfg.batch_size.outlet,
        criteria=And((y>0.0), Or((-1*left_tri_height-1*Lu>x),(x>Lf+Ld+right_tri_height))),
        parameterization=time_range,
    )
    ic_domain.add_constraint(outlet, "outlet")
    window_domain.add_constraint(outlet, "outlet")
    

    # moving plate
    plate = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": Uw ,"v":0},
        batch_size=cfg.batch_size.no_slip,
        lambda_weighting={"u": 1.0, "v": 1.0},
        criteria=Eq(y,0.0),
        parameterization=time_range,
    )
    ic_domain.add_constraint(plate, name="plate")
    window_domain.add_constraint(plate, name="plate")
    
    
    # make interior constraint
    lowres_interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"PDE_m": 0, "PDE_a": 0, "PDE_u": 0, "PDE_v": 0},
        batch_size=cfg.batch_size.lowres_interior,
        lambda_weighting={
            "PDE_m": Symbol("sdf"),#Symbol("sdf"),
            "PDE_a": 1, #Symbol("sdf"),
            "PDE_u": 10*Symbol("sdf"),
            "PDE_v": 10*Symbol("sdf"),
        },
        criteria=Or((x<-1*Lu), And(Or((x>(Lf+right_rx))),(y>H0))),
        parameterization=time_range,
    )
    ic_domain.add_constraint(lowres_interior, name="lowres_interior")
    window_domain.add_constraint(lowres_interior, name="lowres_interior")

    highres_interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"PDE_m": 0, "PDE_a": 0, "PDE_u": 0, "PDE_v": 0},
        batch_size=cfg.batch_size.highres_interior,
        lambda_weighting={
            "PDE_m": Symbol("sdf"),#Symbol("sdf"),
            "PDE_a": 1,#Symbol("sdf"),
            "PDE_u": 10*Symbol("sdf"),#*Symbol(sdf"),
            "PDE_v": 10*Symbol("sdf"),#*Symbol("sdf"),
        },
        criteria=Or(And((x>-1*Lu),(x<(Lf+right_width)),(y<H0)),And((x>Lf+Ld),(x<(Lf+right_rx)),(y>H0))),
        parameterization=time_range,
    )
    ic_domain.add_constraint(highres_interior, name="highres_interior")
    window_domain.add_constraint(highres_interior, name="highres_interior")
    
    inlet_interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"PDE_m": 0, "PDE_a": 0, "PDE_u": 0, "PDE_v": 0},# "penalty_p":0},
        batch_size=cfg.batch_size.lowres_interior,
        lambda_weighting={
            "PDE_m": Symbol("sdf"),
            "PDE_a": 1,#Symbol("sdf"),
            "PDE_u": 10*Symbol("sdf"),
            "PDE_v": 10*Symbol("sdf"),
        },
        criteria=And((x>0),(x<(Lf)),(y>H0)),
        parameterization=time_range,
    )
    ic_domain.add_constraint(inlet_interior, name="inlet_interior")
    window_domain.add_constraint(inlet_interior, name="inlet_interior")


    interface_impmes = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"PDE_m": 0, "PDE_a": 0, "PDE_u": 0, "PDE_v": 0},
        batch_size=cfg.batch_size.interface,
        lambda_weighting={"PDE_m": 1.0, "PDE_a": 1.0, "PDE_u": 10*Symbol("sdf"), "PDE_v": 10*Symbol("sdf")},
        importance_measure=importance_measure,
        parameterization=time_range,
    )
    ic_domain.add_constraint(interface_impmes, name="interface_impmes")
    window_domain.add_constraint(interface_impmes, name="interface_impmes")
    
    interface_left = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo_coating,
        outvar={"PDE_m": 0, "PDE_a": 0, "PDE_u": 0, "PDE_v": 0},
        batch_size=cfg.batch_size.interface_left,
        lambda_weighting={"PDE_m": 1.0, "PDE_a": 1.0, "PDE_u": 10.0, "PDE_v": 10.0},
        criteria=And((x < 0.0), (y > 0.0), (y<H0)),
        parameterization=time_range,
    )
    ic_domain.add_constraint(interface_left, name="interface_left")
    window_domain.add_constraint(interface_left, name="interface_left")
    
    interface_right = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo_coating,
        outvar={"PDE_m": 0, "PDE_a": 0, "PDE_u": 0, "PDE_v": 0},
        #bounds=box_bounds,
        batch_size=cfg.batch_size.interface_right,
        lambda_weighting={"PDE_m": 1.0, "PDE_a": 1.0,   "PDE_u": 10.0, "PDE_v": 10.0},
        criteria=And((x>(Lf+Ld)),(x<(Lf+right_width)),(y > 0.0)),
        parameterization=time_range,
    )
    ic_domain.add_constraint(interface_right, name="interface_right")
    window_domain.add_constraint(interface_right, name="interface_right")
    
    integral_continuity = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=integral_line,
        outvar={"normal_dot_vel": v_in*Lf},
        batch_size=3,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={"normal_dot_vel": 100.0},
        parameterization={Symbol("t"): (0, time_window_size), Parameter("x_pos"): (right_rx,(Lf+right_width))}
    )
    ic_domain.add_constraint(integral_continuity, "integral_continuity")
    window_domain.add_constraint(integral_continuity, "integral_continuity")
    
    integral_continuity_in = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=mid_rec,
        outvar={"normal_dot_vel": v_in*Lf},
        batch_size=1,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={"normal_dot_vel": 100.0},
        criteria=Eq(y,H0),
        parameterization={Symbol("t"): (0, time_window_size)}
    )
    ic_domain.add_constraint(integral_continuity_in, "integral_continuity_in")
    window_domain.add_constraint(integral_continuity_in, "integral_continuity_in")
    
    # monitors for force, residuals and temperature
    global_monitor = PointwiseMonitor(
        geo.sample_interior(500, criteria=(y<H0)),
        output_names=["PDE_m","PDE_u","PDE_v","alpha"],
        metrics={
            "slurry_volume": lambda var: torch.sum(
                var["area"] * torch.abs(1-var["alpha"])
            ),
            "mass_imbalance": lambda var: torch.sum(
                var["area"] * torch.abs(var["PDE_m"])
            ),
            "momentum_imbalance": lambda var: torch.sum(
                var["area"]
                * (torch.abs(var["PDE_u"]) + torch.abs(var["PDE_v"]))
            ),
        },
        nodes=nodes,
        requires_grad=True,
    )
    ic_domain.add_monitor(global_monitor)
    window_domain.add_monitor(global_monitor)

    # add monitor
    p_monitor = PointwiseMonitor(
        geo.sample_boundary(100, criteria=Eq(y, H0+mid_height)),
        output_names=["p"],
        metrics={
            "inlet_p": lambda var: torch.mean(var["p"]),
        },
        nodes=nodes,
    )
    ic_domain.add_monitor(p_monitor)
    window_domain.add_monitor(p_monitor)

    v_monitor = PointwiseMonitor(
        geo.sample_boundary(100, criteria=Eq(y, H0+mid_height)),
        output_names=["v"],
        metrics={
            "inlet_v": lambda var: torch.mean(var["v"]),
        },
        nodes=nodes,
    )
    ic_domain.add_monitor(v_monitor)
    window_domain.add_monitor(v_monitor)

    # add inference data for time slices
    def mask_fn(x, y):
        sdf = geo.sdf({"x": x, "y": y}, {})
        return sdf["sdf"] < 0
    
    vtk_obj = VTKUniformGrid(
        bounds=[(-1*left_width, (Lf+right_width)), (0.0, right_height)],
        npoints=[128, 128],
        export_map={"u": ["u"],"v": ["v"], "p": ["p"], "a": ["a"], "alpha": ["alpha"], "mu":["mu"]},
    )
    grid_inference = PointVTKInferencer(
        vtk_obj=vtk_obj,
        nodes=nodes,
        input_vtk_map={"x": "x", "y": "y"},
        output_names=["u", "v", "p", "a","alpha","mu"],
        requires_grad=True,
        mask_fn=mask_fn,
        invar={"t": np.full([128 ** 2, 1], 0)},
        batch_size=100000,
    )
    ic_domain.add_inferencer(grid_inference, name="time_slice_" + str(t_symbol).zfill(4))
    window_domain.add_inferencer(
        grid_inference, name="time_slice_" + str(t_symbol).zfill(4)
    )

    
    vtk_obj1 = VTKUniformGrid(
        bounds=[(-1*left_width/5, (Lf+right_width/5)), (0.0, right_height/5)],
        npoints=[128, 128],
        export_map={"u": ["u"],"v": ["v"], "p": ["p"], "a": ["a"], "alpha":["alpha"], "mu":["mu"]},
    )
    grid_inference1 = PointVTKInferencer(
        vtk_obj=vtk_obj1,
        nodes=nodes,
        input_vtk_map={"x": "x", "y": "y"},
        output_names=["u", "v", "p", "a","alpha","mu"],
        requires_grad=True,
        mask_fn=mask_fn,
        invar={"t": np.full([128 ** 2, 1], 0)},
        batch_size=100000,
    )
    ic_domain.add_inferencer(grid_inference1, name="window_" + str(t_symbol).zfill(4))
    window_domain.add_inferencer(
        grid_inference1, name="window_" + str(t_symbol).zfill(4)
    )
    
    # make solver
    slv = SequentialSolver(
        cfg,
        [(1, ic_domain), (nr_time_windows, window_domain)],
        custom_update_operation=time_window_net.move_window,
    )

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
