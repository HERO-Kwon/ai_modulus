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
#from modulus.eq.pdes.basic import NormalDotVec

from modulus.models.fully_connected import FullyConnectedArch
from modulus.models.moving_time_window import MovingTimeWindowArch
from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from modulus.domain.constraint import Constraint
from modulus.domain.inferencer import PointVTKInferencer
from modulus.utils.io import (
    VTKUniformGrid,
)
from modulus.key import Key
from modulus.node import Node
from modulus.graph import Graph
from modulus.eq.pde import PDE

from a_params_v4 import *
from a_navier_stokes_vof_2d_v4 import NavierStokes_VOF
from a_slurry_viscosity_eq_v4 import SlurryViscosity
from a_HC_geo_v4 import *


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
v4_1: dynanic viscosity, initial x
v4_2: intecon (x)
v4_3: initial 조건 최적화, viscosity scale, time scale, importance sampling(x)
v4_4: initial 조건 취소
v4_5: bfgs optimizer , no scale
v4_6: no slip 부분 오류 수정, params
 -gcp: integral cont.
v4_7: timestep 001, mu2 3.5
 -gcp: no ini coating layer
v4_8: timestep 1, inlet hires
v4_9: mu2 inference
v4_10: timestep 001
v4_11: timestep 01, (sl_vis,mu) a 
'''

class AlphaConverter(nn.Module):
    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {"alpha": torch.heaviside(in_vars["a"]-0.5,torch.tensor([0.0], device=in_vars["a"].device))}


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



@modulus.main(config_path="conf", config_name="config_coating_v4")
def run(cfg: ModulusConfig) -> None:

    # time window parameters
    time_window_size = 0.1 #/ t_ref
    t_symbol = Symbol("t")
    time_range = {t_symbol: (0, time_window_size)}
    nr_time_windows = 200

    # make navier stokes equations
    slurry_viscosity = SlurryViscosity(dim=2, time=True)
    ns = NavierStokes_VOF(mu1=mu1,mu2=slurry_viscosity.equations["mu2"], rhos=(rho1,rho2), sigma=sigma, g=g, U_ref=U_ref, L_ref=L_ref, dim=2, time=True)
    #ns = NavierStokes_VOF(mu1=mu1,mu2=mu2, rhos=(rho1,rho2), sigma=sigma, g=g, U_ref=U_ref, L_ref=L_ref, dim=2, time=True)
    normal_dot_vel = NormalDotVec(["u", "v"])
    # define sympy variables to parametrize domain curves
    x, y = Symbol("x"), Symbol("y")

    # make network for current step and previous step
    flow_net = FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[Key("u"), Key("v"), Key("p"), Key("a")],
        layer_size=256,
    )
    time_window_net = MovingTimeWindowArch(flow_net, time_window_size)

    # make nodes to unroll graph on
    nodes = (ns.make_nodes() + slurry_viscosity.make_nodes() 
             + normal_dot_vel.make_nodes()
             + [Node(['a'], ['alpha'], AlphaConverter())] 
             + [time_window_net.make_node(name="time_window_network")])
    
    # make importance model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    importance_model_graph = Graph(
        nodes,
        invar=[Key("x"), Key("y"), Key("t")],
        req_names=[
            Key("a", derivatives=[Key("x")]),
            Key("a", derivatives=[Key("y")]),
        ],
    ).to(device)

    def importance_measure(invar):
        outvar = importance_model_graph(
            Constraint._set_device(invar, device=device, requires_grad=True)
        )
        importance = (
            outvar["a__x"] ** 2
            + outvar["a__y"] ** 2
        ) ** 0.5# + 10
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
        lambda_weighting={"u": 100, "v": 100, "p": 100, "a": 100},
        #criteria=Or((x < 0.0), (x > Lf)),
        parameterization={t_symbol: 0},
    )
    ic_domain.add_constraint(ic_air, name="ic_air")

    ic_slurry = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo_coating,
        outvar={
            "u": 0,
            "v": 0,
            "p": 0,
            "a": 0,
        },
        batch_size=cfg.batch_size.initial_condition,
        lambda_weighting={"u": 100, "v": 100, "p": 100, "a": 100},
        #criteria=And((x > 0.0), (x < Lf)),
        parameterization={t_symbol: 0},
    )
    ic_domain.add_constraint(ic_slurry, name="ic_slurry")    

    # make constraint for matching previous windows initial condition
    ic_highres = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u_prev_step_diff": 0, "v_prev_step_diff": 0, "a_prev_step_diff": 0},
        batch_size=cfg.batch_size.highres_interior,
        lambda_weighting={
            "u_prev_step_diff": 100,
            "v_prev_step_diff": 100,
            "a_prev_step_diff": 100,
        },
        criteria=And((x>-1*Lu),(x<(Lf+right_width)),(y<H0)),
        parameterization={t_symbol: 0},
    )
    window_domain.add_constraint(ic_highres, name="ic_highres")

    ic_lowres = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u_prev_step_diff": 0, "v_prev_step_diff": 0, "a_prev_step_diff": 0},
        batch_size=cfg.batch_size.lowres_interior,
        lambda_weighting={
            "u_prev_step_diff": 100,
            "v_prev_step_diff": 100,
            "a_prev_step_diff": 100,
        },
        criteria=Or((x<-1*Lu), And(Or(And((x>0),(x<Lf)),(x>(Lf+right_rx))),(y>H0))),
        parameterization={t_symbol: 0},
    )
    window_domain.add_constraint(ic_lowres, name="ic_lowres")

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
        outvar={"u": 0, "v": -1*v_in, "a": 0},
        batch_size=cfg.batch_size.inlet,
        lambda_weighting={"u": 10.0, "v": 10.0, "a":10.0},
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
        lambda_weighting={"u": 10.0, "v": 10.0},
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
        #bounds=box_bounds,
        batch_size=cfg.batch_size.lowres_interior,
        #lambda_weighting={"PDE_m": 1.0, "PDE_a": 1.0,   "PDE_u": 10.0, "PDE_v": 10.0},
        lambda_weighting={
            "PDE_m": Symbol("sdf"),
            "PDE_a": Symbol("sdf"),
            "PDE_u": 10*Symbol("sdf"),
            "PDE_v": 10*Symbol("sdf"),
        },
        criteria=Or((x<-1*Lu), And(Or(And((x>0),(x<Lf)),(x>(Lf+right_rx))),(y>H0))),
        parameterization=time_range,
    )
    ic_domain.add_constraint(lowres_interior, name="lowres_interior")
    window_domain.add_constraint(lowres_interior, name="lowres_interior")
    
    highres_interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"PDE_m": 0, "PDE_a": 0, "PDE_u": 0, "PDE_v": 0},
        #bounds=box_bounds,
        batch_size=cfg.batch_size.highres_interior,
        #lambda_weighting={"PDE_m": 1.0, "PDE_a": 1.0,   "PDE_u": 10.0, "PDE_v": 10.0},
        lambda_weighting={
            "PDE_m": Symbol("sdf"),
            "PDE_a": Symbol("sdf"),
            "PDE_u": 10*Symbol("sdf"),
            "PDE_v": 10*Symbol("sdf"),
        },
        criteria=And((x>-1*Lu),(x<(Lf+right_width)),(y<H0)),
        importance_measure=importance_measure,
        parameterization=time_range,
    )
    ic_domain.add_constraint(highres_interior, name="highres_interior")
    window_domain.add_constraint(highres_interior, name="highres_interior")
    
    highres_interior1 = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"PDE_m": 0, "PDE_a": 0, "PDE_u": 0, "PDE_v": 0},
        #bounds=box_bounds,
        batch_size=cfg.batch_size.highres_interior1,
        #lambda_weighting={"PDE_m": 1.0, "PDE_a": 1.0,   "PDE_u": 10.0, "PDE_v": 10.0},
        lambda_weighting={
            "PDE_m": Symbol("sdf"),
            "PDE_a": Symbol("sdf"),
            "PDE_u": 10*Symbol("sdf"),
            "PDE_v": 10*Symbol("sdf"),
        },
        criteria=And((x>Lf+Ld),(x<(Lf+right_rx)),(y>H0)),
        importance_measure=importance_measure,
        parameterization=time_range,
    )
    ic_domain.add_constraint(highres_interior1, name="highres_interior1")
    window_domain.add_constraint(highres_interior1, name="highres_interior1")

    interface_left = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo_coating,
        outvar={"PDE_m": 0, "PDE_a": 0, "PDE_u": 0, "PDE_v": 0},
        #bounds=box_bounds,
        batch_size=cfg.batch_size.interface_left,
        lambda_weighting={"PDE_m": 1.0, "PDE_a": 1.0,   "PDE_u": 10.0, "PDE_v": 10.0},
        criteria=And((x < 0.0), (y > 0.0), (y<H0)),
        importance_measure=importance_measure,
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
        importance_measure=importance_measure,
        parameterization=time_range,
    )
    ic_domain.add_constraint(interface_right, name="interface_right")
    window_domain.add_constraint(interface_right, name="interface_right")
    
    
    # integral continuity
    integral_continuity = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=integral_line,
        outvar={"normal_dot_vel": v_in*Lf},
        batch_size=3,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={"normal_dot_vel": 1.0},
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
        lambda_weighting={"normal_dot_vel": 1.0},
        criteria=Eq(y,H0),
        parameterization={Symbol("t"): (0, time_window_size)}
    )
    ic_domain.add_constraint(integral_continuity_in, "integral_continuity_in")
    window_domain.add_constraint(integral_continuity_in, "integral_continuity_in")
    

    # add inference data for time slices
    #for i, specific_time in enumerate(np.linspace(0, time_window_size, 10)):
    def mask_fn(x, y):
        sdf = geo.sdf({"x": x, "y": y}, {})
        return sdf["sdf"] < 0
    
    vtk_obj = VTKUniformGrid(
        bounds=[(-1*left_width, (Lf+right_width)), (0.0, right_height)],
        npoints=[128, 128],
        export_map={"u": ["u"],"v": ["v"], "p": ["p"], "a": ["a"], "alpha":["alpha"], "mu2":["mu2"]},
    )
    grid_inference = PointVTKInferencer(
        vtk_obj=vtk_obj,
        nodes=nodes,
        input_vtk_map={"x": "x", "y": "y"},
        output_names=["u", "v", "p", "a","alpha","mu2"],
        #requires_grad=False,
        requires_grad=True,
        mask_fn=mask_fn,
        invar={"t": np.full([128 ** 2, 1], 0)},
        batch_size=100000,
    )
    ic_domain.add_inferencer(grid_inference, name="time_slice_" + str(t_symbol).zfill(4))
    window_domain.add_inferencer(
        grid_inference, name="time_slice_" + str(t_symbol).zfill(4)
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
