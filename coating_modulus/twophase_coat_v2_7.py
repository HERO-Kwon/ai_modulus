import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

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
)
from modulus.domain.inferencer import PointVTKInferencer
from modulus.utils.io import (
    VTKUniformGrid,
)
from modulus.key import Key
from modulus.node import Node
from navier_stokes_vof_2d import NavierStokes_VOF
from HC_geo_v2_3 import *


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
'''

@modulus.main(config_path="conf", config_name="config_coating_v2_7")
def run(cfg: ModulusConfig) -> None:

    # time window parameters
    time_window_size = 0.001#/L_ref
    t_symbol = Symbol("t")
    time_range = {t_symbol: (0, time_window_size)}
    nr_time_windows = 200

    # parameters
    '''
    rho1 = 1000 # density
    rho2 = 1.1614
    mu1 = 1*rho1 #viscosity
    mu2 = rho2*1.84e-05  # kg/m-s
    sigma=0.06 #surface_tension_coeff 
    g = -0.98 # gravitational acceleration
    U_ref = 1.0
    '''
    # parameters
    rho1 = 100 # density
    rho2 = 1000
    mu1 = 1 #viscosity
    mu2 = 10
    sigma=24.5 #surface_tension_coeff 
    g = -0.98 # gravitational acceleration
    U_ref = 1.0
    
    # make navier stokes equations
    ns = NavierStokes_VOF(mus=(mu1,mu2), rhos=(rho1,rho2), sigma=sigma, g=g, U_ref=U_ref, L_ref=L_ref, dim=2, time=True)

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
    nodes = ns.make_nodes() + [time_window_net.make_node(name="time_window_network")]

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
        #criteria=And((x >= 0.0), (x <= Lf)),
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
        parameterization={t_symbol: 0},
    )
    window_domain.add_constraint(ic_highres, name="ic_lowres")
    '''
    ic_highres = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"u_prev_step_diff": 0, "v_prev_step_diff": 0, "a_prev_step_diff": 0},
        batch_size=cfg.batch_size.highres_interior,
        lambda_weighting={
            "u_prev_step_diff": 100,
            "v_prev_step_diff": 100,
            "a_prev_step_diff": 100,
        },
        criteria=And((y >= highres_length[0]), y <= (highres_length[1])),
        parameterization={t_symbol: 0},
    )
    window_domain.add_constraint(ic_highres, name="ic_highres")
    '''
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
        outvar={"u": 0, "v": -0.016667, "a": 0},
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
        outvar={"u": 0.067 ,"v":0},
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
        criteria=Or((x < -1*Lu), (x > Lf+Ld+right_tri_height),(y>right_ry)),
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
        criteria=And((x >= -1*Lu), (x <= Lf+Ld+right_tri_height),(y<=right_ry)),
        parameterization=time_range,
    )
    ic_domain.add_constraint(highres_interior, name="highres_interior")
    window_domain.add_constraint(highres_interior, name="highres_interior")
    
    interface_left = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo_coating,
        outvar={"PDE_m": 0, "PDE_a": 0, "PDE_u": 0, "PDE_v": 0},
        #bounds=box_bounds,
        batch_size=cfg.batch_size.interface_left,
        lambda_weighting={"PDE_m": 1.0, "PDE_a": 1.0,   "PDE_u": 10.0, "PDE_v": 10.0},
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
        criteria=And((x > (Lf+Ld)), (y > 0.0),(y<(Lf+right_width))),
        parameterization=time_range,
    )
    ic_domain.add_constraint(interface_right, name="interface_right")
    window_domain.add_constraint(interface_right, name="interface_right")
    
    # add inference data for time slices
    #for i, specific_time in enumerate(np.linspace(0, time_window_size, 10)):
    vtk_obj = VTKUniformGrid(
        bounds=[(-0.005/L_ref, (Lf+0.01/L_ref)), (0.0, 0.004/L_ref)],
        npoints=[128, 128],
        export_map={"u": ["u"],"v": ["v"], "p": ["p"], "a": ["a"]},
    )
    grid_inference = PointVTKInferencer(
        vtk_obj=vtk_obj,
        nodes=nodes,
        input_vtk_map={"x": "x", "y": "y"},
        output_names=["u", "v", "p", "a"],
        requires_grad=False,
        invar={"t": np.full([128 ** 2, 1], 0)},
        batch_size=100000,
    )
    ic_domain.add_inferencer(grid_inference, name="time_slice_" + str(0).zfill(4))
    window_domain.add_inferencer(
        grid_inference, name="time_slice_" + str(0).zfill(4)
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
