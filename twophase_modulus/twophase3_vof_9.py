import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

from sympy import Symbol, Eq, Abs, sin, cos, Or, And

import modulus
from modulus.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.utils.io import csv_to_dict
from modulus.solver import SequentialSolver
from modulus.domain import Domain

#from modulus.geometry.primitives_3d import Box
from modulus.geometry.primitives_2d import Rectangle, Circle

from modulus.models.fully_connected import FullyConnectedArch
from modulus.models.fourier_net import FourierNetArch
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


'''
Desc : twophase 3_1
- NSE 식 : pow, sqrt 

Desc : twophase 3_2
- boundary 해석
- 해석 pnt 90% 축소
- timestep 0.1-29

Desc: twophase 3-3
- Twop3-3: boundary moving 추가
,"v":0.3/L_ref
Desc: twophase 3-4
- 3-3 + adaptive ac., no normalize
Desc: twophase 3-5
- 3-3 + adaptive ac. , 12sec
Desc: twophase 3-8
- running
3-8-1 conf 변경
3-8-2 fouriernet
3-9 코팅 물성 적용
'''

@modulus.main(config_path="conf", config_name="config_twophase3_2")
def run(cfg: ModulusConfig) -> None:

    # time window parameters
    time_window_size = 0.01
    t_symbol = Symbol("t")
    time_range = {t_symbol: (0, time_window_size)}
    nr_time_windows = 599
    '''
    # parameters
    rho1 = 100 # density
    rho2 = 1000
    mu1 = 1 #viscosity
    mu2 = 10
    sigma=24.5 #surface_tension_coeff 
    g = -0.98 # gravitational acceleration
    U_ref = 1.0
    L_ref = 0.25
    '''    
    # parameters
    rho2 = 1000 # density
    rho1 = 1.1614
    mu2 = 1*rho2 #viscosity
    mu1 = rho1*1.84e-05  # kg/m-s
    sigma=0.06 #surface_tension_coeff 
    g = -0.98 # gravitational acceleration
    U_ref = 1.0
    L_ref = 0.25

    # make navier stokes equations
    ns = NavierStokes_VOF(mus=(mu1,mu2), rhos=(rho1,rho2), sigma=sigma, g=g, U_ref=U_ref, L_ref=L_ref, dim=2, time=True)

    # define sympy variables to parametrize domain curves
    x, y = Symbol("x"), Symbol("y")

    # make geometry for problem
    channel_width = (-0.5/L_ref, 0.5/L_ref)
    channel_length = (-0.5/L_ref, 1.5/L_ref)
    #box_bounds = {x: channel_width, y: channel_length}

    highres_width = (-0.5/L_ref, 0.5/L_ref)
    highres_length = (-0.25/L_ref, 0.75/L_ref)

    bubble_origin = (0.0, 0.0)
    bubble_radius = 0.25/L_ref


    # define geometry
    rec = Rectangle(
        (channel_width[0], channel_length[0]),
        (channel_width[1], channel_length[1])
    )
    '''
    highres_rec = Rectangle(
        (highres_width[0], highres_length[0]),
        (highres_width[1], highres_length[1])
    )
    '''
    bubble = Circle(bubble_origin, bubble_radius) & (rec)
    #lowres_rec = rec-highres_rec
    fluid2 = rec - bubble

    # make network for current step and previous step
    flow_net = FourierNetArch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[Key("u"), Key("v"), Key("p"), Key("a")],
        periodicity={"x": channel_width},
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
    
    ic = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=fluid2,
        outvar={
            "u": 0,
            "v": 0,
            "p": 0,
            "a": 0,
        },
        batch_size=cfg.batch_size.initial_condition,
        lambda_weighting={"u": 100, "v": 100, "p": 100, "a": 100},
        parameterization={t_symbol: 0},
    )
    ic_domain.add_constraint(ic, name="ic")
    
    ic_b = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=bubble,
        outvar={
            "u": 0,
            "v": 0,
            "p": 0,
            "a": 1,
        },
        batch_size=cfg.batch_size.initial_condition,
        lambda_weighting={"u": 100, "v": 100, "p": 100, "a": 100},
        parameterization={t_symbol: 0},
    )
    ic_domain.add_constraint(ic_b, name="ic_b")
    # make constraint for matching previous windows initial condition
    '''
    # symmetry channel
    symmetry = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"u": 0, "v__x": 0, "p__x": 0},
        batch_size=cfg.batch_size.symmetry,
        criteria=Eq(x, 0.0),
    )
    window_domain.add_constraint(symmetry, "symmetry")
    '''
    ic_lowres = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"u_prev_step_diff": 0, "v_prev_step_diff": 0, "a_prev_step_diff": 0},
        batch_size=cfg.batch_size.lowres_interior,
        lambda_weighting={
            "u_prev_step_diff": 100,
            "v_prev_step_diff": 100,
            "a_prev_step_diff": 100,
        },
        criteria=Or((y < highres_length[0]), y > (highres_length[1])),
        parameterization={t_symbol: 0},
    )
    window_domain.add_constraint(ic_lowres, name="ic_lowres")
    
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
    
    # boundary condition
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"u": 0, "v": 0,},
        batch_size=cfg.batch_size.no_slip,
        criteria=(y < channel_length[1]),
        parameterization=time_range,
    )
    ic_domain.add_constraint(no_slip, "no_slip")
    window_domain.add_constraint(no_slip, "no_slip")

    # pressure boundary
    bd_pressure = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"u": 0, "v": 0, "p": 1},
        batch_size=cfg.batch_size.no_slip,
        criteria=Eq(y, channel_length[1]),
        parameterization=time_range,
    )
    ic_domain.add_constraint(bd_pressure, "bd_pressure")
    window_domain.add_constraint(bd_pressure, "bd_pressure")
    
    
    # make interior constraint
    lowres_interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"PDE_m": 0, "PDE_a": 0, "PDE_u": 0, "PDE_v": 0},
        #bounds=box_bounds,
        batch_size=cfg.batch_size.lowres_interior,
        lambda_weighting={"PDE_m": 1.0, "PDE_a": 1.0,   "PDE_u": 10.0, "PDE_v": 10.0},
        criteria=Or((y < highres_length[0]), y > (highres_length[1])),
        parameterization=time_range,
    )
    ic_domain.add_constraint(lowres_interior, name="lowres_interior")
    window_domain.add_constraint(lowres_interior, name="lowres_interior")
    
    highres_interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"PDE_m": 0, "PDE_a": 0, "PDE_u": 0, "PDE_v": 0},
        #bounds=box_bounds,
        batch_size=cfg.batch_size.highres_interior,
        lambda_weighting={"PDE_m": 1.0, "PDE_a": 1.0,   "PDE_u": 10.0, "PDE_v": 10.0},
        criteria=And((y >= highres_length[0]), y <= (highres_length[1])),
        parameterization=time_range,
    )
    ic_domain.add_constraint(highres_interior, name="highres_interior")
    window_domain.add_constraint(highres_interior, name="highres_interior")
    
    interface = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=bubble,
        outvar={"PDE_m": 0, "PDE_a": 0, "PDE_u": 0, "PDE_v": 0,"v":0.1937},
        #bounds=box_bounds,
        batch_size=cfg.batch_size.interface,
        lambda_weighting={"PDE_m": 1.0, "PDE_a": 1.0,   "PDE_u": 10.0, "PDE_v": 10.0,"v":10.0},
        parameterization=time_range,
    )
    ic_domain.add_constraint(interface, name="interface")
    window_domain.add_constraint(interface, name="interface")
    
    
    # add inference data for time slices
    #for i, specific_time in enumerate(np.linspace(0, time_window_size, 10)):
    vtk_obj = VTKUniformGrid(
        bounds=[(-0.5/L_ref, 0.5/L_ref), (-0.5/L_ref, 1.5/L_ref)],
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
