import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

from sympy import Symbol, Eq, Abs, sin, cos

import modulus
from modulus.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.utils.io import csv_to_dict
from modulus.solver import SequentialSolver
from modulus.domain import Domain

from modulus.geometry.primitives_2d import Rectangle, Circle


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
from modulus.eq.pdes.navier_stokes import NavierStokes


@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:

    # time window parameters
    time_window_size = 1.0
    t_symbol = Symbol("t")
    time_range = {t_symbol: (0, time_window_size)}
    nr_time_windows = 10

    # equation parameters
    rho_1 = 100 # densities
    rho_2 = 1000
    nu_1 = 1 # viscosities
    nu_2 = 10
    
    # make navier stokes equations
    ns = NavierStokes(nu=nu_2, rho=rho_2, dim=2, time=True)

    # define sympy variables to parametrize domain curves
    x, y = Symbol("x"), Symbol("y")

    # make geometry for problem
    channel_width = (-0.5, 0.5)
    channel_height = (-1.0, 1.0)
    t_range = (0.0,3.0)
    
    bubble_origin = (0.0,0.0)
    bubble_radius = 0.25
        
    sigma=24.5 #surface_tension_coeff 
    g = -0.98 # gravitational acceleration
    
    
    # define geometry
    rec = Rectangle(
        (channel_width[0], channel_height[0]),
        (channel_width[1], channel_height[1])
    )
    bubble = Circle(bubble_origin,bubble_radius)
    
    fluid_1 = bubble
    fluid_2 = rec-bubble

    # make network for current step and previous step
    flow_net = FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        periodicity={"x": channel_width, "y": channel_height},
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
        geometry=fluid_2,
        outvar={
            "u": 0,
            "v": 0,
        },
        batch_size=5000,
        lambda_weighting={"u": 100, "v": 100,},
        parameterization={t_symbol: 0},
    )
    ic_domain.add_constraint(ic, name="ic")

    # make constraint for matching previous windows initial condition
    ic = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=fluid_2,
        outvar={"u_prev_step_diff": 0, "v_prev_step_diff": 0},
        batch_size=5000,
        lambda_weighting={
            "u_prev_step_diff": 100,
            "v_prev_step_diff": 100,
        },
        parameterization={t_symbol: 0},
    )
    window_domain.add_constraint(ic, name="ic")
    
    # no slip
    '''
    no_slip_north = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=fluid_2,
        outvar={"u": 0, "v": 0, "p": 1},
        batch_size=cfg.batch_size.no_slip,
    )
    domain.add_constraint(no_slip, "no_slip")
    '''
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=fluid_2,
        outvar={"u": 0, "v": 0},
        batch_size=5000,
    )
    ic_domain.add_constraint(no_slip, "no_slip")
    window_domain.add_constraint(no_slip, "no_slip")
    
    
    # make interior constraint
    interior_2 = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=fluid_2,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        batch_size=4094,
        parameterization=time_range,
    )
    ic_domain.add_constraint(interior_2, name="interior_2")
    window_domain.add_constraint(interior_2, name="interior_2")

    # add inference data for time slices
    for i, specific_time in enumerate(np.linspace(0, time_window_size, 10)):
        vtk_obj = VTKUniformGrid(
            bounds=[(-0.5, 0.5), (-1.0, 1.0)],
            npoints=[128, 128],
            export_map={"u": ["u", "v"], "p": ["p"]},
        )
        grid_inference = PointVTKInferencer(
            vtk_obj=vtk_obj,
            nodes=nodes,
            input_vtk_map={"x": "x", "y": "y"},
            output_names=["u", "v", "p"],
            requires_grad=False,
            invar={"t": np.full([128 ** 2, 1], specific_time)},
            batch_size=100000,
        )
        ic_domain.add_inferencer(grid_inference, name="time_slice_" + str(i).zfill(4))
        window_domain.add_inferencer(
            grid_inference, name="time_slice_" + str(i).zfill(4)
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
