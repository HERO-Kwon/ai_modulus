from modulus.geometry.primitives_2d import Rectangle, Triangle, Circle, Line
from modulus.geometry import Parameterization, Parameter
from modulus.geometry.parameterization import OrderedParameterization

from a_params_v4_gcp import *

from sympy import Symbol, Eq, Abs, sin, cos, Or, And
'''
v2:코팅 표면 추가
v4: params
v5: paretrization for monitor
'''
# time window parameters
time_window_size = 0.0001 / t_ref
t_symbol = Symbol("t")
time_range = {t_symbol: (0, time_window_size)}
nr_time_windows = 200

# geometry parameters
Lf = 0.0005 / L_ref
Lu = 0.001 / L_ref
Ld = 0.001 / L_ref
H0 = 0.0002 / L_ref
mid_height=0.003 / L_ref
right_height=0.004 / L_ref
left_height=0.0025 / L_ref
left_width=0.005 / L_ref
right_width=0.01 / L_ref

v_in = 0.016667 / U_ref #m/s
Uw = 4/60 / U_ref #m/s
hw = v_in*Lf/Uw

# circle_doe0
left_r = 0.000273  / L_ref
left_rx = -0.000048 / L_ref
left_ry = 0.000250 / L_ref
right_r = 0.000583 / L_ref
right_rx = 0.001874 / L_ref
right_ry = 0.000736 / L_ref

# make geometry for problem
mid_rec_width = (0.0, Lf)
mid_rec_height = (H0, H0+mid_height)

bottom_left_rec_width = (-1*left_width, 0.0)
bottom_mid_rec_width = (0.0, Lf)
bottom_right_rec_width = (Lf, Lf+right_width)
bottom_rec_height = (0.0, H0)
coating_thick_rec_height = (0.0, hw)

left_tri_height = left_height-H0
left_tri_center = (-1*Lu-left_tri_height, H0)

left_rec_width = (-1*left_width, -1*Lu-left_tri_height)
left_rec_height = (H0, left_height)

right_tri_height = right_height-H0
right_tri_center = (Lf+Ld+right_tri_height,H0)

right_rec_width = (Lf+Ld+right_tri_height, Lf+right_width)
right_rec_height = (H0, right_height)


# define geometry
mid_rec = Rectangle(
    (mid_rec_width[0],mid_rec_height[0]),
    (mid_rec_width[1],mid_rec_height[1]),
    parameterization=OrderedParameterization(time_range, key=t_symbol)
)
bottom_left_rec = Rectangle(
    (bottom_left_rec_width[0],bottom_rec_height[0]),
    (bottom_left_rec_width[1],bottom_rec_height[1]),
    parameterization=OrderedParameterization(time_range, key=t_symbol)
)
bottom_mid_rec = Rectangle(
    (bottom_mid_rec_width[0],bottom_rec_height[0]),
    (bottom_mid_rec_width[1],bottom_rec_height[1]),
    parameterization=OrderedParameterization(time_range, key=t_symbol)
)
bottom_right_rec = Rectangle(
    (bottom_right_rec_width[0],bottom_rec_height[0]),
    (bottom_right_rec_width[1],bottom_rec_height[1]),
    parameterization=OrderedParameterization(time_range, key=t_symbol)
)
left_rec = Rectangle(
    (left_rec_width[0],left_rec_height[0]),
    (left_rec_width[1],left_rec_height[1]),
    parameterization=OrderedParameterization(time_range, key=t_symbol)
)
left_triangle = Triangle(
    left_tri_center, #center
    2*left_tri_height, left_tri_height, #base, height,
    parameterization=OrderedParameterization(time_range, key=t_symbol)
)
right_rec = Rectangle(
    (right_rec_width[0],right_rec_height[0]),
    (right_rec_width[1],right_rec_height[1]),
    parameterization=OrderedParameterization(time_range, key=t_symbol)
)
right_triangle = Triangle(
    right_tri_center,
    2*right_tri_height,right_tri_height,
    parameterization=OrderedParameterization(time_range, key=t_symbol)
)

#geo = mid_rec + bottom_left_rec+ bottom_mid_rec+ bottom_right_rec + left_rec + left_triangle + right_rec + right_triangle

left_circle = Circle(
    (left_rx,left_ry),
    left_r,
    parameterization=OrderedParameterization(time_range, key=t_symbol)
)
right_circle = Circle(
    (right_rx,right_ry),
    right_r,
    parameterization=OrderedParameterization(time_range, key=t_symbol)
)
coating_thick_rec = Rectangle(
    (bottom_right_rec_width[0],coating_thick_rec_height[0]),
    (bottom_right_rec_width[1],coating_thick_rec_height[1]),
    parameterization=OrderedParameterization(time_range, key=t_symbol)
)
right_window_rec = Rectangle(
    (Lf,0.0),
    (right_rx,right_ry),
    parameterization=OrderedParameterization(time_range, key=t_symbol)
)

bottom_left = left_triangle + bottom_left_rec + left_rec
bottom_left_coating = bottom_left & left_circle
bottom_left_uncoating = bottom_left - left_circle

bottom_right = right_triangle + bottom_right_rec + right_rec
bottom_right_leftwindow = bottom_right & right_window_rec
bottom_right_rightwindow = bottom_right - right_window_rec
bottom_right_coating = (bottom_right_leftwindow - right_circle) + coating_thick_rec
bottom_right_uncoating = (bottom_right_leftwindow & right_circle) + (bottom_right_rightwindow - coating_thick_rec)

geo_coating = bottom_left_coating + bottom_right_coating + mid_rec + bottom_mid_rec
geo_uncoating = bottom_left_uncoating + bottom_right_uncoating

geo = geo_coating + geo_uncoating

x_pos = Parameter("x_pos")
integral_line = Line(
    (x_pos, 0),
    (x_pos, H0),
    1,
    parameterization=Parameterization({x_pos: (right_rx,(Lf+right_width))}),
)