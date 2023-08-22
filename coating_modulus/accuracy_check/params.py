'''
========
################
# Non dim params
################
length_scale = 0.0575  # m
velocity_scale = 5.7  # m/s
time_scale = length_scale / velocity_scale  # s
density_scale = 1.1614  # kg/m3
mass_scale = density_scale * length_scale ** 3  # kg
pressure_scale = mass_scale / (length_scale * time_scale ** 2)  # kg / (m s**2)
temp_scale = 273.15  # K
watt_scale = (mass_scale * length_scale ** 2) / (time_scale ** 3)  # kg m**2 / s**3
joule_scale = (mass_scale * length_scale ** 2) / (time_scale ** 2)  # kg * m**2 / s**2

##############################
# Nondimensionalization Params
##############################
# fluid params
nd_fluid_viscosity = fluid_viscosity / (
    length_scale ** 2 / time_scale
)  # need to divide by density to get previous viscosity
nd_fluid_density = fluid_density / density_scale
nd_fluid_specific_heat = fluid_specific_heat / (joule_scale / (mass_scale * temp_scale))
nd_fluid_conductivity = fluid_conductivity / (watt_scale / (length_scale * temp_scale))
nd_fluid_diffusivity = nd_fluid_conductivity / (
    nd_fluid_specific_heat * nd_fluid_density
)
'''
# parameters

rho2 = 1000 # density
rho1 = 1.18415
mu1 = 1.85508e-05  # kg/m-s
mu2 = 3.5
sigma=0.06 #surface_tension_coeff 
g = -9.8 # gravitational acceleration

# normalize params
length_scale = 0.0002  # m
velocity_scale = 1.0  # m/s
time_scale = length_scale / velocity_scale  # s
density_scale = rho2  # kg/m3
viscosity_scale = length_scale ** 2 / time_scale

L_ref = length_scale
U_ref = velocity_scale
rho_ref = density_scale
vis_ref = viscosity_scale
t_ref = time_scale


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