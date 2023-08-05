from sympy import Symbol, Function, sqrt, Number, Min, Pow, sqrt

from modulus.eq.pde import PDE
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
class SlurryViscosity(PDE):

    name = "SlurryViscosity"

    def __init__(
        self, dim=2, time=True
    ):  # TODO add density into model
        
        # set params
        self.dim = dim
        self.time = time

        # params
        L_ref = 0.0005
        U_ref = 1.0
        time_scale = L_ref / U_ref
        viscosity_scale = L_ref ** 2 / time_scale

        # model coefficients
        self.mu0 = 15  / viscosity_scale
        self.mu00 = 0.1  / viscosity_scale
        self.mu_lambda = 16
        self.mu_a = 2
        self.mu_n = 0.83

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        # velocity componets
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        if self.dim == 3:
            w = Function("w")(*input_variables)
        else:
            w = Number(0)

        # slurry eq
        gamma_dot = sqrt(Pow(u.diff(y),2)+Pow(v.diff(x),2))
        
        # set equations
        self.equations = {}
        self.equations["mu2"] = self.mu00 + (self.mu0-self.mu00)*Pow(1+Pow(self.mu_lambda*gamma_dot,self.mu_a),((self.mu_n-1)/self.mu_a))
