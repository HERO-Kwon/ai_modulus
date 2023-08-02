from sympy import Symbol, Function, sqrt, Number, Min, Pow, sqrt

from modulus.eq.pde import PDE


class MuEquation(PDE):
    """
    Zero Equation Turbulence model

    Parameters
    ==========
    nu : float
        The kinematic viscosity of the fluid.
    max_distance : float
        The maximum wall distance in the flow field.
    rho : float, Sympy Symbol/Expr, str
        The density. If `rho` is a str then it is
        converted to Sympy Function of form 'rho(x,y,z,t)'.
        If 'rho' is a Sympy Symbol or Expression then this
        is substituted into the equation. Default is 1.
    dim : int
        Dimension of the Zero Equation Turbulence model (2 or 3).
        Default is 3.
    time : bool
        If time-dependent equations or not. Default is True.

    Example
    ========
    >>> zeroEq = ZeroEquation(nu=0.1, max_distance=2.0, dim=2)
    >>> kEp.pprint()
      nu: sqrt((u__y + v__x)**2 + 2*u__x**2 + 2*v__y**2)
      *Min(0.18, 0.419*normal_distance)**2 + 0.1
    """

    name = "MuEquation"

    def __init__(
        self, dim=2, time=True
    ):  # TODO add density into model
        
        # set params
        self.dim = dim
        self.time = time

        # model coefficients
        self.mu0 = 15
        self.mu00 = 0.1
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

        # normals
        normal_x = -1 * Symbol(
            "normal_x"
        )  # Multiply by -1 to flip the direction of normal
        normal_y = -1 * Symbol(
            "normal_y"
        )  # Multiply by -1 to flip the direction of normal
        
        # wall distance
        normal_distance = Function("normal_distance")(*input_variables)
        #normal_distance = Function("sdf")(*input_variables)

        # sheer rate
        u_parallel_to_wall = [
            u - (u * normal_x + v * normal_y) * normal_x,
            v - (u * normal_x + v * normal_y) * normal_y,
        ]
        du_parallel_to_wall_dx = [
            u.diff(x) - (u.diff(x) * normal_x + v.diff(x) * normal_y) * normal_x,
            v.diff(x) - (u.diff(x) * normal_x + v.diff(x) * normal_y) * normal_y,
        ]
        du_parallel_to_wall_dy = [
            u.diff(y) - (u.diff(y) * normal_x + v.diff(y) * normal_y) * normal_x,
            v.diff(y) - (u.diff(y) * normal_x + v.diff(y) * normal_y) * normal_y,
        ]

        du_dsdf = [
            du_parallel_to_wall_dx[0] * normal_x + du_parallel_to_wall_dy[0] * normal_y,
            du_parallel_to_wall_dx[1] * normal_x + du_parallel_to_wall_dy[1] * normal_y,
        ]

        # wall distance
        #normal_distance = Function("sdf")(*input_variables)

        # slurry eq
        gamma_dot = sqrt(Pow(du_dsdf[0],2)+Pow(du_dsdf[1],2))

        # set equations
        self.equations = {}
        self.equations["mu2"] = self.mu00 + (self.mu0-self.mu00)*Pow(1+Pow(self.mu_lambda*gamma_dot,self.mu_a),((self.mu_n-1)/self.mu_a))
