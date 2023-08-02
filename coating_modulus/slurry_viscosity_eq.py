from sympy import Symbol, Function, sqrt, Number, Min, Pow, sqrt

from modulus.eq.pde import PDE


class SlurryViscosity(PDE):

    name = "SlurryViscosity"

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

        # slurry eq
        gamma_dot = sqrt(Pow(u.diff(y),2)+Pow(v.diff(x),2))

        # set equations
        self.equations = {}
        self.equations["mu2"] = self.mu00 + (self.mu0-self.mu00)*Pow(1+Pow(self.mu_lambda*gamma_dot,self.mu_a),((self.mu_n-1)/self.mu_a))
