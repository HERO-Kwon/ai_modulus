"""Equations related to Navier Stokes Equations
"""

from sympy import Symbol, Function, Number, Pow, sqrt
import numpy as np

from modulus.eq.pde import PDE
from modulus.node import Node


class NavierStokes_VOF(PDE):
    """
    Compressible Navier Stokes equations

    Parameters
    ==========
    nu : float, Sympy Symbol/Expr, str
        The kinematic viscosity. If `nu` is a str then it is
        converted to Sympy Function of form `nu(x,y,z,t)`.
        If `nu` is a Sympy Symbol or Expression then this
        is substituted into the equation. This allows for
        variable viscosity.
    rho : float, Sympy Symbol/Expr, str
        The density of the fluid. If `rho` is a str then it is
        converted to Sympy Function of form 'rho(x,y,z,t)'.
        If 'rho' is a Sympy Symbol or Expression then this
        is substituted into the equation to allow for
        compressible Navier Stokes. Default is 1.
    dim : int
        Dimension of the Navier Stokes (2 or 3). Default is 3.
    time : bool
        If time-dependent equations or not. Default is True.
    mixed_form: bool
        If True, use the mixed formulation of the Navier-Stokes equations.

    Examples
    ========
    >>> ns = NavierStokes(nu=0.01, rho=1, dim=2)
    >>> ns.pprint()
      continuity: u__x + v__y
      momentum_x: u*u__x + v*u__y + p__x + u__t - 0.01*u__x__x - 0.01*u__y__y
      momentum_y: u*v__x + v*v__y + p__y + v__t - 0.01*v__x__x - 0.01*v__y__y
    >>> ns = NavierStokes(nu='nu', rho=1, dim=2, time=False)
    >>> ns.pprint()
      continuity: u__x + v__y
      momentum_x: -nu*u__x__x - nu*u__y__y + u*u__x + v*u__y - nu__x*u__x - nu__y*u__y + p__x
      momentum_y: -nu*v__x__x - nu*v__y__y + u*v__x + v*v__y - nu__x*v__x - nu__y*v__y + p__y
    """

    name = "NavierStokes_VOF"   
    #def __init__(self, mus=mus, rhos=rhos, sigma=sigma, g=g, U_ref=U_ref,L_ref=L_ref, dim=2, time=True, mixed_form=False):
    def __init__(self, mu1, mu2, rhos, sigma, g, U_ref,L_ref, dim=2, time=True, mixed_form=False):
        # set params
        self.dim = dim
        self.time = time
        self.mixed_form = mixed_form

        # coordinates
        x, y = Symbol("x"), Symbol("y")
        normal_x = Symbol("normal_x")
        normal_y = Symbol("normal_y")
        normal_z = Symbol("normal_z")


        # time
        t = Symbol("t")

        # parameters
        #self.mus = mus
        self.rhos = rhos
        self.sigma = sigma
        self.g = g
        
        #mu1 = mus[0]
        #mu2 = mus[1]
        rho1 = rhos[0]
        rho2 = rhos[1]
        
        
        # make input variables
        input_variables = {"x": x, "y": y, "t": t}

        # velocity componets
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        w = Number(0)

        # pressure
        p = Function("p")(*input_variables)

        # vof
        a = Function("a")(*input_variables)

        # kinematic viscosity
        if isinstance(mu1, str):
            mu1 = Function(mu1)(*input_variables)
        elif isinstance(mu1, (float, int)):
            mu1 = Number(mu1)
        if isinstance(mu2, str):
            mu2 = Function(mu2)(*input_variables)
        elif isinstance(mu2, (float, int)):
            mu2 = Number(mu2)

        # density
        if isinstance(rho1, str):
            rho1 = Function(rho1)(*input_variables)
        elif isinstance(rho1, (float, int)):
            rho1 = Number(rho1)
        if isinstance(rho2, str):
            rho2 = Function(rho2)(*input_variables)
        elif isinstance(rho2, (float, int)):
            rho2 = Number(rho2)

        #v_x = Function("v_x")(*input_variables)
        #p_x = Function("p_x")(*input_variables)
        # dynamic viscosity
        #mu1 = rho1 * nu1
        #mu2 = rho2 * nu2
        
        # vof eq
        mu = mu2 + (mu1 - mu2) * a
        mu_x = (mu1 - mu2) * a.diff(x)
        mu_y = (mu1 - mu2) * a.diff(y)
        rho = rho2 + (rho1 - rho2) * a        

        abs_interface_grad = sqrt(Pow(a.diff(x),2) + Pow(a.diff(y),2) + np.finfo(float).eps)
        curvature = - ( (a.diff(x).diff(x) + a.diff(y).diff(y))/abs_interface_grad - (a.diff(x)**2*a.diff(x).diff(x) + a.diff(y)**2*a.diff(y).diff(y) + 2*a.diff(x)*a.diff(y)*(a.diff(x).diff(y)))/Pow(abs_interface_grad, 3) )
        
        self.U_ref = U_ref
        self.L_ref = L_ref
        rho_ref = rho2

        one_Re = mu/(rho_ref*self.U_ref*self.L_ref)
        one_Re_x = mu_x/(rho_ref*self.U_ref*self.L_ref)
        one_Re_y = mu_y/(rho_ref*self.U_ref*self.L_ref)
        one_We = self.sigma/(rho_ref*self.U_ref**2*self.L_ref)
        one_Fr = self.g*self.L_ref/self.U_ref**2 
        
        # set equations
        self.equations = {}
        self.equations['PDE_m'] = u.diff(x) + v.diff(y)
        self.equations['PDE_a'] = a.diff(t) + u*a.diff(x) + v*a.diff(y)
        self.equations['PDE_u'] = (u.diff(t) + u*u.diff(x) + v*u.diff(y))*rho/rho_ref + p.diff(x) - one_We*curvature*a.diff(x) - one_Re*(u.diff(x).diff(x) + u.diff(y).diff(y)) - 2.0*one_Re_x*u.diff(x) - one_Re_y*(u.diff(y) + v.diff(x)) 
        self.equations['PDE_v'] = (v.diff(t) + u*v.diff(x) + v*v.diff(y))*rho/rho_ref + p.diff(y) - one_We*curvature*a.diff(y) - one_Re*(v.diff(x).diff(x) + v.diff(y).diff(y)) - rho/rho_ref*one_Fr - 2.0*one_Re_y*v.diff(y) - one_Re_x*(u.diff(y) + v.diff(x)) 


        '''
        # set equations
        self.equations = {}
        self.equations["continuity"] = (
            rho.diff(t) + (rho * u).diff(x) + (rho * v).diff(y) + (rho * w).diff(z)
        )

        if not self.mixed_form:
            curl = Number(0) if rho.diff(x) == 0 else u.diff(x) + v.diff(y) + w.diff(z)
            self.equations["momentum_x"] = (
                (rho * u).diff(t)
                + (
                    u * ((rho * u).diff(x))
                    + v * ((rho * u).diff(y))
                    + w * ((rho * u).diff(z))
                    + rho * u * (curl)
                )
                + p.diff(x)
                - (-2 / 3 * mu * (curl)).diff(x)
                - (mu * u.diff(x)).diff(x)
                - (mu * u.diff(y)).diff(y)
                - (mu * u.diff(z)).diff(z)
                - (mu * (curl).diff(x))
            )
            self.equations["momentum_y"] = (
                (rho * v).diff(t)
                + (
                    u * ((rho * v).diff(x))
                    + v * ((rho * v).diff(y))
                    + w * ((rho * v).diff(z))
                    + rho * v * (curl)
                )
                + p.diff(y)
                - (-2 / 3 * mu * (curl)).diff(y)
                - (mu * v.diff(x)).diff(x)
                - (mu * v.diff(y)).diff(y)
                - (mu * v.diff(z)).diff(z)
                - (mu * (curl).diff(y))
            )
            self.equations["momentum_z"] = (
                (rho * w).diff(t)
                + (
                    u * ((rho * w).diff(x))
                    + v * ((rho * w).diff(y))
                    + w * ((rho * w).diff(z))
                    + rho * w * (curl)
                )
                + p.diff(z)
                - (-2 / 3 * mu * (curl)).diff(z)
                - (mu * w.diff(x)).diff(x)
                - (mu * w.diff(y)).diff(y)
                - (mu * w.diff(z)).diff(z)
                - (mu * (curl).diff(z))
            )

            if self.dim == 2:
                self.equations.pop("momentum_z")

        elif self.mixed_form:
            u_x = Function("u_x")(*input_variables)
            u_y = Function("u_y")(*input_variables)
            u_z = Function("u_z")(*input_variables)
            v_x = Function("v_x")(*input_variables)
            v_y = Function("v_y")(*input_variables)
            v_z = Function("v_z")(*input_variables)

            if self.dim == 3:
                w_x = Function("w_x")(*input_variables)
                w_y = Function("w_y")(*input_variables)
                w_z = Function("w_z")(*input_variables)
            else:
                w_x = Number(0)
                w_y = Number(0)
                w_z = Number(0)
                u_z = Number(0)
                v_z = Number(0)

            curl = Number(0) if rho.diff(x) == 0 else u_x + v_y + w_z
            self.equations["momentum_x"] = (
                (rho * u).diff(t)
                + (
                    u * ((rho * u.diff(x)))
                    + v * ((rho * u.diff(y)))
                    + w * ((rho * u.diff(z)))
                    + rho * u * (curl)
                )
                + p.diff(x)
                - (-2 / 3 * mu * (curl)).diff(x)
                - (mu * u_x).diff(x)
                - (mu * u_y).diff(y)
                - (mu * u_z).diff(z)
                - (mu * (curl).diff(x))
            )
            self.equations["momentum_y"] = (
                (rho * v).diff(t)
                + (
                    u * ((rho * v.diff(x)))
                    + v * ((rho * v.diff(y)))
                    + w * ((rho * v.diff(z)))
                    + rho * v * (curl)
                )
                + p.diff(y)
                - (-2 / 3 * mu * (curl)).diff(y)
                - (mu * v_x).diff(x)
                - (mu * v_y).diff(y)
                - (mu * v_z).diff(z)
                - (mu * (curl).diff(y))
            )
            self.equations["momentum_z"] = (
                (rho * w).diff(t)
                + (
                    u * ((rho * w.diff(x)))
                    + v * ((rho * w.diff(y)))
                    + w * ((rho * w.diff(z)))
                    + rho * w * (curl)
                )
                + p.diff(z)
                - (-2 / 3 * mu * (curl)).diff(z)
                - (mu * w_x).diff(x)
                - (mu * w_y).diff(y)
                - (mu * w_z).diff(z)
                - (mu * (curl).diff(z))
            )
            self.equations["compatibility_u_x"] = u.diff(x) - u_x
            self.equations["compatibility_u_y"] = u.diff(y) - u_y
            self.equations["compatibility_u_z"] = u.diff(z) - u_z
            self.equations["compatibility_v_x"] = v.diff(x) - v_x
            self.equations["compatibility_v_y"] = v.diff(y) - v_y
            self.equations["compatibility_v_z"] = v.diff(z) - v_z
            self.equations["compatibility_w_x"] = w.diff(x) - w_x
            self.equations["compatibility_w_y"] = w.diff(y) - w_y
            self.equations["compatibility_w_z"] = w.diff(z) - w_z
            self.equations["compatibility_u_xy"] = u_x.diff(y) - u_y.diff(x)
            self.equations["compatibility_u_xz"] = u_x.diff(z) - u_z.diff(x)
            self.equations["compatibility_u_yz"] = u_y.diff(z) - u_z.diff(y)
            self.equations["compatibility_v_xy"] = v_x.diff(y) - v_y.diff(x)
            self.equations["compatibility_v_xz"] = v_x.diff(z) - v_z.diff(x)
            self.equations["compatibility_v_yz"] = v_y.diff(z) - v_z.diff(y)
            self.equations["compatibility_w_xy"] = w_x.diff(y) - w_y.diff(x)
            self.equations["compatibility_w_xz"] = w_x.diff(z) - w_z.diff(x)
            self.equations["compatibility_w_yz"] = w_y.diff(z) - w_z.diff(y)

            if self.dim == 2:
                self.equations.pop("momentum_z")
                self.equations.pop("compatibility_u_z")
                self.equations.pop("compatibility_v_z")
                self.equations.pop("compatibility_w_x")
                self.equations.pop("compatibility_w_y")
                self.equations.pop("compatibility_w_z")
                self.equations.pop("compatibility_u_xz")
                self.equations.pop("compatibility_u_yz")
                self.equations.pop("compatibility_v_xz")
                self.equations.pop("compatibility_v_yz")
                self.equations.pop("compatibility_w_xy")
                self.equations.pop("compatibility_w_xz")
                self.equations.pop("compatibility_w_yz")
        '''

class GradNormal(PDE):
    """
    Implementation of the gradient boundary condition

    Parameters
    ==========
    T : str
        The dependent variable.
    dim : int
        Dimension of the equations (1, 2, or 3). Default is 3.
    time : bool
        If time-dependent equations or not. Default is True.

    Examples
    ========
    >>> gn = ns = GradNormal(T='T')
    >>> gn.pprint()
      normal_gradient_T: normal_x*T__x + normal_y*T__y + normal_z*T__z
    """

    name = "GradNormal"

    def __init__(self, T, dim=3, time=True):
        self.T = T
        self.dim = dim
        self.time = time

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        normal_x = Symbol("normal_x")
        normal_y = Symbol("normal_y")
        normal_z = Symbol("normal_z")

        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 1:
            input_variables.pop("y")
            input_variables.pop("z")
        elif self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        # variables to set the gradients (example Temperature)
        T = Function(T)(*input_variables)

        # set equations
        self.equations = {}
        self.equations["normal_gradient_" + self.T] = (
            normal_x * T.diff(x) + normal_y * T.diff(y) + normal_z * T.diff(z)
        )


class Curl(PDE):
    """
    del cross vector operator

    Parameters
    ==========
    vector : tuple of 3 Sympy Exprs, floats, ints or strings
        This will be the vector to take the curl of.
    curl_name : tuple of 3 strings
        These will be the output names of the curl operations.

    Examples
    ========
    >>> c = Curl((0,0,'phi'), ('u','v','w'))
    >>> c.pprint()
      u: phi__y
      v: -phi__x
      w: 0
    """

    name = "Curl"

    def __init__(self, vector, curl_name=["u", "v", "w"]):
        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z}

        # vector
        v_0 = vector[0]
        v_1 = vector[1]
        v_2 = vector[2]

        # make funtions
        if type(v_0) is str:
            v_0 = Function(v_0)(*input_variables)
        elif type(v_0) in [float, int]:
            v_0 = Number(v_0)
        if type(v_1) is str:
            v_1 = Function(v_1)(*input_variables)
        elif type(v_1) in [float, int]:
            v_1 = Number(v_1)
        if type(v_2) is str:
            v_2 = Function(v_2)(*input_variables)
        elif type(v_2) in [float, int]:
            v_2 = Number(v_2)

        # curl
        curl_0 = v_2.diff(y) - v_1.diff(z)
        curl_1 = v_0.diff(z) - v_2.diff(x)
        curl_2 = v_1.diff(x) - v_0.diff(y)

        # set equations
        self.equations = {}
        self.equations[curl_name[0]] = curl_0
        self.equations[curl_name[1]] = curl_1
        self.equations[curl_name[2]] = curl_2


class CompressibleIntegralContinuity(PDE):
    """
    Compressible Integral Continuity

    Parameters
    ==========
    rho : float, Sympy Symbol/Expr, str
        The density of the fluid. If `rho` is a str then it is
        converted to Sympy Function of form 'rho(x,y,z,t)'.
        If 'rho' is a Sympy Symbol or Expression then this
        is substituted into the equation to allow for
        compressibility. Default is 1.
    dim : int
        Dimension of the equations (1, 2, or 3). Default is 3.
    """

    name = "CompressibleIntegralContinuity"

    def __init__(self, rho=1, vec=["u", "v", "w"]):
        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z}
        self.dim = len(vec)
        if self.dim == 1:
            input_variables.pop("y")
            input_variables.pop("z")
        elif self.dim == 2:
            input_variables.pop("z")

        # normal
        normal = [Symbol("normal_x"), Symbol("normal_y"), Symbol("normal_z")]

        # density
        if isinstance(rho, str):
            rho = Function(rho)(*input_variables)
        elif isinstance(rho, (float, int)):
            rho = Number(rho)

        # make input variables
        self.equations = {}
        self.equations["integral_continuity"] = 0
        for v, n in zip(vec, normal):
            self.equations["integral_continuity"] += Symbol(v) * n * rho


class FluxContinuity(PDE):
    """
    Flux Continuity for arbitrary variable. Includes advective and diffusive flux

    Parameters
    ==========
    T : str
        The dependent variable.
    rho : float, Sympy Symbol/Expr, str
        The density of the fluid. If `rho` is a str then it is
        converted to Sympy Function of form 'rho(x,y,z,t)'.
        If 'rho' is a Sympy Symbol or Expression then this
        is substituted into the equation to allow for
        compressibility. Default is 1.
    dim : int
        Dimension of the equations (1, 2, or 3). Default is 3.
    """

    name = "FluxContinuity"

    def __init__(self, T="T", D="D", rho=1, vec=["u", "v", "w"]):
        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z}
        self.dim = len(vec)
        if self.dim == 1:
            input_variables.pop("y")
            input_variables.pop("z")
        elif self.dim == 2:
            input_variables.pop("z")

        # normal
        normal = [Symbol("normal_x"), Symbol("normal_y"), Symbol("normal_z")]

        # density
        if isinstance(rho, str):
            rho = Function(rho)(*input_variables)
        elif isinstance(rho, (float, int)):
            rho = Number(rho)

        # diffusion coefficient
        if isinstance(D, str):
            D = Function(D)(*input_variables)
        elif isinstance(D, (float, int)):
            D = Number(D)

        # variables to set the flux (example Temperature)
        T = Function(T)(*input_variables)

        gradient = [T.diff(x), T.diff(y), T.diff(z)]

        # make input variables
        self.equations = {}
        self.equations[str(T) + "_flux"] = 0
        for v, n, g in zip(vec, normal, gradient):
            self.equations[str(T) + "_flux"] += (
                Symbol(v) * n * rho * T - rho * D * n * g
            )
