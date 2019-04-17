from sympy import *
import dill

dill.settings['recurse'] = True

class ComputeJacobianEquation:
    def __init__(self, num_tubes, save_path=None, load_path=None):
        self.save_path = save_path
        self.load_path = load_path
        self.num_tubes = num_tubes
        self.J_lambda = self._compute_jacobian()

    def get_jacobian(self):
        return self.J_lambda

    def _get_T_twist(self, gamma):
        T_twist = eye(4)
        T_twist[0, 0] = cos(gamma)
        T_twist[1, 0] = sin(gamma)
        T_twist[0, 1] = -sin(gamma)
        T_twist[1, 1] = cos(gamma)
        return T_twist

    def _get_T_ext(self, k, l):
        T_ext = eye(4)
        T_ext[1, 1] = cos(k * l)
        T_ext[2, 1] = sin(k * l)
        T_ext[1, 2] = -sin(k * l)
        T_ext[2, 2] = cos(k * l)
        T_ext[1, 3] = (cos(k * l) - 1) / k
        T_ext[2, 3] = sin(k * l) / k
        return T_ext

    def _get_T_tip(self, l_tip):
        T_tip = eye(4)
        T_tip[2, 3] = l_tip
        return T_tip

    def _get_fx_fy_fz(self, T):
        fx = simplify(T[0, 3])
        fy = simplify(T[1, 3])
        fz = simplify(T[2, 3])
        return fx, fy, fz

    def _get_derivatives(self, gamma, l, fx, fy, fz):
        dx_gamma, dx_l, dy_gamma, dy_l, dz_gamma, dz_l = [], [], [], [], [], []
        for tube in range(self.num_tubes):
            dx_gamma.append(diff(fx, gamma[tube]))
            dx_l.append(diff(fx, l[tube]))
            # Derivative of y
            dy_gamma.append(diff(fy, gamma[tube]))
            dy_l.append(diff(fy, l[tube]))
            # Derivative of z
            dz_gamma.append(diff(fz, gamma[tube]))
            dz_l.append(diff(fz, l[tube]))
        return dx_gamma, dx_l, dy_gamma, dy_l, dz_gamma, dz_l

    def _compute_jacobian(self):
        if self.load_path:
            J_lambda = dill.load(open(self.load_path, "rb"))
            return J_lambda

        if self.num_tubes == 1:
            l = [symbols('l')]
            k = [symbols('k')]
            gamma = [symbols('gamma')]
            l_tip = [symbols('l_tip')]
            J = eye(3, 2)
        elif self.num_tubes == 2:
            l = symbols('l1, l2')
            k = symbols('k1, k2')
            gamma = symbols('gamma1, gamma2')
            l_tip = symbols('l_tip1, l_tip2')
            J = eye(3, 4)
        else:
            raise NotImplemented

        T_arcTube = eye(4, 4)
        for tube in range(self.num_tubes):
            T_twist = self._get_T_twist(gamma[tube])
            T_ext = self._get_T_ext(k[tube], l[tube])
            T_tip = self._get_T_tip(l_tip[tube])
            T_arcTube = T_arcTube * T_twist * T_ext * T_tip

        fx, fy, fz = self._get_fx_fy_fz(T_arcTube)

        dx_gamma, dx_l, dy_gamma, dy_l, dz_gamma, dz_l = self._get_derivatives(gamma, l, fx, fy, fz)

        if self.num_tubes == 1:
            # non-square jacobian
            J[0, 0] = dx_gamma
            J[0, 1] = dx_l
            J[1, 0] = dy_gamma
            J[1, 1] = dy_l
            J[2, 0] = dz_gamma
            J[2, 1] = dz_l
        elif self.num_tubes == 2:
            J[0, 0] = dx_gamma[0]
            J[0, 1] = dx_l[0]
            J[0, 2] = dx_gamma[1]
            J[0, 3] = dx_l[1]

            J[1, 0] = dy_gamma[0]
            J[1, 1] = dy_l[0]
            J[1, 2] = dy_gamma[1]
            J[1, 3] = dy_l[1]

            J[2, 0] = dz_gamma[0]
            J[2, 1] = dz_l[0]
            J[2, 2] = dz_gamma[1]
            J[2, 3] = dz_l[1]
        else:
            raise NotImplemented

        J_lambda = lambdify((k, l_tip, gamma, l), J)
        if self.save_path:
            dill.dump(J_lambda, open(self.save_path, "wb+"))

        return J_lambda

# def get_jacobian_matrix(num_tubes):
#     if num_tubes == 1:
#         # Distal 1- tube
#         # Compute arc transform
#         T_twist = eye(4)
#         T_ext = eye(4)
#         T_tip = eye(4)
#         l, k, gamma, l_tip = symbols('l, k, gamma, l_tip')
#         T_twist[0, 0] = cos(gamma)
#         T_twist[1, 0] = sin(gamma)
#         T_twist[0, 1] = -sin(gamma)
#         T_twist[1, 1] = cos(gamma)
#         T_ext[1, 1] = cos(k * l)
#         T_ext[2, 1] = sin(k * l)
#         T_ext[1, 2] = -sin(k * l)
#         T_ext[2, 2] = cos(k * l)
#         T_ext[1, 3] = (cos(k * l) - 1) / k
#         T_ext[2, 3] = sin(k * l) / k
#         T_tip[2, 3] = l_tip
#         T_arcTube = T_twist * T_ext * T_tip
#         print('T_arcTube rot: {}'.format(T_arcTube[0:3, 0:3]))
#         print('T_arcTube vec: {}'.format(T_arcTube[0:3, 3]))
#         x, y, z, xrot, yrot = symbols('x, y, z, xrot, yrot')
#         fx = simplify(T_arcTube[0, 3])
#         fy = simplify(T_arcTube[1, 3])
#         fz = simplify(T_arcTube[2, 3])
#         print('x = {}'.format(fx))
#         print('y = {}'.format(fy))
#         print('z = {}'.format(fz))
#         dx_l = diff(fx, l)
#         dx_gamma = diff(fx, gamma)
#         # Derivative of y
#         dy_l = diff(fy, l)
#         dy_gamma = diff(fy, gamma)
#         # Derivative of z
#         dz_l = diff(fz, l)
#         dz_gamma = diff(fz, gamma)
#         # non-square jacobian
#         J = eye(3, 2)
#         J[0, 0] = dx_gamma
#         J[0, 1] = dx_l
#         J[1, 0] = dy_gamma
#         J[1, 1] = dy_l
#         J[2, 0] = dz_gamma
#         J[2, 1] = dz_l
#
#         return lambdify((k, l_tip, gamma, l), J)
#
#     if num_tubes == 2:
#         # Distal 1-tube
#         # Compute arc transform
#         T_twist = eye(4)
#         T_ext = eye(4)
#         T_tip = eye(4)
#         T_arcTube = eye(4)
#         l = symbols('l1, l2')
#         k = symbols('k1, k2')
#         gamma = symbols('gamma1, gamma2')
#         l_tip = symbols('l_tip1, l_tip2')
#         for i in range(num_tubes):
#             T_twist[0, 0] = cos(gamma[i])
#             T_twist[1, 0] = sin(gamma[i])
#             T_twist[0, 1] = -sin(gamma[i])
#             T_twist[1, 1] = cos(gamma[i])
#             T_ext[1, 1] = cos(k[i] * l[i])
#             T_ext[2, 1] = sin(k[i] * l[i])
#             T_ext[1, 2] = -sin(k[i] * l[i])
#             T_ext[2, 2] = cos(k[i] * l[i])
#             T_ext[1, 3] = (cos(k[i] * l[i]) - 1) / k[i]
#             T_ext[2, 3] = sin(k[i] * l[i]) / k[i]
#             T_tip[2, 3] = l_tip[i]
#             T_arcTube = T_arcTube * T_twist * T_ext * T_tip
#         print('T_arcTube rot: {}'.format(T_arcTube[0:3, 0:3]))
#         print('T_arcTube vec: {}'.format(T_arcTube[0:3, 3]))
#
#         x, y, z, xrot, yrot = symbols('x, y, z, xrot, yrot')
#         fx = simplify(T_arcTube[0, 3])
#         fy = simplify(T_arcTube[1, 3])
#         fz = simplify(T_arcTube[2, 3])
#         print('x = {}'.format(fx))
#         print('y = {}'.format(fy))
#         print('z = {}'.format(fz))
#
#         dx_l = []
#         dy_l = []
#         dz_l = []
#         dx_gamma = []
#         dy_gamma = []
#         dz_gamma = []
#
#         for i in range(num_tubes):
#             dx_l.append(diff(fx, l[i]))
#             dy_l.append(diff(fy, l[i]))
#             dz_l.append(diff(fz, l[i]))
#             dx_gamma.append(diff(fx, gamma[i]))
#             dy_gamma.append(diff(fy, gamma[i]))
#             dz_gamma.append(diff(fz, gamma[i]))
#
#         J = zeros(3, 4)
#         J[0, 0] = dx_gamma[0]
#         J[0, 1] = dx_l[0]
#         J[0, 2] = dx_gamma[1]
#         J[0, 3] = dx_l[1]
#
#         J[1, 0] = dy_gamma[0]
#         J[1, 1] = dy_l[0]
#         J[1, 2] = dy_gamma[1]
#         J[1, 3] = dy_l[1]
#
#         J[2, 0] = dz_gamma[0]
#         J[2, 1] = dz_l[0]
#         J[2, 2] = dz_gamma[1]
#         J[2, 3] = dz_l[1]
#
#         print(J)
#
#         return lambdify((k, l_tip, gamma, l), J)
