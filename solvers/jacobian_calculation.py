from sympy import *
import numpy as np
# Distal 1- tube
T_twist = eye(4)
T_ext = eye(4)
T_tip = eye(4)
l, k, gamma, l_tip = symbols('l, k, gamma, l_tip')
T_twist[0, 0] = cos(gamma)
T_twist[1, 0] = sin(gamma)
T_twist[0, 1] = -sin(gamma)
T_twist[1, 1] = cos(gamma)
T_ext[1, 1] = cos(k * l)
T_ext[2, 1] = sin(k * l)
T_ext[1, 2] = -sin(k * l)
T_ext[2, 2] = cos(k * l)
# T_ext[1, 3] = (cos(k * l) - 1) / k
# T_ext[2, 3] = sin(k * l) / k
# T_ext[0, 0] = cos(k * l)
# T_ext[0, 2] = sin(k * l)
# T_ext[2, 0] = -sin(k * l)
# T_ext[2, 2] = cos(k * l)
T_ext[1, 3] = (cos(k * l) - 1) / k
T_ext[2, 3] = sin(k * l) / k
T_tip[2, 3] = l_tip
T_arcTube = T_twist * T_ext * T_tip
print('T_arcTube rot: {}'.format(T_arcTube[0:3, 0:3]))
print('T_arcTube vec: {}'.format(T_arcTube[0:3, 3]))
x, y, z, xrot, yrot = symbols('x, y, z, xrot, yrot')
fx = simplify(T_arcTube[0, 3])
fy = simplify(T_arcTube[1, 3])
fz = simplify(T_arcTube[2, 3])
print('x = {}'.format(fx))
print('y = {}'.format(fy))
print('z = {}'.format(fz))
# Z Y Intrinsic
fzrot = simplify(atan2(T_arcTube[1, 2], T_arcTube[0, 2]))
fyrot = simplify(acos(T_arcTube[2, 2]))
print('z_rot = {}'.format(fzrot))
print('y_rot = {}'.format(fyrot))
# X Y Extrinsic
# fxrot = simplify(atan2(T_rcmTip[2, 1], T_rcmTip[2, 2]))
# fyrot = simplify(atan2(-T_rcmTip[2, 0], sqrt(pow(T_rcmTip[0, 0], 2) + pow(T_rcmTip[1, 0], 2))))
# print('x_rot = {}'.format(fxrot))
# print('y_rot = {}'.format(fyrot))
# iterative for position only
# Derivative of x
dx_l = diff(fx, l)
dx_gamma = diff(fx, gamma)
# Derivative of y
dy_l = diff(fy, l)
dy_gamma = diff(fy, gamma)
# Derivative of z
dz_l = diff(fz, l)
dz_gamma = diff(fz, gamma)
# Derivative of zrot
dzrot_l = diff(fzrot, l)
dzrot_gamma = diff(fzrot, gamma)
# # Derivative of xrot
# dxrot_psi = diff(fxrot, psi)
# dxrot_phi = diff(fxrot, phi)
# dxrot_r = diff(fxrot, r)
# dxrot_l = diff(fxrot, l)
# dxrot_gamma = diff(fxrot, gamma)
# Derivative of yrot
dyrot_l = diff(fyrot, l)
dyrot_gamma = diff(fyrot, gamma)
# J = eye(3, 5)
# J[0, 0] = dx_psi
# J[0, 1] = dx_phi
# J[0, 2] = dx_r
# J[0, 3] = dx_gamma
# J[0, 4] = dx_l
#
# J[1, 0] = dy_psi
# J[1, 1] = dy_phi
# J[1, 2] = dy_r
# J[1, 3] = dy_gamma
# J[1, 4] = dy_l
#
# J[2, 0] = dz_psi
# J[2, 1] = dz_phi
# J[2, 2] = dz_r
# J[2, 3] = dz_gamma
# J[2, 4] = dz_l
# J = eye(4, 5)
# J[0, 0] = dx_psi
# J[0, 1] = dx_phi
# J[0, 2] = dx_r
# J[0, 3] = dx_gamma
# J[0, 4] = dx_l
#
# J[1, 0] = dy_psi
# J[1, 1] = dy_phi
# J[1, 2] = dy_r
# J[1, 3] = dy_gamma
# J[1, 4] = dy_l
#
# J[2, 0] = dz_psi
# J[2, 1] = dz_phi
# J[2, 2] = dz_r
# J[2, 3] = dz_gamma
# J[2, 4] = dz_l
#
# J[3, 0] = dxrot_psi
# J[3, 1] = dxrot_phi
# J[3, 2] = dxrot_r
# J[3, 3] = dxrot_gamma
# J[3, 4] = dxrot_l
J = eye(3, 3)
J[0, 0] = dx_gamma
J[0, 1] = dx_l
J[0, 2] = 0
J[1, 0] = dy_gamma
J[1, 1] = dy_l
J[1, 2] = 0
J[2, 0] = dz_gamma
J[2, 1] = dz_l
J[2, 2] = 0

print('Jacobian: {}'.format(J))
print(np.array(J.tolist()))
print('Done!')
# # X Y Extrinsic
# J[3, 0] = dxrot_psi
# J[3, 1] = dxrot_phi
# J[3, 2] = dxrot_r
# J[3, 3] = dxrot_gamma
# J[3, 4] = dxrot_l
#
# Z Y Intrinsic
#J[3, 1] = dzrot_gamma
#J[3, 2] = dzrot_l
#J[4, 1] = dyrot_gamma
#J[4, 2] = dyrot_l
#print('Jacobian: {}'.format(J))
#print(simplify(J[0, :]))
#print(simplify(J[1, :]))
#print(simplify(J[2, :]))
#print(simplify(J[3, :]))
#print(simplify(J[4, :]))
#print('Done!')
# inv_J = inv_quick(J)
# s_inv_J = simplify(inv_J)
# init_printing(use_unicode=True)
#
# print('Inv J size: {}'.format(len(inv_J)))
# print('Inverse Jacobian: ')
# print(inv_J)
#
# print('Simplified Inverse Jacobian: ')
# print(s_inv_J)