# Solution of Non-linear Coupled Reaction-Diffusion Equation
# Joao Pedro Colaco Romana 2023

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la
import time


def initial_conditions(N, a, b, r):
    u = a + b + r
    v = np.full((N + 1, N + 1), b / (a + b) ** 2)
    return u, v


def FDLaplacian2D_Neumann(N, d):
    Dx = sp.diags([N * [-1 / d], N * [1 / d]], [0, 1], shape=(N, N + 1))
    Lxx = Dx.transpose().dot(Dx)
    A = sp.kron(sp.eye(N + 1), Lxx) + sp.kron(Lxx, sp.eye(N + 1))
    return A


def FDLaplacian2D_Dirichlet(N, d):
    Dx = sp.diags([(N - 1) * [1 / d], (N - 1) * [-1 / d]], [0, -1], shape=(N, N - 1))
    Lxx = Dx.transpose().dot(Dx)
    A = sp.kron(sp.eye(N - 1), Lxx) + sp.kron(Lxx, sp.eye(N - 1))
    return A


# Geometric and Time Parameters
L = 4
T = 20

# Model Constants
Du = 0.05
Dv = 1.0
k = 5
a = 0.1305
b = 0.7695

# Spatial Discretization
Nx = 100
dx = L / Nx

r = 0.01 * (a + b) * np.random.rand(Nx + 1, Nx + 1)

option = 0

while option != 5:
    plt.close('all')
    print('\n1 - Initial Condition')
    print('2 - Linear Problem')
    print('3 - FE Method')
    print('4 - BENR Method')
    print('5 - Exit')
    option = int(input('Choose an option: '))
    print('\n')

    if option == 1:
        uArr, vArr = initial_conditions(Nx, a, b, r)

        # Solution u_0
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        im1 = axs[0].imshow(uArr.T, extent=[-dx / 2, L + dx / 2, -dx / 2, L + dx / 2], origin='lower')
        fig.colorbar(im1, ax=axs[0])
        axs[0].set_title(r'$u(x,y,t=0)$')
        axs[0].set_xlabel(r'$x$')
        axs[0].set_ylabel(r'$y$')
        axs[0].set_xlim([0, 4])
        axs[0].set_ylim([0, 4])
        im2 = axs[1].imshow(vArr.T, extent=[-dx / 2, L + dx / 2, -dx / 2, L + dx / 2], origin='lower')
        fig.colorbar(im2, ax=axs[1])
        axs[1].set_title(r'$v(x,y,t=0)$')
        axs[1].set_xlabel(r'$x$')
        axs[1].set_xlim([0, 4])
        axs[1].set_ylim([0, 4])
        plt.savefig('initial_condition_Nx%02d.png' % Nx, bbox_inches='tight', dpi = 300)

        plt.show(block='True')

    elif option == 2:
        N = 25
        d = L / Nx

        A = FDLaplacian2D_Dirichlet(N, d)
        ki = [2, 5, 10]

        # Stability Condition for the Simpler Linear Problem
        for i in range(3):
            # Computes maximum eigenvalue
            max_eig = max(la.eigsh(A + ki[i] * np.identity((N - 1) ** 2), which='LM')[0])

            min_Nt = 0.5 * T * max_eig
            delta_t = T / min_Nt
            print('\nFor the linear problem to be stable, the minimum time number of time intervals is ' + str(min_Nt) + ' for k=' + str(ki[i]))
            print('For the linear problem to be stable, the maximum time step is ' + str(delta_t) + ' for k=' + str(ki[i]))

    elif option == 3:
        # Time Discretization
        Nt = 10 ** 5
        dt = T / Nt

        # Coupled System Formulation
        u_k = np.reshape(uArr, (Nx + 1) ** 2, order='F')
        v_k = np.reshape(vArr, (Nx + 1) ** 2, order='F')

        A = FDLaplacian2D_Neumann(Nx, dx)

        t = 0
        start_time = time.time()

        for nt in range(Nt):
            t = t + dt

            f_u = a - u_k + np.multiply(np.multiply(u_k, u_k), v_k)
            f_v = b - np.multiply(np.multiply(u_k, u_k), v_k)
            u_k = u_k + dt * (-Du * A @ u_k + k * f_u)
            v_k = v_k + dt * (-Dv * A @ v_k + k * f_v)

            # Visualizing the Solution
            if (nt + 1) % 10 ** 4 == 0:
                # Reshaping the Solution Vector into 2D array
                u = np.reshape(u_k, ((Nx + 1), (Nx + 1)), order='F')
                v = np.reshape(v_k, ((Nx + 1), (Nx + 1)), order='F')

                fig, axs = plt.subplots(1, 2, figsize=(12, 5))
                im1 = axs[0].imshow(u.T, extent=[-dx / 2, L + dx / 2, -dx / 2, L + dx / 2], origin='lower')
                fig.colorbar(im1, ax=axs[0])
                axs[0].set_title(r'$u(x,y,t=$' + str("%.1f" % t) + r'$)$')
                axs[0].set_xlabel(r'$x$')
                axs[0].set_ylabel(r'$y$')
                axs[0].set_xlim([0, 4])
                axs[0].set_ylim([0, 4])
                im2 = axs[1].imshow(v.T, extent=[-dx / 2, L + dx / 2, -dx / 2, L + dx / 2], origin='lower')
                fig.colorbar(im2, ax=axs[1])
                axs[1].set_title(r'$v(x,y,t=$' + str("%.1f" % t) + r'$)$')
                axs[1].set_xlabel(r'$x$')
                axs[1].set_xlim([0, 4])
                axs[1].set_ylim([0, 4])
                plt.savefig('FE_method_Nx%02d_t%.1f.png' % (Nx, t), bbox_inches='tight', dpi = 300)
                plt.pause(0.5)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('The time step is ' + str(dt))
        print('The total CPU time is ' + str("%.3f" % elapsed_time) + ' seconds')

    elif option == 4:
        # Time Discretization
        Nt = 40
        dt = T / Nt

        # Single System Formulation
        u_k = np.reshape(uArr, (Nx + 1) ** 2, order='F')
        v_k = np.reshape(vArr, (Nx + 1) ** 2, order='F')
        U_k = np.concatenate((u_k, v_k))

        # Initial Guess
        v_0 = np.concatenate((u_k, v_k))

        A = FDLaplacian2D_Neumann(Nx, dx)

        # Identity Matrices for Newton-Raphson method
        I_J = sp.eye((Nx + 1) ** 2)
        I_NR = sp.eye(2 * (Nx + 1) ** 2)

        # Residual
        epsilon = 10 ** (-3)

        t = 0
        start_time = time.time()

        for nt in range(Nt):
            t = t + dt
            i = 1

            u_k, v_k = np.split(U_k, 2)

            while True:
                u_u = np.multiply(u_k, u_k)
                u_v = np.multiply(u_k, v_k)
                u_u_v = np.multiply(np.multiply(u_k, u_k), v_k)
                u_u_diag = sp.diags(u_u)
                u_v_diag = sp.diags(u_v)

                # Jacobian matrix
                J_11 = -Du * A - k * I_J + 2 * k * u_v_diag
                J_12 = k * u_u_diag
                J_21 = -2 * k * u_v_diag
                J_22 = -Dv * A - k * u_u_diag

                J = sp.bmat([[J_11, J_12], [J_21, J_22]])

                F_u_k = -Du * A @ u_k + k * (a - u_k + u_u_v)
                F_v_k = -Dv * A @ v_k + k * (b - u_u_v)
                F_k = np.concatenate((F_u_k, F_v_k))

                # Newton-Raphson algorithm
                residual = v_0 + dt * F_k - U_k
                p_i = la.spsolve(I_NR - dt * J, residual)
                U_k = U_k + p_i
                u_k, v_k = np.split(U_k, 2)

                error = np.linalg.norm(residual)

                i += 1

                if error < epsilon:
                    break

            v_0 = U_k

            print("Time " + str(t))
            print("Number of inner iterations " + str(i))

            if (nt + 1) % 4 == 0:
                # Reshaping the Solution Vector into 2D array for Plot
                u = np.reshape(u_k, ((Nx + 1), (Nx + 1)), order='F')
                v = np.reshape(v_k, ((Nx + 1), (Nx + 1)), order='F')

                fig, axs = plt.subplots(1, 2, figsize=(12, 5))
                im1 = axs[0].imshow(u.T, extent=[-dx / 2, L + dx / 2, -dx / 2, L + dx / 2], origin='lower')
                fig.colorbar(im1, ax=axs[0])
                axs[0].set_title(r'$u(x,y,t=$' + str("%.1f" % t) + r'$)$')
                axs[0].set_xlabel(r'$x$')
                axs[0].set_ylabel(r'$y$')
                axs[0].set_xlim([0, 4])
                axs[0].set_ylim([0, 4])
                im2 = axs[1].imshow(v.T, extent=[-dx / 2, L + dx / 2, -dx / 2, L + dx / 2], origin='lower')
                fig.colorbar(im2, ax=axs[1])
                axs[1].set_title(r'$v(x,y,t=$' + str("%.1f" % t) + r'$)$')
                axs[1].set_xlabel(r'$x$')
                axs[1].set_xlim([0, 4])
                axs[1].set_ylim([0, 4])
                plt.savefig('BENR_method_Nx%02d_t%.1f.png' % (Nx, t), bbox_inches='tight', dpi = 300)
                plt.pause(0.5)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('The time step is ' + str(dt))
        print('The total CPU time is ' + str("%.3f" % elapsed_time) + ' seconds')
