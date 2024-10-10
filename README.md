# 2D_Reac_Diff_FDM
Solution of 2D Reaction-Diffusion non-linear equation with Finite Difference Method. The Schnakenberg model is related to the physical mechanisms behind the emergence of Turing patterns in nature. The model states that a chemical reaction occurs between two substances: the ‘slow’ activator $`u`$ and the ‘fast’ inhibitor $`v`$. The following boundary-value problem is considered in a space-time domain $`\Omega = (0, 4) \times (0, 4) \times I = (0, 20)`$:

```math
\begin{equation}
    \begin{dcases} 
        \frac{\partial u}{\partial t} = D_u \Delta u + k(a-u+u^2v), \ \ \ \ \ (x, y, t) \in \Omega \times I \\
        \frac{\partial v}{\partial t} = D_v \Delta v + k(b-u^2v), \ \ \ \ \ (x, y, t) \in \Omega \times I \\
        \nabla u \cdot \boldsymbol{n}=0, \ \ \ \ \ (x, y, t) \in \partial \Omega \\
        \nabla v \cdot \boldsymbol{n}=0, \ \ \ \ \ (x, y, t) \in \partial \Omega \\
        u(x, y, 0)=a+b+r(x,y), \ \ \ \ \ (x, y) \in \Omega \\
        v(x, y, 0)=\frac{b}{(a+b)^2}, \ \ \ \ \ (x, y) \in \Omega \\
    \end{dcases}
\end{equation}
```

where $`D_u=0.05`$ and $`D_v=1.0`$ are the effective diffusivity constants whereas the reaction constants are given by $`k=5.0`$, $`a=0.1305`$ and $`b=0.7695`$. A small non-uniform pseudo-random perturbation $`r(x,y)`$ is introduced in the initial concentration of the activator. 

A doubly-uniform grid is presented with $`N_x=N_y=100`$. The problem is discretized in space with the Finite Difference Method. Two different methods are used for the time integration scheme: the Forward-Euler (FE) method and the Backward-Euler (BE) method. The stability condition for the explicit FE method is investigated by considering a simpler linear problem, from which a suitable time step for the original problem is selected. The last method is combined with the Newton-Raphson (NR) scheme as the non-linear solver algorithm, thus it is designated as the BENR scheme. The NR algorithm is given by the following expression:

```math
\begin{equation}
    \boldsymbol{U}_{n+1}^{m+1} = \boldsymbol{U}_{n+1}^m + \left[ \boldsymbol{I} - \Delta t J \left( \boldsymbol{U}_{n+1}^{m} \right) \right]^-1 \left[ \boldsymbol{U}_{n} + \Delta t F(\boldsymbol{U}_{n+1}^{m}) - \boldsymbol{U}_{n+1}^{m} \right]
\end{equation}
```

where $`\boldsymbol{U}`$ is the solution vector, `n`$ is the time step number, $`m`$ is the iteration number, $`J`$ is the Jacobian and $`F`$ is the discretized formulation following the spatial discretization of the problem. The residual that is minimized by the NR algorithm is given by $`\epsilon^m= || \boldsymbol{U}_n + \Delta t F(\boldsymbol{U}_{n+1}^{m}) - \boldsymbol{U}_{n+1}^{m} ||_2`$. The solution convergence with both methods is analyzed as well as the computational time.
