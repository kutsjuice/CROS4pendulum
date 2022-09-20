using ForwardDiff
using LinearAlgebra
using PyPlot


dt = 0.001;
m = 1; L = 5;
g(t) = 9.81 + 0.05*sin(2*π*t)

function pendulum(u)
    F = similar(u);
    F[1] = u[3];
    F[2] = u[4];
    F[3] = - 1 / (m*L) * u[1] * u[6]
    F[4] = - 1 / (m*L) * u[2] * u[6] - g(u[5]);
    F[5] = 1;
    F[6] = u[1]^2 + u[2]^2 - L^2; 
    return F
end

function solveCROS(func, u0, dt, tspan, NN)
    tt = tspan[1]:dt:tspan[2];
    u = Matrix{Float64}(undef, length(tt), length(u0));
    u[1,:] = u0;

    MM = length(u0) - NN;
    E = diagm(vcat(ones(NN), zeros(MM)));
    α = (1+1im)/2;

    for i in 2:length(tt)
        J = ForwardDiff.jacobian(func, u[i-1,:])
        k = (E-α*dt*J)\func(u[i-1,:])
        u[i,:] = u[i-1, :] + dt * real.(k);
    end

    return (tt, u)

end

(tt, u) = solveCROS(pendulum, [3.0, -4.0, 0.0, 0.0, 0.0, 0.0], dt, [0, 2], 5)
begin
    fig, ax = subplots(2,1)
    ax[1].plot(u[:,1], u[:,2]); ax[1].set_aspect(1)
    ax[1].set_title("траектория маятника")
    ax[1].set_xlabel("x, [м]"); ax[1].set_ylabel("y, [м]")
    ax[2].plot(tt, sqrt.(u[:,1].^2+u[:,2].^2))
    ax[2].set_title("изменение длины стержня")
    ax[2].set_xlabel("Время, [с]"); ax[2].set_ylabel("Длина, [м]")
end