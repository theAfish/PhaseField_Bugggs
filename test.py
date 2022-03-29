import taichi as ti

ti.init(arch=ti.gpu)
PI = 3.14159265

d0 = 12.17  # m
Dl = 4.4
a2 = 0.078337
Omega = 0.55
Delta = 0.55
ke = 0.15
lam = 0.02
mel = -5.3  # liquidus slope


N = 256
m = 0.3
dx = 0.722 * d0
dt = 0.8 * dx**2 / (6 * Dl) * 1e-4
delta_s = 0.2  # -0.2
epsilon_0 = d0 / 0.277
tau_0 = a2 * lam * epsilon_0**2 / Dl

N_s = 4.0
N_well = 8.0
b = 0.05
N_0 = 4.0
l = 4.0
n_s = 3.0


phi = ti.field(dtype=ti.f32, shape=(N, N))
theta = ti.field(dtype=ti.f32, shape=(N, N))
p = ti.Vector.field(2, dtype=ti.f32, shape=(N, N))
debug = ti.field(dtype=ti.f32, shape=(N, N))

grad_px = ti.Vector.field(2, dtype=ti.f32, shape=(N, N))
grad_py = ti.Vector.field(2, dtype=ti.f32, shape=(N, N))
wgpx = ti.Vector.field(2, dtype=ti.f32, shape=(N, N))
wgpy = ti.Vector.field(2, dtype=ti.f32, shape=(N, N))


img = ti.Vector.field(3, dtype=ti.f32, shape=(N, N))
window = ti.ui.Window("test", (512, 512))
canvas = window.get_canvas()


@ti.func
def grad_x(u, x, y, h):
    return ti.Vector([u[x + 1, y][0] - u[x - 1, y][0], u[x, y + 1][0] - u[x, y - 1][0]]) / (2.0 * h)

@ti.func
def grad_y(u, x, y, h):
    return ti.Vector([u[x + 1, y][1] - u[x - 1, y][1], u[x, y + 1][1] - u[x, y - 1][1]]) / (2.0 * h)

@ti.func
def div(u, x, y, h):
    return (u[x + 1, y][0] - u[x - 1, y][0] + u[x, y + 1][1] - u[x, y - 1][1]) / (2.0 * h)


@ti.kernel
def init():
    # theta[N/2,N/2] = 0.0001
    for x, y in theta:
        # theta[x, y] = ti.random() * 2.0 * PI
        phi[x, y] = 1e-5
        if (x - N / 2) ** 2 + (y - N / 2) ** 2 < 15:
            phi[x, y] = 1.0
            # theta[x, y] = PI/3
        p[x, y][0] = phi[x, y] * ti.cos(theta[x, y])
        p[x, y][1] = phi[x, y] * ti.sin(theta[x, y])

@ti.kernel
def debug_init():
    for x, y in theta:
        # theta[x, y] = ti.random() * 2.0 * PI
        phi[x, y] = 1e-5
    for x, y in ti.ndrange(N/3, N/3):
        phi[2*x,2*y] = 1.
    for x, y in p:
        if (x - N / 2) ** 2 + (y - N / 2) ** 2 < 15:
            phi[x, y] = 1.0
            theta[x, y] = PI/4
        p[x, y][0] = phi[x, y] * ti.cos(theta[x, y])
        p[x, y][1] = phi[x, y] * ti.sin(theta[x, y])


@ti.kernel
def update():
    for x, y in phi:
        grad_px[x, y] = grad_x(p, x, y, dx)
        grad_py[x, y] = grad_y(p, x, y, dx)

        tgpx = ti.atan2(grad_px[x, y][1], grad_px[x, y][0])
        tgpy = ti.atan2(grad_py[x, y][1], grad_py[x, y][0])
        wx = epsilon_0 * (1. + delta_s*ti.cos(4.*tgpx-theta[x, y]))
        wy = epsilon_0 * (1. + delta_s*ti.cos(4.*tgpy-theta[x, y]))
        wgpx[x, y] = wx**2 * grad_px[x, y]
        wgpy[x, y] = wy**2 * grad_py[x, y]
        
    for x, y in phi:
        part1x = div(wgpx, x, y, dx) # Remember to add the second term!!!!!!!!!!!!!!!!!!!!!!!!!!!1
        part2x = (2. + 3. * phi[x, y] - 4. * phi[x, y]**2) * p[x, y][0]
        part3x = 6. * b * (ti.cos(N_well * theta[x, y]) - 1.) * phi[x, y] * p[x, y][0] / (1. + b)
        part4x = -2. * b * N_well * ti.sin(N_well * theta[x, y]) * phi[x, y] * p[x, y][1] / p[x, y].norm()**2 / (1. + b)
        part5x = -lam * ((Omega-1.)/(1.-ke)-Delta) * (1. - phi[x, y])**2 * p[x, y][0]

        part1y = div(wgpy, x, y, dx)
        part2y = (2. + 3. * phi[x, y] - 4. * phi[x, y]**2) * p[x, y][1]
        part3y = 6. * b * (ti.cos(N_well * theta[x, y]) - 1.) * phi[x, y] * p[x, y][1] / (1. + b)
        part4y = 2. * b * N_well * ti.sin(N_well * theta[x, y]) * phi[x, y] * p[x, y][0] / p[x, y].norm()**2 / (1. + b)
        part5y = -lam * ((Omega-1.)/(1.-ke)-Delta) * (1. - phi[x, y])**2 * p[x, y][1]

        p[x, y][0] += dt / tau_0 * (part1x + part2x + part3x + part4x + part5x)
        p[x, y][1] += dt / tau_0 * (part1y + part2y + part3y + part4y + part5y)
        theta[x, y] = ti.atan2(p[x, y][1], p[x, y][0])
        phi[x, y] = p[x, y].norm()


@ti.kernel
def draw():
    # print(theta[N/2, N/2])
    for x, y in img:
        # img[x, y] = debug[x, y] * ti.Vector([1.0,1.0,1.0])
        img[x, y] = phi[x, y] * ti.Vector([ti.sin(theta[x, y]), ti.sin(theta[x, y]+PI/4), ti.cos(theta[x, y])])
        # img[x, y] = theta[x, y] * ti.Vector([1., 1., 1.]) / 2 / PI


def main():
    # theta.fill(PI/2)
    init()
    while window.running:
        for _ in range(10):
            update()
        draw()
        canvas.set_image(img)
        window.show()


if __name__ == "__main__":
    main()
