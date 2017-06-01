Navier stokes equations

du/dt = u-advection - pressure + diffusion + forcefield
dv/dt = v-advection - pressure + diffusion + forcefield
0 = divergence(u)
  = sum of pressure from four adjacent cells - 4 * cell-pressure / d^2

velocity field w can be decomposed to 
  w = u + gradient(p)
  therefore
  u = w - gradient(p)

divergence(w) = divergence(u + gradient(p)) 
              = divergence(u) + gradient(p)
from above, we know that divergence(u) = 0, therefore
divergence(w) = laplacian(p)

1. calculate divergent velocity w
2. use w to calculate p
3. use p to calculate u, the new divergence-free velocity

introduce P which is an operator that projects w onto its divergence-free component u

P(w) = P(u) + P(gradient(p))

P(w) is the projection of w onto u
P(u) is the projection of u onto itself ( u )
therefore, P(w) = P(u) = u
0 = P(gradient(p))

Apply P to both sides of the navier stokes equation for du/dt and you observe two things:
1. since u is divergence free, P(du/dt) = du/dt
2. we know P(gradient(p)) = 0.  That causes the entire pressure term in the equation to 
   be set to 0 and "removed" from the equation leaving only

du/dt = P(advection + diffusion + forcefield)

summing advection diffusion and forcefield gives you w, the divergent velocity
compute p by solving:  divergence(w) = laplacian(p) for p
u = w - divergence(p) to get the divergence-free velocity u

Define an operator S as the composition of the operators:
  Advection (A)
  Diffusion (D)
  Force (F)
  Projection (P)

S t u = (P t . F t . D t . A t) u

ADVECTION
  the effect of advection is to carry a value "along" the velocity field.  To calculate
  advection for property q:
    q(x, t + dT) = q(x - u(x, t)dT, t)

  The intuition here is that we're starting from a cell and using its current velocity and a small
  time interval to project "backward in time" to where this cell used to be.  This will land somewhere
  between cell centerpoints which means we can use bilinear ( or trilinear in 3d ) to compute 
  this value and set it equal to the current cell's new value.  This is like saying "to find out what's
  here now, look backwards in time a little bit in the correct direction".

DIFFUSION ( VISCOCITY )
  Diffusion is described by:   
    du/dt = v * laplacian(u)    
  We can write this in a discrete form as follows
    d(x, t + dT) = u(x, t) + v * dT * discrete_laplacian(u(x, t))
  As with advection, we would prefer an implicit form of the equation above which is stable
  for large values of v and dT.
    u(x,t) = (I - v * dT * discrete_laplacian) * u(x, t + dT) where I is an identity matrix
  We are left here with two poisson equations that we need to solve:
    1. laplacian(p) = divergence(w)
    2. solve the equation for viscocity

  Both equations can be discretized and re-written to the following format:
    xn = (sum_four_nearest_cells(x) + a * b) / B
  Each equation has different values for a, b and B.
    for diffusion
      alpha = x^2 / t
      rBeta = 1 / (4 + (x^2 / t))
      x = velocity
      b = velocity
    for pressure
      alpha = -(x^2)
      rBeta = 1/4
      x = pressure
      b = divergence

FORCE
  Forces are applied to the field as they would be in traditional physics simulations.  Gravity and
  other uni-directional force-fields would be applied everywhere to the velocity field.  Local impulses
  may be applied to the velocity field according to your preference.
