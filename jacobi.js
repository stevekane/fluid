/*
  linearly approximate the solution to the following system of
  equations:

  5x + 3y = -1
  15x + 14y = 8

  x + 3/5y = -1/5
  x = -.2 - .6y

  y + 15/14x = 8/14
  y = 8/14 - 15/14x
*/

const { abs } = Math
const x = y => -(1/5) - (3/5) * y
const y = x => (8/14) - (15/14) * x
const MAX_STEPS = 100
const EPSILON = .0001

function approximate ( fx, fy, x0, y0 ) {
  var px = x0
  var py = y0
  var cx = 0
  var cy = 0

  for ( var i = 0; i < MAX_STEPS; i++ ) {
    cx = x(py)
    cy = y(px)
    if ( abs(cx - px) <= EPSILON && abs(cy - py) <= EPSILON )
      return [ cx, cy ]
    px = cx
    py = cy
  }
  return [ cx, cy ]
}

const each = approximate(x, y, 0, 0)

console.log(5 * each[0] + 3 * each[1])
console.log(15 * each[0] + 14 * each[1])
