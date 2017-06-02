const Regl = require('regl')
const Camera = require('regl-camera')
const FullScreenQuad = require('full-screen-quad')
const NDArray = require('ndarray')
const { cos, sin } = Math

function field ( regl, width, height, data ) {
  const tConfig = { 
    data, 
    width, 
    height, 
    wrap: 'repeat',
    mag: 'linear',
    min: 'linear',
    type: 'float',
    format: 'rgba'
  }

  return [
    regl.framebuffer({ 
      color: regl.texture(tConfig),
      depth: false,
      stencil: false
    }),
    regl.framebuffer({ 
      color: regl.texture(tConfig),
      depth: false,
      stencil: false
    })
  ]
}

function TextureBuffer ( w, h, d ) {
  return NDArray(new Float32Array(w * h * d), [ w, h, d ])
}

const regl = Regl({
  extensions: [ 
    'OES_texture_float', 
    'OES_texture_float_linear',
    'WEBGL_draw_buffers'
  ]
})

const SIZE = 256

const initialVelocity = TextureBuffer(SIZE, SIZE, 4)
const initialColors = TextureBuffer(SIZE, SIZE, 4)
const initialPressures = TextureBuffer(SIZE, SIZE, 4)

for ( var i = 0; i < initialVelocity.shape[0]; i++ ) {
  for ( var j = 0; j < initialVelocity.shape[1]; j++ ) {
    initialVelocity.set(i, j, 0, cos((i + j) / 4) + Math.random()) 
    initialVelocity.set(i, j, 1, sin((j - i) / 4) + Math.random())
    initialColors.set(i, j, 0, i / SIZE)
    initialColors.set(i, j, 1, j / SIZE)
  }
}

const fullScreenQuad = regl.buffer(FullScreenQuad(4))
const velocityBuffers = field(regl, SIZE, SIZE, initialVelocity)
const colorBuffers = field(regl, SIZE, SIZE, initialColors)
const pressureBuffers = field(regl, SIZE, SIZE, initialPressures)
const divergenceBuffer = regl.framebuffer({ 
  depth: false,
  stencil: false,
  color: regl.texture({ 
    type: 'float', 
    format: 'rgba',
    width: SIZE, 
    height: SIZE,
    wrap: 'repeat'
  })
})
const camera = Camera(regl, { 
  distance: 2, 
  theta: -Math.PI / 2,
  phi: -Math.PI / 4
})

const kernel_vert = `
  attribute vec4 position;

  varying vec2 coord;

  void main () {
    coord = position.xy * .5 + vec2(.5);
    gl_Position = position;
  }
`

const advect = regl({
  vert: kernel_vert,
  frag: `
    precision highp float;

    uniform sampler2D u;
    uniform sampler2D q;
    uniform vec2 rd;
    uniform float dT;

    varying vec2 coord;

    void main () {
      vec2 velocity = texture2D(u, coord).xy;
      vec2 position = coord - velocity * rd * dT;
      vec2 q_out = texture2D(q, position).xy;

      gl_FragColor = vec4(q_out, 0, 1);
    } 
  `,
  attributes: {
    position: fullScreenQuad
  },
  uniforms: {
    u: regl.prop('u'),
    q: regl.prop('q'),
    dT: regl.prop('dT'),
    rd: regl.prop('rd')
  },
  depth: { enable: false },
  count: 6,
  framebuffer: regl.prop('dest')
})

const divergence = regl({
  vert: kernel_vert,
  frag: `
    precision highp float;

    uniform sampler2D w;
    uniform vec2 rd;

    varying vec2 coord;

    void main () {
      float wL = texture2D(w, coord - rd.x).x;
      float wR = texture2D(w, coord + rd.x).x;
      float wB = texture2D(w, coord - rd.y).y;
      float wT = texture2D(w, coord + rd.y).y;
      float divergence = .5 * (wR - wL + wT - wB);

      gl_FragColor = vec4(divergence);
    }
  `,
  uniforms: {
    w: regl.prop('w'),
    rd: regl.prop('rd')
  },
  attributes: {
    position: fullScreenQuad 
  },
  count: 6,
  depth: { enable: false },
  framebuffer: regl.prop('dest')
})

const pressure_jacobi = regl({
  vert: kernel_vert,
  frag: `
    precision highp float;

    uniform sampler2D x;
    uniform sampler2D b;
    uniform vec2 rd;

    varying vec2 coord;

    void main () {
      float alpha = -1.;
      float rBeta = .25;
      vec2 dx = vec2(rd.x, 0);
      vec2 dy = vec2(0, rd.y);
      float xL = texture2D(x, coord - dx).x; 
      float xR = texture2D(x, coord + dx).x; 
      float xB = texture2D(x, coord - dy).x; 
      float xT = texture2D(x, coord + dy).x; 
      float bC = texture2D(b, coord).x;
      float x_out = (alpha * bC + xL + xR + xB + xT) * rBeta;

      gl_FragColor = vec4(x_out);
    } 
  `,
  uniforms: { 
    x: regl.prop('x'), 
    b: regl.prop('b'), 
    dT: regl.prop('dT'),
    rd: regl.prop('rd')
  },
  attributes: {
    position: fullScreenQuad 
  },
  count: 6,
  depth: { enable: false },
  framebuffer: regl.prop('dest')
})

const subtract_gradient = regl({
  vert: kernel_vert,
  frag: `
    precision highp float;

    uniform sampler2D pressure;
    uniform sampler2D velocity;
    uniform vec2 rd;

    varying vec2 coord;

    void main () {
      vec2 dx = vec2(rd.x, 0);
      vec2 dy = vec2(0, rd.y);
      float pL = texture2D(pressure, coord - dx).x;
      float pR = texture2D(pressure, coord + dx).x;
      float pB = texture2D(pressure, coord - dy).x;
      float pT = texture2D(pressure, coord + dy).x;
      vec2 v = texture2D(velocity, coord).xy;
      vec2 grad_p = .5 * vec2(pR - pL, pT - pB);

      gl_FragColor = vec4(v - grad_p, 0, 1);
    }
  `,
  uniforms: {
    pressure: regl.prop('p'),
    velocity: regl.prop('w'),
    rd: regl.prop('rd')
  },
  attributes: {
    position: fullScreenQuad
  },
  count: 6,
  depth: { enable: false },
  framebuffer: regl.prop('dest')
})

const render = regl({
  vert: `
    attribute vec4 position; 

    uniform mat4 view;
    uniform mat4 projection;

    varying vec2 v_uv;

    const vec2 offset = vec2(.5);

    void main () {
      v_uv = position.xy * .5 + offset;
      gl_Position = projection * view * position; 
    }
  `,
  frag: `
    precision highp float; 

    uniform sampler2D field;

    varying vec2 v_uv;

    void main () {
      gl_FragColor = texture2D(field, v_uv);
    }
  `,
  attributes: {
    position: fullScreenQuad 
  },
  uniforms: {
    field: regl.prop('field') 
  },
  count: 6
})

function pressure ( rd, b, ps, count ) {
  var i = 0
  var index = 0
  var src
  var dest

  for ( ; i < count; i++ ) {
    x = ps[index]
    index = (index + 1) % 2
    dest = ps[index]
    pressure_jacobi({ rd, b, x, dest })
  }
  return ps[index]
}

var index = 0
var p = null

document.body.style.backgroundColor = 'black'
regl.frame(function () {
  const color_src = colorBuffers[index]
  const u_src = velocityBuffers[index]
  const i = (index + 1) % 2
  const color_dest = colorBuffers[i]
  const u_dest = velocityBuffers[i]
  const rd = [ 1 / SIZE, 1 / SIZE ]
  const dT = 1

  index = i 
  advect({ dT, rd, u: u_src, q: u_src, dest: u_dest })
  // regl({ framebuffer: u_dest })(_ => console.log(regl.read()[0]))
  divergence({ rd, w: u_dest, dest: divergenceBuffer })
  p = pressure(rd, divergenceBuffer, pressureBuffers, 40)
  subtract_gradient({ rd, p, w: u_dest, dest: u_src })
  advect({ dT, rd, u: u_src, q: color_src, dest: color_dest })
  camera(_ => render({ field: color_dest }))
})
