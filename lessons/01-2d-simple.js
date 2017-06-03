const Regl = require('regl')
const FullScreenQuad = require('full-screen-quad')
const NDArray = require('ndarray')
const { cos, sin } = Math
const extend = (a, b) => Object.assign(b, a)

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

const SIZE = 128

const initialVelocity = TextureBuffer(SIZE, SIZE, 4)
const initialColors = TextureBuffer(SIZE, SIZE, 4)
const initialPressures = TextureBuffer(SIZE, SIZE, 4)

for ( var i = 0; i < initialVelocity.shape[0]; i++ ) {
  for ( var j = 0; j < initialVelocity.shape[1]; j++ ) {
    initialVelocity.set(i, j, 0, Math.random() * 2 - 1)
    initialVelocity.set(i, j, 1, sin(Math.PI * i / SIZE))
    initialColors.set(i, j, 0, i / SIZE)
    initialColors.set(i, j, 1, j / SIZE)
  }
}

const fullScreenQuad = regl.buffer(FullScreenQuad(4))
const velocityBuffers = field(regl, SIZE, SIZE, initialVelocity)
const pressureBuffers = field(regl, SIZE, SIZE, initialPressures)
const colorBuffers = field(regl, SIZE, SIZE, initialColors)
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

const kernel = {
  attributes: { position: fullScreenQuad },
  count: 6,
  framebuffer: regl.prop('dst'),
  vert: `
    attribute vec4 position;

    varying vec2 coord;

    void main () {
      coord = position.xy * .5 + vec2(.5);
      gl_Position = position;
    }
  `
}
const advect = regl(extend(kernel, {
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
  uniforms: {
    u: regl.prop('u'),
    q: regl.prop('q'),
    dT: regl.prop('dT'),
    rd: regl.prop('rd')
  },
}))

const add_force = regl(extend(kernel, {
  blend: {
    enable: true,
    func: {
      src: 'src alpha',
      dst: 'one'
    }
  },
  frag: `
    precision highp float;

    uniform vec2 force;
    uniform vec2 center;
    uniform vec2 scale;

    varying vec2 coord;

    void main() {
      float distance = 1.0 - min(length((coord - center) / scale), 1.0);

      gl_FragColor = vec4(force * distance, 0, 1);
    }
  `,
  uniforms: {
    force: regl.prop('force'),
    center: regl.prop('center'),
    scale: regl.prop('scale')
  }
}))

const divergence = regl(extend(kernel, {
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
  }
}))

const pressure_jacobi = regl(extend(kernel, {
  frag: `
    precision highp float;

    uniform sampler2D x;
    uniform sampler2D b;
    uniform vec2 rd;

    varying vec2 coord;

    void main () {
      vec2 dx = vec2(rd.x, 0);
      vec2 dy = vec2(0, rd.y);
      float alpha = -1.;
      float rBeta = .25;
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
    rd: regl.prop('rd')
  }
}))

const subtract_gradient = regl(extend(kernel, {
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
  }
}))

const render = regl({
  vert: `
    attribute vec4 position; 

    varying vec2 v_uv;

    const vec2 offset = vec2(.5);

    void main () {
      v_uv = position.xy * .5 + offset;
      gl_Position = position; 
    }
  `,
  frag: `
    precision highp float; 

    uniform sampler2D u;
    uniform sampler2D p;

    varying vec2 v_uv;

    void main () {
      vec2 velocity = abs(texture2D(u, v_uv).xy);
      float pressure = abs(texture2D(p, v_uv).x);

      gl_FragColor = vec4(pressure, velocity, 1);
    }
  `,
  attributes: {
    position: fullScreenQuad 
  },
  uniforms: {
    u: regl.prop('u'),
    p: regl.prop('p') 
  },
  count: 6
})

function pressure ( rd, b, ps, count ) {
  var i = 0
  var index = 0
  var src
  var dst

  for ( ; i < count; i++ ) {
    x = ps[index]
    index = (index + 1) % 2
    dst = ps[index]
    pressure_jacobi({ rd, b, x, dst })
  }
  return ps[index]
}

var index = 0
var p = null

regl.frame(function () {
  const ITERATION_COUNT = 60
  const u_src = velocityBuffers[index]
  const color_src = colorBuffers[index]
  const i = (index + 1) % 2
  const u_dst = velocityBuffers[i]
  const color_dst = colorBuffers[i]
  const rd = [ 1 / SIZE, 1 / SIZE ]
  const dT = 1 / 60

  index = i 
  advect({ dT, rd, u: u_src, q: color_src, dst: color_dst })
  advect({ dT, rd, u: u_src, q: u_src, dst: u_dst })
  divergence({ rd, w: u_dst, dst: divergenceBuffer })
  p = pressure(rd, divergenceBuffer, pressureBuffers, ITERATION_COUNT)
  subtract_gradient({ rd, p, w: u_dst, dst: u_src })
  render({ p: color_dst, u: u_src })
})
