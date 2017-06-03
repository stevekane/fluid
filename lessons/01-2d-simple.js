const Regl = require('regl')
const FullScreenQuad = require('full-screen-quad')
const NDArray = require('ndarray')
const { cos, sin } = Math
const extend = (a, b) => Object.assign(b, a)

const regl = Regl({
  extensions: [ 'OES_texture_float', 'OES_texture_float_linear' ]
})
const DT = 1 / 60
const fullScreenQuad = regl.buffer(FullScreenQuad(4))
const SIZE = 128
const BUFFER_COUNT = 2
const TEX_PROPS = {
  type: 'float', 
  format: 'rgba',
  mag: 'linear',
  min: 'linear',
  wrap: 'repeat',
  width: SIZE, 
  height: SIZE
}
const FB_PROPS = {
  depth: false,
  stencil: false
}
const KERNEL_PROPS = {
  attributes: { position: fullScreenQuad },
  count: 6,
  framebuffer: regl.prop('dst'),
  vert: `
    attribute vec4 position;

    varying vec2 uv;

    void main () {
      uv = position.xy * .5 + vec2(.5);
      gl_Position = position;
    }
  `
}

const field = FrameBufferList.bind(null, regl, TEX_PROPS, FB_PROPS, BUFFER_COUNT)
const kernel = p => regl(extend(KERNEL_PROPS, p))

const initialVelocity = TextureBuffer(SIZE, SIZE, 4)
const initialPressure = TextureBuffer(SIZE, SIZE, 4)

for ( var i = 0; i < initialVelocity.shape[0]; i++ ) {
  for ( var j = 0; j < initialVelocity.shape[1]; j++ ) {
  }
}

const velocityBuffers = field(initialVelocity)
const pressureBuffers = field(initialPressure)
const divergenceBuffer = regl.framebuffer(extend(FB_PROPS, { 
  color: regl.texture(TEX_PROPS)
}))

const advect = kernel({
  frag: `
    precision highp float;

    uniform sampler2D u;
    uniform sampler2D q;
    uniform vec2 rd;
    uniform float dT;

    varying vec2 uv;

    void main () {
      vec2 velocity = texture2D(u, uv).xy;
      vec2 position = uv - velocity * rd * dT;
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
})

const add_force = kernel({
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
    uniform vec2 rd;

    varying vec2 uv;

    void main() {
      float distance = 1.0 - min(length((uv - center) / rd), 1.0);

      gl_FragColor = vec4(force * distance, 0, 1);
    }
  `,
  uniforms: {
    force: regl.prop('force'),
    center: regl.prop('center'),
    rd: regl.prop('rd')
  }
})

const divergence = kernel({
  frag: `
    precision highp float;

    uniform sampler2D w;
    uniform vec2 rd;

    varying vec2 uv;

    void main () {
      vec2 dx = vec2(rd.x, 0);
      vec2 dy = vec2(0, rd.y);
      float wL = texture2D(w, uv - dx).x;
      float wR = texture2D(w, uv + dx).x;
      float wB = texture2D(w, uv - dy).y;
      float wT = texture2D(w, uv + dy).y;
      float divergence = (wR - wL + wT - wB) * .5;

      gl_FragColor = vec4(divergence);
    }
  `,
  uniforms: {
    w: regl.prop('w'),
    rd: regl.prop('rd')
  }
})

const jacobi = kernel({
  frag: `
    precision highp float;

    uniform sampler2D x;
    uniform sampler2D b;
    uniform vec2 rd;
    uniform float alpha;
    uniform float rBeta;

    varying vec2 uv;

    void main () {
      vec2 dx = vec2(rd.x, 0);
      vec2 dy = vec2(0, rd.y);
      float xL = texture2D(x, uv - dx).x; 
      float xR = texture2D(x, uv + dx).x; 
      float xB = texture2D(x, uv - dy).x; 
      float xT = texture2D(x, uv + dy).x; 
      float bC = texture2D(b, uv).x;
      float x_out = (alpha * bC + xL + xR + xB + xT) * rBeta;

      gl_FragColor = vec4(x_out);
    } 
  `,
  uniforms: { 
    x: regl.prop('x'), 
    b: regl.prop('b'), 
    alpha: regl.prop('alpha'),
    rBeta: regl.prop('rBeta'),
    rd: regl.prop('rd')
  }
})

const subtract_gradient = kernel({
  frag: `
    precision highp float;

    uniform sampler2D pressure;
    uniform sampler2D velocity;
    uniform vec2 rd;

    varying vec2 uv;

    void main () {
      vec2 dx = vec2(rd.x, 0);
      vec2 dy = vec2(0, rd.y);
      float pL = texture2D(pressure, uv - dx).x;
      float pR = texture2D(pressure, uv + dx).x;
      float pB = texture2D(pressure, uv - dy).x;
      float pT = texture2D(pressure, uv + dy).x;
      vec2 v = texture2D(velocity, uv).xy;
      vec2 grad_p = .5 * vec2(pR - pL, pT - pB);

      gl_FragColor = vec4(v - grad_p, 0, 1);
    }
  `,
  uniforms: {
    pressure: regl.prop('p'),
    velocity: regl.prop('w'),
    rd: regl.prop('rd')
  }
})

const render = regl({
  vert: `
    attribute vec4 position; 

    varying vec2 uv;

    const vec2 offset = vec2(.5);

    void main () {
      uv = position.xy * .5 + offset;
      gl_Position = position; 
    }
  `,
  frag: `
    precision highp float; 

    uniform sampler2D u;
    uniform sampler2D p;

    varying vec2 uv;

    void main () {
      vec2 velocity = texture2D(u, uv).xy * 1.5 + .5;
      float pressure = texture2D(p, uv).x;

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

function FrameBufferList ( regl, txprops, fbprops, count, data ) {
  const l = []

  for ( var i = 0, fb, color; i < count; i++ ) {
    color = regl.texture(extend(txprops, { data }))
    fb = regl.framebuffer(extend(fbprops, { color }))
    l.push(fb)
  }
  return l
}

function TextureBuffer ( w, h, d ) {
  return NDArray(new Float32Array(w * h * d), [ w, h, d ])
}

function pressure ( { rd, b, alpha, rBeta }, ps, count ) {
  var i = 0
  var index = 0
  var src
  var dst

  for ( ; i < count; i++ ) {
    x = ps[index]
    index = (index + 1) % 2
    dst = ps[index]
    jacobi({ rd, alpha, rBeta, b, x, dst })
  }
  return ps[index]
}

var p = null
var c = [ 0, 0 ]

regl.frame(function ({ tick }) {
  const ITERATION_COUNT = 60
  const u_src = velocityBuffers[0]
  const u_dst = velocityBuffers[1]
  const dT = DT
  const rd = [ 
    1 / SIZE, 
    1 / SIZE 
  ]
  const center = [ 
    sin(tick / 10) * cos(tick / 100) * .5 + .5, 
    cos(tick / 10) * sin(tick / 100) * .5 + .5
  ]
  const force = [ 
    100000 * (center[0] - c[0]), 
    100000 * (center[1] - c[1]) 
  ]

  c[0] = center[0]
  c[1] = center[1]
  index = i 
  advect({ dT, rd, u: u_src, q: u_src, dst: u_dst })
  add_force({ rd, force, center, dst: u_dst })
  divergence({ rd, w: u_dst, dst: divergenceBuffer })
  p = pressure({ rd, alpha: -1, rBeta: .25, b: divergenceBuffer }, pressureBuffers, ITERATION_COUNT)
  subtract_gradient({ rd, p, w: u_dst, dst: u_src })
  render({ p, u: u_src })
})
