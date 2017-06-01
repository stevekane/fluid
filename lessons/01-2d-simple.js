const Regl = require('regl')
const Camera = require('regl-camera')
const BigTriangle = require('big-triangle')
const FullScreenQuad = require('full-screen-quad')
const NDArray = require('ndarray')
const fill = require('ndarray-fill')
const { random, sin, cos } = Math
const rand = (min, max) => min + random() * (max - min)

function field ( regl, width, height, data ) {
  const tConfig = { 
    data, 
    width, 
    height, 
    wrap: 'repeat',
    // wrap: 'clamp', 
    mag: 'linear',
    min: 'linear',
    type: 'float'
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
const initialColors = fill(TextureBuffer(SIZE, SIZE, 4), (i, j, k) => i / SIZE)
const initialPressures = TextureBuffer(SIZE, SIZE, 4)

for ( var i = 0; i < initialVelocity.shape[0]; i++ ) {
  for ( var j = 0; j < initialVelocity.shape[1]; j++ ) {
    initialVelocity.set(i, j, 0, rand(-i / SIZE, 1)) 
    initialVelocity.set(i, j, 1, rand(-j / SIZE, 1)) 
  }
}

const bigTriangle = regl.buffer(BigTriangle(4))
const fullScreenQuad = regl.buffer(FullScreenQuad(4))
const velocityBuffers = field(regl, SIZE, SIZE, initialVelocity)
const colorBuffers = field(regl, SIZE, SIZE, initialColors)
const pressureBuffers = field(regl, SIZE, SIZE, initialPressures)
const divergenceBuffer = regl.framebuffer({ 
  depth: false,
  stencil: false,
  color: regl.texture({ 
    type: 'float', 
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

const advect = regl({
  vert: `
    attribute vec4 position;

    void main () {
      gl_Position = position;
    }
  `,
  frag: `
    precision highp float;

    uniform sampler2D u;
    uniform sampler2D q;
    uniform float dT;
    uniform float rViewportWidth;
    uniform float rViewportHeight;

    void main () {
       vec2 rViewport = vec2(rViewportWidth, rViewportHeight);
       vec2 p = gl_FragCoord.xy * rViewport;
       vec2 velocity = texture2D(u, p).xy;
       vec2 position = p - rViewport * velocity * dT;
       vec2 q_out = texture2D(q, position).xy;

      gl_FragColor = vec4(q_out, 0, 1);
    } 
  `,
  attributes: {
    position: bigTriangle
  },
  uniforms: {
    u: regl.prop('u'),
    q: regl.prop('q'),
    dT: regl.prop('dT'),
    rViewportWidth: ({ viewportWidth }) => 1 / viewportWidth,
    rViewportHeight: ({ viewportHeight }) => 1 / viewportHeight
  },
  blend: {
    enable: true
  },
  count: 3,
  framebuffer: regl.prop('dest')
})

const divergence = regl({
  vert: `
    attribute vec4 position;

    void main () {
      gl_Position = position;
    }
  `,
  frag: `
    precision highp float;

    uniform sampler2D w;
    uniform float rViewportWidth;
    uniform float rViewportHeight;

    void main () {
      vec2 c = vec2(rViewportWidth, rViewportHeight);
      vec2 p = gl_FragCoord.xy * c;
      vec2 dx = vec2(rViewportWidth, 0);
      vec2 dy = vec2(0, rViewportHeight);
      vec2 wL = texture2D(w, p - dx).xy;
      vec2 wR = texture2D(w, p + dx).xy;
      vec2 wB = texture2D(w, p - dy).xy;
      vec2 wT = texture2D(w, p + dy).xy;
      float dudx = .5 * rViewportWidth * (wR.x - wL.x);
      float dudy = .5 * rViewportHeight * (wT.y - wB.y);

      gl_FragColor = vec4(dudx + dudy, 0, 0, 1);
    }
  `,
  uniforms: {
    w: regl.prop('w'),
    rViewportWidth: ({ viewportWidth }) => 1 / viewportWidth,
    rViewportHeight: ({ viewportHeight }) => 1 / viewportHeight
  },
  attributes: {
    position: bigTriangle
  },
  count: 3,
  framebuffer: regl.prop('dest')
})

const pressure_jacobi = regl({
  vert: `
    attribute vec4 position;

    void main () {
      gl_Position = position;
    }
  `,
  frag: `
    precision highp float;

    uniform sampler2D x;
    uniform sampler2D b;
    uniform float rViewportWidth;
    uniform float rViewportHeight;

    const float rBeta = .25;

    void main () {
      vec2 c = vec2(rViewportWidth, rViewportHeight);
      vec2 p = gl_FragCoord.xy * c;
      vec2 dx = vec2(rViewportWidth, 0);
      vec2 dy = vec2(0, rViewportHeight);
      vec2 xC = texture2D(x, p).xy;
      vec2 xL = texture2D(x, p - dx).xy; 
      vec2 xR = texture2D(x, p + dx).xy; 
      vec2 xB = texture2D(x, p - dy).xy; 
      vec2 xT = texture2D(x, p + dy).xy; 
      vec2 bC = texture2D(b, p).xy;
      // vec2 alpha = -(xC * xC);
      float alpha = -1.;
      vec2 x_out = (xL + xR + xB + xT + alpha * bC) * rBeta;

      gl_FragColor = vec4(x_out, 0, 1);
    } 
  `,
  uniforms: { 
    x: regl.prop('x'), 
    b: regl.prop('b'), 
    alpha: regl.prop('alpha'), 
    rBeta: regl.prop('rBeta'), 
    dT: regl.prop('dT'),
    rViewportWidth: ({ viewportWidth }) => 1 / viewportWidth,
    rViewportHeight: ({ viewportHeight }) => 1 / viewportHeight
  },
  attributes: {
    position: bigTriangle
  },
  count: 3,
  framebuffer: regl.prop('dest')
})

const subtract_gradient = regl({
  vert: `
    attribute vec4 position;

    void main () {
      gl_Position = position;
    }
  `,
  frag: `
    precision highp float;

    uniform sampler2D pressure;
    uniform sampler2D velocity;
    uniform float rViewportWidth;
    uniform float rViewportHeight;

    void main () {
      vec2 c = vec2(rViewportWidth, rViewportHeight);
      vec2 p = gl_FragCoord.xy * c;
      vec2 dx = vec2(rViewportWidth, 0);
      vec2 dy = vec2(0, rViewportHeight);
      vec2 u = texture2D(velocity, p).xy;
      float pL = texture2D(pressure, p - dx).x;
      float pR = texture2D(pressure, p + dx).x;
      float pB = texture2D(pressure, p - dy).x;
      float pT = texture2D(pressure, p + dy).x;
      vec2 grad_p = vec2(.5 * rViewportWidth * (pR - pL), .5 * rViewportHeight * (pT - pB));

      gl_FragColor = vec4(u - grad_p, 0, 1);
    }
  `,
  uniforms: {
    pressure: regl.prop('p'),
    velocity: regl.prop('w'),
    rViewportWidth: ({ viewportWidth }) => 1 / viewportWidth,
    rViewportHeight: ({ viewportHeight }) => 1 / viewportHeight
  },
  attributes: {
    position: bigTriangle
  },
  count: 3,
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

function pressure ( b, ps, count ) {
  var i = 0
  var index = 0
  var src
  var dest

  // TODO: skeptical we need to clear every frame.  use last frames value for guess?
  regl({ framebuffer: ps[0] }, _ => regl.clear({ 
    color: [ 0, 0, 0, 0],
    depth: false
  }))
  for ( ; i < count; i++ ) {
    src = ps[index]
    index = (index + 1) % 2
    dest = ps[index]
    pressure_jacobi({ 
      x: src,
      b, 
      dest: dest,
    })
  }
  return ps[index]
}

var index = 0
var then = 0
var now = 0
var dT = 0

var b = null

window.getPixels = function () { console.log(b) }

document.body.style.backgroundColor = 'black'
regl.frame(function () {
  const color_src = colorBuffers[index]
  const u_src = velocityBuffers[index]
  const i = (index + 1) % 2
  const color_dest = colorBuffers[i]
  const u_dest = velocityBuffers[i]

  index = i 
  then = now
  now = performance.now()
  dT = 1
  advect({ dT, u: u_src, q: color_src, dest: color_dest })
  advect({ dT, u: u_src, q: u_src, dest: u_dest })
  divergence({ w: u_dest, dest: divergenceBuffer }, _ => {
    b = regl.read()
  })
  const p = pressure(divergenceBuffer, pressureBuffers, 10)
  subtract_gradient({ p: p, w: u_dest, dest: u_src })
  camera(_ => render({ field: color_dest }))
})
