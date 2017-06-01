const NDA = require('ndarray')
const ndafill = require('ndarray-fill')

const DIMENSION = 2
const ELEMENT_SIZE = 4
const a = NDA(new Float32Array(DIMENSION * DIMENSION * ELEMENT_SIZE), [ DIMENSION, DIMENSION, ELEMENT_SIZE ])

ndafill(a, function (i, j, k) {
  return k
})

console.log(a)
