// What train_gradient.comp and train_sh_step.comp share: how the chain rule's
// gradients become steps. Both take the same push constants, as one pass
// follows the other. Needs bf16.glsl and train_types.glsl.
#ifndef TRAIN_STEP_GLSL
#define TRAIN_STEP_GLSL

const uint MODE_STATISTICS = 0u; // Densification statistics only
const uint MODE_GRADIENTS = 1u;  // Store the gradients
const uint MODE_STEP = 2u;       // Adam's step, or second order with FLAG_SECOND_ORDER
const uint MODE_CURVATURE = 3u;  // Add the squared gradients to the curvature
const uint FLAG_STATISTICS = 1u; // Add to the densification statistics
const uint FLAG_POSE = 2u;       // Sum how the loss changes with the camera
const uint FLAG_SELECTIVE = 4u;  // Adam leaves parameters without a gradient alone
const uint FLAG_SECOND_ORDER = 8u; // All but rotation step by curvature, within pc.trustRegion

layout(push_constant) uniform Push {
  // 3DGS-MCMC's penalties, reg * mean(sigmoid(opacity)) and reg * mean(exp(scale))
  // over the Gaussians, already divided by their count.
  float opacityReg;
  float scaleReg;
  uint mipFilter; // As in train_preprocess.comp
  float shDropout;
  uint shRetained;
  uint shSeed;
  uint mode;
  uint flags;
  float rates[5];      // Learning rates: position, scale, rotation, opacity, SH
  float restScale;     // SH above degree 0, relative to rates[4]
  float beta1;
  float beta2;
  float epsilon;
  float correction1;   // 1 - beta1^t
  float correction2;   // 1 - beta2^t
  float trustRegion;   // Epsilon of the squared Hellinger distance
  float curvatureScale; // 1 / (1 - curvatureBeta^k)
  float curvatureBeta;
  float perValue;      // MODE_CURVATURE: 1 / values in the image
  uint seed;           // Of this pass's random rounding
  uint shFloats;       // Of GaussianCloud's sh per Gaussian: the stride of its state and gradients
} pc;

const float SH_C0 = 0.28209479177387814;

// Adam (Kingma and Ba 2015), or with radius at or above 0 a step of 3DGS^2-TR:
// the averaged gradient over the curvature, clipped to the radius. The first
// moment moves by a tenth a step, far more than its rounding; the second by a
// thousandth, under half its rounding, so it is rounded at random, which keeps
// its average exact. The step uses both before rounding.
void step_parameter(inout float value, inout uint state, float g, float rate, float radius, uint index) {
  const float m = pc.beta1 * bf16_load(state, 0u) + (1.0 - pc.beta1) * g;
  if (radius >= 0.0) {
    const float change = (m / pc.correction1) / max(bf16_load(state, 1u) * pc.curvatureScale, 1e-30);
    value -= clamp(change, -radius, radius);
    state = bf16_store(state, 0u, bf16_round(m));
  } else {
    const float v = pc.beta2 * bf16_load(state, 1u) + (1.0 - pc.beta2) * g * g;
    value -= rate * (m / pc.correction1) / (sqrt(v / pc.correction2) + pc.epsilon);
    state = bf16_store(bf16_store(state, 0u, bf16_round(m)), 1u, bf16_stochastic(v, index, pc.seed));
  }
}

// One estimate of the diagonal of J^T J per value in the image, into its running average.
uint with_curvature(uint state, float g, uint index) {
  const float c = pc.curvatureBeta * bf16_load(state, 1u) + (1.0 - pc.curvatureBeta) * g * g * pc.perValue;
  return bf16_store(state, 1u, bf16_stochastic(c, index, pc.seed));
}

// Half floats, as the coefficients above degree 0 are stored, rounded at
// random like the second moment: a step is often under half their rounding.
uint half_stochastic(float x, uint index, uint seed) {
  const uint nearest = packHalf2x16(vec2(x, 0.0));
  const float rounded = unpackHalf2x16(nearest).x;
  if (rounded == x || isinf(rounded)) {
    return nearest;
  }
  // The other neighbour: one step away from zero if x lies beyond, else towards it.
  const uint other = abs(x) > abs(rounded) ? nearest + 1u : nearest - 1u;
  const float gap = unpackHalf2x16(other).x - rounded;
  return train_uniform(index, seed) < (x - rounded) / gap ? other : nearest;
}

#endif
