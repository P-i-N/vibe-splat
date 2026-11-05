// The trainer's SH, in the layout of trainer_sh_word() in splat_types.glsl.
// Include after declaring the buffer as `uint sh[]`.
#ifndef TRAIN_SH_GLSL
#define TRAIN_SH_GLSL

// Coefficient k of the Gaussian whose words start at `base`.
vec3 sh_coefficient(uint base, uint k) {
  vec3 c;
  for (uint channel = 0u; channel < 3u; ++channel) {
    const uint j = k * 3u + channel;
    c[channel] = trainer_sh_value(sh[base + trainer_sh_word(j)], j);
  }
  return c;
}

#endif
