// Workgroup-wide exclusive prefix sum for 256-thread workgroups.
// Uses subgroup operations and works for any subgroup size, including
// partially filled subgroups (varying subgroup size in SPIR-V 1.6).
// Requires: GL_KHR_shader_subgroup_arithmetic, GL_KHR_shader_subgroup_basic.

#ifndef WORKGROUP_SCAN_GLSL
#define WORKGROUP_SCAN_GLSL

shared uint s_scanSubgroupTotals[128];
shared uint s_scanTotal;

// Returns the sum of `value` over all lower invocations of the workgroup and
// stores the workgroup total in `total`. Every invocation must call it.
uint workgroup_exclusive_add(uint value, out uint total) {
  const uint inclusive = subgroupInclusiveAdd(value);
  const uint subgroupTotal = subgroupAdd(value);
  if (subgroupElect()) {
    s_scanSubgroupTotals[gl_SubgroupID] = subgroupTotal;
  }
  barrier();
  if (gl_LocalInvocationIndex == 0) {
    uint running = 0;
    for (uint i = 0; i < gl_NumSubgroups; ++i) {
      const uint t = s_scanSubgroupTotals[i];
      s_scanSubgroupTotals[i] = running;
      running += t;
    }
    s_scanTotal = running;
  }
  barrier();
  const uint result = s_scanSubgroupTotals[gl_SubgroupID] + inclusive - value;
  total = s_scanTotal;
  barrier(); // Shared scratch may be reused by the next call.
  return result;
}

#endif
