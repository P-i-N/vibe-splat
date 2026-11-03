// The sum of one float over a whole workgroup, for workgroups of up to 1024
// invocations. Every invocation must call it (it contains barriers); the
// result is valid in every invocation.
#ifndef WORKGROUP_SUM_GLSL
#define WORKGROUP_SUM_GLSL

shared float workgroupSumPartials[32];

float workgroup_sum(float value) {
  const float partial = subgroupAdd(value);
  if (subgroupElect()) {
    workgroupSumPartials[gl_SubgroupID] = partial;
  }
  barrier();
  float total = 0.0;
  for (uint i = 0u; i < gl_NumSubgroups; ++i) {
    total += workgroupSumPartials[i];
  }
  barrier(); // So that a following call may overwrite the partials
  return total;
}

#endif
