# Faster training iterations

Date: 2026-11-02
Status: done

## Intent

`splat train` ran at 9.4 iterations/s on the 194-photo test set at 1600x1063
(103 ms of GPU time per iteration), far below what the GPU can do. The user
asked for the iterations to be as fast as possible.

## What was wrong

The compositing, backward and loss shaders ran one thread per 16x16 tile or
per image row, looping over pixels in global memory: a few thousand threads
on a GPU that wants a hundred thousand, and a read-modify-write of the image
per pixel per splat. Composite took 22 ms, backward 69 ms and the loss 11 ms.

## Changes, in order of effect

1. **One thread per pixel** in the forward and backward passes, one workgroup
   per tile. The tile's splats are staged through shared memory 256 at a time
   and every thread blends its own pixel in registers, back to front as
   before, so the forward output is bit-identical to the CPU reference.
   Backward keeps the composite behind the current splat in registers too.
2. **Transposed subgroup reduction.** Each splat's nine gradient sums over a
   tile are reduced per subgroup with one butterfly that halves the number of
   values per lane at each step (16 shuffles for 32 lanes instead of 45), and
   up to nine lanes add the totals with one atomic each. Backward went from
   13.7 ms to 4.9 ms with this alone. The atomics themselves cost about 1 ms.
3. **Fused, tiled SSIM.** The four separable passes (two per direction) became
   two tile shaders with the horizontal pass in shared memory, removing the
   80-byte-per-pixel intermediate buffer. Loss: 4-5 ms to about 2 ms.
4. **Fewer submissions and copies.** Adam is recorded in the same command
   buffer as the forward and backward passes. The target image is copied once
   into a device-local buffer rather than read from host memory by every loss
   pass, and the next view's copy into staging memory runs on a persistent
   worker thread while the GPU works on the current view.
5. **Densification keeps the optimiser state on the GPU.** A gather shader
   builds the survivors' Adam moments from the old buffers through a
   new-to-old index table, instead of downloading and re-uploading them.

## Results (194 views, 1600x1063, RTX 4070 Laptop)

| Run | Before | After |
| --- | --- | --- |
| 600 iterations, 32k-45k Gaussians | 9.4 it/s, 103.5 ms GPU | 80-88 it/s, 9.2-10 ms GPU |
| GPU stages | composite 22, loss 11, backward 69 ms | composite 1.7, loss 1.8, backward 4.1 ms |
| 3000 iterations, up to 400k Gaussians | not measured | 59-61 it/s, 13.7-13.9 ms GPU at 400k |

Quality is unchanged: the same loss curve and PSNR (18.7 dB after 600
iterations) to within the summation order of the gradients, and the tests
comparing GPU gradients to the CPU reference pass with the same margins.

## Tried and dropped

- A batch of 128 splats per shared-memory load instead of 256: slower.

## What is left

About 1.5-2 ms per iteration is CPU-side: two submissions with fence waits
(the tile-entry count is read back between them to size the sort), command
recording, and the per-iteration bookkeeping. Pipelining the next iteration's
preprocessing behind the current one, or dispatching the entry sort
indirectly from a GPU-side count, would recover most of it. Densification
still moves every buffer through the CPU; at 400k Gaussians each round costs
a few hundred milliseconds, every 100 iterations.
