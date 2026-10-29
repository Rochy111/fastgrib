# GPU and threads

## The GPU

```python
fields = f.read(':TMP:', gpu=True)        # cupy arrays, left on the card
```

`gpu=True` returns cupy arrays that stay on the GPU. That's by far the
fastest path when your next step runs on the GPU too, such as drawing with
fastplot, which takes cupy arrays as they are.

For numpy results, fastgrib chooses the device by measurement:

- A fresh result array comes from the CPU, because the GPU would have to
  page-lock new memory first, which makes it two to three times slower.
- Big fields read into an array you reuse (`read(..., out=array)`) come from
  the GPU. GFS 0.25° takes 0.14 ms a field there against 0.21 ms on the CPU.
- `fields()` uses the CPU unless the GPU is already running and the stream
  is long. All 769 of NBM's fields take 0.37 ms a field on the GPU against
  0.55 ms on the CPU.

To force a device:

```python
fastgrib.set_device('gpu')      # or 'cpu', or 'auto' (the default)
```

or set `FASTGRIB_DEVICE`. Forcing the GPU leaves the CPU almost idle while
decoding; on NBM it uses a twelfth of the CPU time.

You need an NVIDIA GPU with CUDA 12 and cupy (the `gpu` or `all` extra).
GPU work is serialised per card, so many threads can share one safely.

## Threads

The CPU decoder uses one thread per logical CPU. Fields too big for one
core are split across all of them in a single pass.

```python
fastgrib.set_threads(8)        # or FASTGRIB_THREADS=8
fastgrib.set_threads(None)     # back to all of them
```

Fewer threads leave the machine free for other work. When memory bandwidth
is the limit, as with large simple- or complex-packed fields read through
`fields()`, they cost very little time: GFS 0.25° on 8 of 24 threads is 4%
slower for a fifth less CPU time.

## Processes

fastgrib already uses every core within one process, so one process is
usually enough. If you do use a pool of worker processes, keep the pool for
the whole job and warm each worker as it starts:

```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(4, initializer=fastgrib.warm) as pool:
    ...
```

Worker processes forked from a parent that has already decoded (the
default with `multiprocessing` on Linux) start their own threads and HTTP
connections.

## Memory

Memory stays bounded:

- `fields()` decodes into a fixed ring of buffers.
- Each decoding thread keeps at most 32 MB of JPEG2000 scratch space.
- Page-locked memory is limited to the file being read on the GPU (at most
  2 GB and an eighth of RAM) plus two arrays you reuse with `out=`.

`f.close()` releases a file's memory, and `fastgrib.release()` releases
everything fastgrib holds.
