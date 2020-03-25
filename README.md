BurstFFT
========

**BurstFFT** is an FFT (Fast Fourier Transform) implementation in
high-performance C# with Unity's Burst compiler.

This repository contains the following three variants of Fourier transform
implementation.

- NaiveDFT: Unoptimized naive C# implementation of DFT
- BurstDFT: Vectorized/parallelized DFT implementation, optimized with Burst
- BurstFFT: Vectorized Cooley-Tukey FFT implementation, optimized with Burst

You can also enable parallelization on BurstFFT by disabling the `SINGLE_THREAD`
symbol in `BurstFft.cs`.

Results
-------

### Windows Desktop (Ryzen 7 3700X, 3.6GHz, 8 cores)

![table](https://i.imgur.com/yAr8hW6.png)

![graph](https://i.imgur.com/1K5a3mR.png)

### MacBook Pro 15 Late 2013 (Core i7, 2.3GHz, 4 cores)

![table](https://i.imgur.com/bVoMbdP.png)

![graph](https://i.imgur.com/CidTQKx.png)

Thoughts and Findings
---------------------

- It's quite easy to parallelize DFT with Unity's C# Job System. The more cores
  it has, the faster it runs.
- Although the parallelized DFT runs quite fast compared to the unoptimized one,
  it never beat the single-threaded FFT.
- The traditional Cooley-Tukey FFT is hard to parallelize in a performant way.
  The results above show that the 8-core BurstFFT runs slower than the 4-core
  one.
