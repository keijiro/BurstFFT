using System.Linq;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Naive DFT vectorized/parallelized with the Burst compiler

public sealed class DftBuffer : System.IDisposable
{
    #region Public properties and methods

    public NativeArray<float> Spectrum => _buffer;

    public DftBuffer(int width)
    {
        // DFT coefficients
        var coeffs = Enumerable.Range(0, width / 2 * width).
            Select(i => (k: i / width, n: i % width)).
            Select(I => 2 * math.PI / width * I.k * I.n);

        var coeffs_r = coeffs.Select(x => math.cos(x));
        var coeffs_i = coeffs.Select(x => math.sin(x));

        _coeffs_r = PersistentMemory.New<float>(coeffs_r);
        _coeffs_i = PersistentMemory.New<float>(coeffs_i);

        // Output buffer
        _buffer = PersistentMemory.New<float>(width / 2);
    }

    public void Dispose()
    {
        if (_coeffs_r.IsCreated) _coeffs_r.Dispose();
        if (_coeffs_i.IsCreated) _coeffs_i.Dispose();
        if (_buffer.IsCreated) _buffer.Dispose();
    }

    public void Transform(NativeArray<float> input)
    {
        // Dispatch and complete the DFT jobs with the input.
        new DftJob 
          { I  = input.Reinterpret<float4>(4),
            Cr = _coeffs_r.Reinterpret<float4>(4),
            Ci = _coeffs_i.Reinterpret<float4>(4),
            O  = _buffer }
          .Schedule(input.Length / 2, 16).Complete();
    }

    #endregion

    #region Private members

    NativeArray<float> _coeffs_r;
    NativeArray<float> _coeffs_i;
    NativeArray<float> _buffer;

    #endregion

    #region DFT job

    [Unity.Burst.BurstCompile(CompileSynchronously = true)]
    struct DftJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float4> I;
        [ReadOnly] public NativeArray<float4> Cr;
        [ReadOnly] public NativeArray<float4> Ci;
        [WriteOnly] public NativeArray<float> O;

        public void Execute(int i)
        {
            var N = I.Length;
            var offs = i * N;

            var rl = 0.0f;
            var im = 0.0f;

            for (var n = 0; n < N; n++)
            {
                var x = I[n];
                rl += math.dot(x, Cr[offs + n]);
                im -= math.dot(x, Ci[offs + n]);
            }

            O[i] = math.sqrt(rl * rl + im * im) * 0.5f / N;
        }
    }

    #endregion
}
