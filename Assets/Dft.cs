using System.Collections.Generic;
using System.Linq;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// DFT with the C# job system and the Burst compiler

public static class Dft
{
    #region Public method

    static public NativeArray<float> Transform(IEnumerable<float> input)
    {
        using (var input_array = TempJobMemory.New(input))
            return Transform(input_array);
    }

    static public NativeArray<float> Transform(NativeArray<float> input)
    {
        int width = input.Length;

        // DFT coefficients
        var e_coeffs = Enumerable.Range(0, width / 2 * width).
            Select(i => (k: i / width, n: i % width)).
            Select(I => 2 * math.PI / width * I.k * I.n);

        var e_coeffs_r = e_coeffs.Select(x => math.cos(x));
        var e_coeffs_i = e_coeffs.Select(x => math.sin(x));

        // Output buffer
        var output = new NativeArray<float>(width / 2, Allocator.Persistent);

        using (var coeffs_r = TempJobMemory.New(e_coeffs_r))
        using (var coeffs_i = TempJobMemory.New(e_coeffs_i))
        using (var temp     = TempJobMemory.New<float>(width))
        {
            // DFT job
            var job = new DftJob
            {
                input    = input   .Reinterpret<float4>(4),
                coeffs_r = coeffs_r.Reinterpret<float4>(4),
                coeffs_i = coeffs_i.Reinterpret<float4>(4),
                output   = output
            };

            // Dispatch and wait.
            job.Schedule(width / 2, 4).Complete();
        }

        return output;
    }

    #endregion

    #region DFT job

    [Unity.Burst.BurstCompile(CompileSynchronously = true)]
    struct DftJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float4> input;
        [ReadOnly] public NativeArray<float4> coeffs_r;
        [ReadOnly] public NativeArray<float4> coeffs_i;
        [WriteOnly] public NativeArray<float> output;

        public void Execute(int i)
        {
            var offs = i * input.Length;

            var rl = 0.0f;
            var im = 0.0f;

            for (var n = 0; n < input.Length; n++)
            {
                var x_n = input[n];
                rl += math.dot(x_n, coeffs_r[offs + n]);
                im -= math.dot(x_n, coeffs_i[offs + n]);
            }

            output[i] = math.sqrt(rl * rl + im * im) * 0.5f / input.Length;
        }
    }

    #endregion
}
