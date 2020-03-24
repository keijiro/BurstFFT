using System.Collections.Generic;
using System.Linq;
using Unity.Mathematics;
using Unity.Collections;

public static class BitReversalFft
{
    static public NativeArray<float> Transform(IEnumerable<float> input)
    {
        var buffer = Prepare(input);
        Fft(buffer);
        var output = Power(buffer, buffer.Length / 2).ToArray();
        return new NativeArray<float>(output, Allocator.Persistent);
    }

    static void Fft(float2[] A)
    {
        var N = A.Length;

        for (var m = 2; m <= N; m <<= 1)
        {
            for (var k = 0; k < N; k += m)
            {
                for (var j = 0; j < m / 2; j++)
                {
                    var i1 = k + j;
                    var i2 = k + j + m / 2;

                    var t = Mulc(Expi(-2 * math.PI * j / m), A[i2]);
                    var u = A[i1];

                    A[i1] = u + t;
                    A[i2] = u - t;
                }
            }
        }
    }

    static float2[] Prepare(IEnumerable<float> input)
    {
        var source = input.ToArray();
        var output = new float2[source.Length];
        var logN = (int)math.log2(source.Length);
        for (var i = 0u; i < source.Length; i++)
            output[BitReverseIndex(i, logN)] = math.float2(source[i], 0);
        return output;
    }

    static uint BitReverseIndex(uint index, int logN)
    {
        var acc = 0u;
        for (var i = 0; i < logN; i++)
            if (((1u << i) & index) != 0) acc += 1u << (logN - 1 - i);
        return acc;
    }

    static float2 Expi(float x)
      => math.float2(math.cos(x), math.sin(x));

    static float2 Mulc(float2 a, float2 b)
      => math.float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);

    static IEnumerable<float> Power(IEnumerable<float2> input, int length)
      => input.Select(c => math.sqrt(c.x * c.x + c.y * c.y) / length);
}
