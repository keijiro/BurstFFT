using System.Collections.Generic;
using System.Linq;
using Unity.Mathematics;
using Unity.Collections;

public static class NaiveFft
{
    static public NativeArray<float> Transform(IEnumerable<float> input)
    {
        var buffer = input.Select(r => math.float2(r, 0)).ToArray();
        var length = buffer.Length;
        var output = Power(Ditfft(buffer, length, 1), length / 2).ToArray();
        return new NativeArray<float>(output, Allocator.Persistent);
    }

    static IEnumerable<float2> Ditfft(IEnumerable<float2> input, int N, int s)
    {
        if (N == 1) return input.Take(1);

        var sub1 = Ditfft(input        , N / 2, s * 2);
        var sub2 = Ditfft(input.Skip(s), N / 2, s * 2);

        var X = sub1.Concat(sub2).ToArray();

        for (var k = 0; k < N / 2; k++)
        {
            var t = X[k];
            var wX = Mulc(Expi(-2 * math.PI * k / N), X[k + N / 2]);
            X[k        ] = t + wX;
            X[k + N / 2] = t - wX;
        }

        return X;
    }

    static float2 Expi(float x)
      => math.float2(math.cos(x), math.sin(x));

    static float2 Mulc(float2 a, float2 b)
      => math.float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);

    static IEnumerable<float> Power(IEnumerable<float2> input, int length)
      => input.Select(c => math.sqrt(c.x * c.x + c.y * c.y) / length);
}
