using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Unity.Mathematics;
using Unity.Collections;

public static class BitReversalFft
{
    static public NativeArray<float> Transform(IEnumerable<float> input)
    {
        var buffer = Preprocess(input);
        Fft(buffer);
        return new NativeArray<float>(Postprocess(buffer), Allocator.Persistent);
    }

    static uint BitReverseIndex(uint index, int logN)
    {
        var acc = 0u;
        for (var i = 0; i < logN; i++)
            if (((1u << i) & index) != 0) acc += 1u << (logN - 1 - i);
        return acc;
    }

    static float2[] Preprocess(IEnumerable<float> input)
    {
        var source = input.ToArray();
        var output = new float2[source.Length];
        var logN = (int)math.log2(source.Length);
        for (var i = 0u; i < source.Length; i++)
            output[BitReverseIndex(i, logN)] = math.float2(source[i], 0);
        return output;
    }

    static float2 Expi(float t)
      => math.float2(math.cos(t), math.sin(t));

    static float4 Expi(float2 t)
      => math.float4(math.cos(t.x), math.sin(t.x), math.cos(t.y), math.sin(t.y));

    static float2 Mulc(float2 a, float2 b)
      => math.float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);

    static float4 Mulc(float4 a, float4 b)
      => a.xxzz * b.xyzw + math.float4(-1, 1, -1, 1) * a.yyww * b.yxwz;
      
    static void Fft(float2[] A)
    {
        var A4 = MemoryMarshal.Cast<float2, float4>(new Span<float2>(A));
        var N = A.Length;

        var Cppnn = math.float4(1, 1, -1, -1);

        for (var i = 0; i < N / 2; i++)
            A4[i] = A4[i].xyxy + Cppnn * A4[i].zwzw;

        for (var m = 4; m <= N; m <<= 1)
        {
            for (var k = 0; k < N; k += m)
            {
                for (var j = 0; j < m / 2; j += 2)
                {
                    var i1 = (k + j) / 2;
                    var i2 = (k + j + m / 2) / 2;

                    var w = Expi(-2 * math.PI / m * math.float2(j, j + 1));
                    var t = Mulc(w, A4[i2]);
                    var u = A4[i1];

                    A4[i1] = u + t;
                    A4[i2] = u - t;
                }
            }
        }
    }

    static float[] Postprocess(float2[] input)
      => input.Select(c => math.length(c) * 2 / input.Length).ToArray();
}
