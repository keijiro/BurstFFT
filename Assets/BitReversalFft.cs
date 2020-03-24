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
        var (A_r, A_i) = Prepare(input);
        Fft(A_r, A_i);
        Postprocess(A_r, A_i);
        return new NativeArray<float>(A_r, Allocator.Persistent);
    }

    static uint BitReverseIndex(uint index, int logN)
    {
        var acc = 0u;
        for (var i = 0; i < logN; i++)
            if (((1u << i) & index) != 0) acc += 1u << (logN - 1 - i);
        return acc;
    }

    static (float[], float[]) Prepare(IEnumerable<float> input)
    {
        var source = input.ToArray();
        var output = new float[source.Length];
        var logN = (int)math.log2(source.Length);
        for (var i = 0u; i < source.Length; i++)
            output[BitReverseIndex(i, logN)] = source[i];
        return (output, new float[source.Length]);
    }

    static float MulExpi_r(float r, float i, float t)
      => r * math.cos(t) - i * math.sin(t);

    static float MulExpi_i(float r, float i, float t)
      => r * math.sin(t) + i * math.cos(t);

    static float4 MulExpi_r(float4 r, float4 i, float4 t)
      => r * math.cos(t) - i * math.sin(t);

    static float4 MulExpi_i(float4 r, float4 i, float4 t)
      => r * math.sin(t) + i * math.cos(t);

    static void Fft(float[] A_r, float[] A_i)
    {
        var N = A_r.Length;

        for (var m = 2; m <= N; m <<= 1)
        {
            for (var k = 0; k < N; k += m)
            {
                if (m / 2 < 4)
                    for (var j = 0; j < m / 2; j++)
                    {
                        var i1 = k + j;
                        var i2 = k + j + m / 2;

                        var x = -2 * math.PI * j / m;

                        var t_r = MulExpi_r(A_r[i2], A_i[i2], x);
                        var t_i = MulExpi_i(A_r[i2], A_i[i2], x);
                        var (u_r, u_i) = (A_r[i1], A_i[i1]);

                        (A_r[i1], A_i[i1]) = (u_r + t_r, u_i + t_i);
                        (A_r[i2], A_i[i2]) = (u_r - t_r, u_i - t_i);
                    }
                else
                {
                    var A4_r = MemoryMarshal.Cast<float, float4>(new Span<float>(A_r));
                    var A4_i = MemoryMarshal.Cast<float, float4>(new Span<float>(A_i));

                    for (var j = 0; j < m / 2; j += 4)
                    {
                        var i1 = (k + j) / 4;
                        var i2 = (k + j + m / 2) / 4;

                        var x = math.float4(
                          -2 * math.PI *  j      / m,
                          -2 * math.PI * (j + 1) / m,
                          -2 * math.PI * (j + 2) / m,
                          -2 * math.PI * (j + 3) / m
                        );

                        var t_r = MulExpi_r(A4_r[i2], A4_i[i2], x);
                        var t_i = MulExpi_i(A4_r[i2], A4_i[i2], x);
                        var (u_r, u_i) = (A4_r[i1], A4_i[i1]);

                        (A4_r[i1], A4_i[i1]) = (u_r + t_r, u_i + t_i);
                        (A4_r[i2], A4_i[i2]) = (u_r - t_r, u_i - t_i);
                    }
                }
            }
        }
    }

    static void Postprocess(float[] A_r, float[] A_i)
    {
        var len = A_r.Length;
        for (var i = 0; i < len; i++)
            A_r[i] = math.sqrt(A_r[i] * A_r[i] + A_i[i] * A_i[i]) * 2 / len;
    }
}
