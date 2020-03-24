using System.Linq;
using Unity.Mathematics;
using Unity.Collections;

public sealed class FftBuffer
{
    public FftBuffer(int width)
    {
        _N = width;
        _logN = (int)math.log2(_N);

        _rev = Enumerable.Range(0, _N)
          .Select(i => BitReverseIndex(i)).ToArray();

        BuildOperatorTable();
    }

    public NativeArray<float> Transform(float[] input)
    {
        var A = new float4[_N / 2];

        for (var i = 0; i < _N; i += 2)
            A[i / 2] =
              math.float4(input[_rev[i    ]], 0,
                          input[_rev[i + 1]], 0);

        var Cppnn = math.float4(1, 1, -1, -1);

        for (var i = 0; i < _N / 2; i++)
            A[i] = A[i].xyxy + Cppnn * A[i].zwzw;

        var op_i = 0;

        for (var i = 0; i < _logN - 1; i++)
            for (var j = 0; j < _N / 4; j++)
            {
                var o = _op[op_i++];

                var t = Mulc(o.W4, A[o.I.y]);
                var u = A[o.I.x];

                A[o.I.x] = u + t;
                A[o.I.y] = u - t;
            }

        var output = new NativeArray<float>(_N, Allocator.Persistent,
                                            NativeArrayOptions.UninitializedMemory);
        for (var i = 0; i < _N / 2; i++)
        {
            output[i * 2    ] = math.length(A[i].xy) * 2 / _N;
            output[i * 2 + 1] = math.length(A[i].zw) * 2 / _N;
        }

        return output;
    }

    int _N;
    int _logN;
    int[] _rev;
    Operator[] _op;

    struct Operator
    {
        public int2 I;
        public float2 W;

        public float4 W4
          => math.float4(W.x, math.sqrt(1 - W.x * W.x),
                         W.y, math.sqrt(1 - W.y * W.y));
    }

    int BitReverseIndex(int x)
      => Enumerable.Range(0, _logN)
        .Aggregate(0, (acc, i) => acc += ((x >> i) & 1) << (_logN - 1 - i));

    void BuildOperatorTable()
    {
        _op = new Operator[(_logN - 1) * (_N / 4)];

        var i = 0;

        for (var m = 4; m <= _N; m <<= 1)
            for (var k = 0; k < _N; k += m)
                for (var j = 0; j < m / 2; j += 2)
                    _op[i++] = new Operator {
                      W = math.cos(-2 * math.PI / m * math.float2(j, j + 1)),
                      I = math.int2((k + j) / 2, (k + j + m / 2) / 2)
                    };
    }

    static float4 Mulc(float4 a, float4 b)
      => a.xxzz * b.xyzw + math.float4(-1, 1, -1, 1) * a.yyww * b.yxwz;
}
