using System.Linq;
using Unity.Mathematics;
using Unity.Collections;

public sealed class FftBuffer : System.IDisposable
{
    public FftBuffer(int width)
    {
        _N = width;
        _logN = (int)math.log2(_N);

        BuildBitReverseTable();
        BuildOperators();
    }

    public void Dispose()
    {
        if (_rev.IsCreated) _rev.Dispose();
        if (_op.IsCreated) _op.Dispose();
    }

    public NativeArray<float> Transform(NativeArray<float> input)
    {
        var A = TempJobMemory.New<float4>(_N / 2);

        for (var i = 0; i < _N / 2; i++)
        {
            var a1 = input[_rev[i].x];
            var a2 = input[_rev[i].y];
            A[i] = math.float4(a1 + a2, 0, a1 - a2, 0);
        }

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

        A.Dispose();

        return output;
    }

    static float4 Mulc(float4 a, float4 b)
      => a.xxzz * b.xyzw + math.float4(-1, 1, -1, 1) * a.yyww * b.yxwz;

    #region Transform configuration

    readonly int _N;
    readonly int _logN;

    #endregion

    #region Bit reverse table

    NativeArray<int2> _rev;

    void BuildBitReverseTable()
    {
        _rev = new NativeArray<int2>
          ( _N / 2, Allocator.Persistent,
            NativeArrayOptions.UninitializedMemory );

        for (var i = 0; i < _N; i += 2)
            _rev[i / 2] = math.int2(BitReverseIndex(i), BitReverseIndex(i + 1));
    }

    int BitReverseIndex(int x)
      => Enumerable.Range(0, _logN)
         .Aggregate(0, (acc, i) => acc += ((x >> i) & 1) << (_logN - 1 - i));

    #endregion

    #region FFT Operators

    struct Operator
    {
        public int2 I;
        public float2 W;

        public float4 W4
          => math.float4(W.x, math.sqrt(1 - W.x * W.x),
                         W.y, math.sqrt(1 - W.y * W.y));
    }

    NativeArray<Operator> _op;

    void BuildOperators()
    {
        _op = new NativeArray<Operator>
          ( (_logN - 1) * (_N / 4), Allocator.Persistent,
            NativeArrayOptions.UninitializedMemory );

        var i = 0;
        for (var m = 4; m <= _N; m <<= 1)
            for (var k = 0; k < _N; k += m)
                for (var j = 0; j < m / 2; j += 2)
                    _op[i++] = new Operator
                      { I = math.int2((k + j) / 2, (k + j + m / 2) / 2),
                        W = math.cos(-2 * math.PI / m * math.float2(j, j + 1)) };
    }

    #endregion
}
