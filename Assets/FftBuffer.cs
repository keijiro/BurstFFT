using System.Linq;
using Unity.Collections;
using Unity.Mathematics;
using Unity.Jobs;

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

        var job1 = new FirstPassJob { I = input, Rev = _rev, A = A };
        var handle = job1.Schedule(_N / 2, 16);

        for (var i = 0; i < _logN - 1; i++)
        {
            var slice = new NativeSlice<Operator>(_op, _N / 4 * i);
            var job2 = new FftCoreJob { A = A, Op = slice };
            handle = job2.Schedule(_N / 4, 16, handle);
        }

        var output = new NativeArray<float>(_N, Allocator.Persistent,
                                            NativeArrayOptions.UninitializedMemory);

        var job3 = new PostprocessJob { A = A, O = output.Reinterpret<float2>(sizeof(float)) };
        handle = job3.Schedule(_N / 2, 16, handle);

        handle.Complete();

        A.Dispose();

        return output;
    }

    static float4 Mulc(float4 a, float4 b)
      => a.xxzz * b.xyzw + math.float4(-1, 1, -1, 1) * a.yyww * b.yxwz;

    #region First pass job

    [Unity.Burst.BurstCompile(CompileSynchronously = true)]
    struct FirstPassJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float> I;
        [ReadOnly] public NativeArray<int2> Rev;
        [WriteOnly] public NativeArray<float4> A;

        public void Execute(int i)
        {
            var a1 = I[Rev[i].x];
            var a2 = I[Rev[i].y];
            A[i] = math.float4(a1 + a2, 0, a1 - a2, 0);
        }
    }

    #endregion

    #region FFT core job

    [Unity.Burst.BurstCompile(CompileSynchronously = true)]
    struct FftCoreJob : IJobParallelFor
    {
        [NativeDisableParallelForRestriction] public NativeArray<float4> A;
        [ReadOnly] public NativeSlice<Operator> Op;

        public void Execute(int i)
        {
            var o = Op[i];
            var t = Mulc(o.W4, A[o.i2]);
            var u = A[o.i1];
            A[o.i1] = u + t;
            A[o.i2] = u - t;
        }
    }

    #endregion

    #region Postprocess Job

    [Unity.Burst.BurstCompile(CompileSynchronously = true)]
    struct PostprocessJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float4> A;
        [WriteOnly] public NativeArray<float2> O;

        public void Execute(int i)
        {
            var a = A[i];
            O[i] = math.float2(math.length(a.xy), math.length(a.zw)) / O.Length;
        }
    }

    #endregion

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

        public int i1 => I.x;
        public int i2 => I.y;

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
