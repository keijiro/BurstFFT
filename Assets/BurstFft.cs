#define SINGLE_THREAD

using System.Linq;
using Unity.Collections;
using Unity.Mathematics;
using Unity.Jobs;

// Cooley–Tukey FFT vectorized/parallelized with the Burst compiler

public sealed class BurstFft : IDft, System.IDisposable
{
    #region Public properties and methods

    public NativeArray<float> Spectrum => _O;

    public BurstFft(int width)
    {
        _N = width;
        _logN = (int)math.log2(width);

        BuildPermutationTable();
        BuildTwiddleFactors();

        _O = PersistentMemory.New<float>(_N);
        _X = PersistentMemory.New<float4>(_N / 2);
    }

    public void Dispose()
    {
        if (_P.IsCreated) _P.Dispose();
        if (_T.IsCreated) _T.Dispose();
        if (_O.IsCreated) _O.Dispose();
        if (_X.IsCreated) _X.Dispose();
    }

    #if SINGLE_THREAD

    public void Transform(NativeArray<float> input)
    {
        // Bit-reversal permutation and first DFT pass
        new FirstPassJob { I = input, P = _P, X = _X }.Run(_N / 2);

        // 2nd and later DFT passes
        for (var i = 0; i < _logN - 1; i++)
        {
            var T_slice = new NativeSlice<TFactor>(_T, _N / 4 * i);
            new DftPassJob { T = T_slice, X = _X }.Run(_N / 4);
        }

        // Postprocess (power spectrum calculation)
        var O2 = _O.Reinterpret<float2>(sizeof(float));
        new PostprocessJob { X = _X, O = O2, s = 2.0f / _N }.Run(_N / 2);
    }

    #else

    public JobHandle Schedule(NativeArray<float> input, int parallelism = 32)
    {
        // Bit-reversal permutation and first DFT pass
        var handle = new FirstPassJob { I = input, P = _P, X = _X }
          .Schedule(_N / 2, parallelism);

        // 2nd and later DFT passes
        for (var i = 0; i < _logN - 1; i++)
        {
            var T_slice = new NativeSlice<TFactor>(_T, _N / 4 * i);
            handle = new DftPassJob { T = T_slice, X = _X }
              .Schedule(_N / 4, parallelism, handle);
        }

        // Postprocess (power spectrum calculation)
        var O2 = _O.Reinterpret<float2>(sizeof(float));
        handle = new PostprocessJob { X = _X, O = O2, s = 2.0f / _N }
          .Schedule(_N / 2, parallelism, handle);

        return handle;
    }

    public void Transform(NativeArray<float> input)
    {
        Schedule(input).Complete();
    }

    #endif

    #endregion

    #region Private members

    readonly int _N;
    readonly int _logN;
    NativeArray<float> _O;
    NativeArray<float4> _X;

    #endregion

    #region Bit-reversal permutation table

    NativeArray<int2> _P;

    void BuildPermutationTable()
    {
        _P = PersistentMemory.New<int2>(_N / 2);
        for (var i = 0; i < _N; i += 2)
            _P[i / 2] = math.int2(Permutate(i), Permutate(i + 1));
    }

    int Permutate(int x)
      => Enumerable.Range(0, _logN)
         .Aggregate(0, (acc, i) => acc += ((x >> i) & 1) << (_logN - 1 - i));

    #endregion

    #region Precalculated twiddle factors

    struct TFactor
    {
        public int2 I;
        public float2 W;

        public int i1 => I.x;
        public int i2 => I.y;

        public float4 W4
          => math.float4(W.x, math.sqrt(1 - W.x * W.x),
                         W.y, math.sqrt(1 - W.y * W.y));
    }

    NativeArray<TFactor> _T;

    void BuildTwiddleFactors()
    {
        _T = PersistentMemory.New<TFactor>((_logN - 1) * (_N / 4));

        var i = 0;
        for (var m = 4; m <= _N; m <<= 1)
            for (var k = 0; k < _N; k += m)
                for (var j = 0; j < m / 2; j += 2)
                    _T[i++] = new TFactor
                      { I = math.int2((k + j) / 2, (k + j + m / 2) / 2),
                        W = math.cos(-2 * math.PI / m * math.float2(j, j + 1)) };
    }

    #endregion

    #region First pass job

    [Unity.Burst.BurstCompile(CompileSynchronously = true)]
    struct FirstPassJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float> I;
        [ReadOnly] public NativeArray<int2> P;
        [WriteOnly] public NativeArray<float4> X;

        public void Execute(int i)
        {
            var a1 = I[P[i].x];
            var a2 = I[P[i].y];
            X[i] = math.float4(a1 + a2, 0, a1 - a2, 0);
        }
    }

    #endregion

    #region DFT pass job

    [Unity.Burst.BurstCompile(CompileSynchronously = true)]
    struct DftPassJob : IJobParallelFor
    {
        [ReadOnly] public NativeSlice<TFactor> T;
        [NativeDisableParallelForRestriction] public NativeArray<float4> X;

        static float4 Mulc(float4 a, float4 b)
          => a.xxzz * b.xyzw + math.float4(-1, 1, -1, 1) * a.yyww * b.yxwz;

        public void Execute(int i)
        {
            var t = T[i];
            var e = X[t.i1];
            var o = Mulc(t.W4, X[t.i2]);
            X[t.i1] = e + o;
            X[t.i2] = e - o;
        }
    }

    #endregion

    #region Postprocess Job

    [Unity.Burst.BurstCompile(CompileSynchronously = true)]
    struct PostprocessJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float4> X;
        [WriteOnly] public NativeArray<float2> O;
        public float s;

        public void Execute(int i)
        {
            var x = X[i];
            O[i] = math.float2(math.length(x.xy), math.length(x.zw)) * s;
        }
    }

    #endregion
}
