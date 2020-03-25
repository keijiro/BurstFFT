using Unity.Collections;
using Unity.Mathematics;

// Naive DFT implementation

public sealed class NaiveDft : IDft, System.IDisposable
{
    public NativeArray<float> Spectrum => _buffer;

    NativeArray<float> _buffer;

    public NaiveDft(int width)
      => _buffer = PersistentMemory.New<float>(width / 2);

    public void Dispose()
    {
        if (_buffer.IsCreated) _buffer.Dispose();
    }

    public void Transform(NativeArray<float> input)
    {
        var N = _buffer.Length * 2;

        for (var k = 0; k < N / 2; k++)
        {
            var acc = float2.zero;

            for (var n = 0; n < N; n++)
            {
                var x = input[n];
                var t = 2 * math.PI / N * k * n;
                acc += math.float2(math.cos(t) * x,  -math.sin(t) * x);
            }

            _buffer[k] = math.length(acc) * 2 / N;
        }
    }
}
