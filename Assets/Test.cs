using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.Collections;
using Unity.Mathematics;
using Stopwatch = System.Diagnostics.Stopwatch;

sealed class Test : MonoBehaviour
{
    const int Width = 1024;

    IEnumerable<float> TestData
      => Enumerable.Range(0, Width)
         .Select(i => (float)i / Width)
         .Select(x => 5 * noise.snoise(math.float2(0.324f, x * 1000))
                      + math.sin(x * 33)
                      + math.sin(x * 300)
                      + math.sin(x * 900));

    Texture2D Benchmark<TDft>(TDft dft, NativeArray<float> input)
      where TDft : IDft, System.IDisposable
    {
        var texture = new Texture2D(Width / 2, 1, TextureFormat.RFloat, false);

        // We reject the first test.
        dft.Transform(input);

        var sw = new System.Diagnostics.Stopwatch();

        // Benchmark
        const int iteration = 32;
        sw.Start();
        for (var i = 0; i < iteration; i++) dft.Transform(input);
        sw.Stop();

        // Show the average time.
        var us = 1000.0 * 1000 * sw.ElapsedTicks / Stopwatch.Frequency;
        Debug.Log(us / iteration);

        texture.LoadRawTextureData(dft.Spectrum);
        texture.Apply();

        return texture;
    }

    Texture2D _dft1;
    Texture2D _dft2;
    Texture2D _fft;

    void Start()
    {
        using (var data = TempJobMemory.New<float>(TestData))
        {
            using (var ft = new NaiveDft(Width)) _dft1 = Benchmark(ft, data);
            using (var ft = new BurstDft(Width)) _dft2 = Benchmark(ft, data);
            using (var ft = new BurstFft(Width)) _fft  = Benchmark(ft, data);
        }
    }

    void OnGUI()
    {
        Graphics.DrawTexture(new Rect(10, 10, Width / 2, 16), _dft1);
        Graphics.DrawTexture(new Rect(10, 38, Width / 2, 16), _dft2);
        Graphics.DrawTexture(new Rect(10, 64, Width / 2, 16), _fft);
    }
}

