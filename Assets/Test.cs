using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UIElements;
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

        var root = GetComponent<UIDocument>().rootVisualElement;
        root.Q("image1").style.backgroundImage = _dft1;
        root.Q("image2").style.backgroundImage = _dft2;
        root.Q("image3").style.backgroundImage = _fft;
    }
}

