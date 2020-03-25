using System.Linq;
using UnityEngine;
using Unity.Collections;
using Unity.Mathematics;

sealed class Test : MonoBehaviour
{
    Texture2D _dftTexture;
    Texture2D _fftTexture;

    void Start()
    {
        var source = Enumerable.Range(0, 1024).
          Select(i => i / 1024.0f).
          Select(x => 5 * noise.snoise(math.float2(0.324f, x * 1000)) +
                      math.sin(x * 33) + math.sin(x * 300) + math.sin(x * 900));

        _dftTexture = new Texture2D(512, 1, TextureFormat.RFloat, false);
        _fftTexture = new Texture2D(512, 1, TextureFormat.RFloat, false);

        using (var source_na = new NativeArray<float>(source.ToArray(), Allocator.Persistent))
        {
            using (var spectrum = Dft.Transform(source))
                _dftTexture.LoadRawTextureData(spectrum);
            _dftTexture.Apply();

            using (var fft = new FftBuffer(1024))
            {
                fft.Transform(source_na);
                _fftTexture.LoadRawTextureData(fft.Spectrum);
            }
            _fftTexture.Apply();
        }
    }

    void OnGUI()
    {
        if (!Event.current.type.Equals(EventType.Repaint)) return;
        Graphics.DrawTexture(new Rect(10, 10, 512, 16), _dftTexture);
        Graphics.DrawTexture(new Rect(10, 38, 512, 16), _fftTexture);
    }
}

