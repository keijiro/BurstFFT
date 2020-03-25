using System.Linq;
using UnityEngine;
using Unity.Collections;
using Unity.Mathematics;

sealed class Test : MonoBehaviour
{
    const int Width = 1024;

    Texture2D _dftTexture;
    Texture2D _fftTexture;

    void Start()
    {
        var source = Enumerable.Range(0, Width).
          Select(i => (float)i / Width).
          Select(x => 5 * noise.snoise(math.float2(0.324f, x * 1000)) +
                      math.sin(x * 33) + math.sin(x * 300) + math.sin(x * 900));

        _dftTexture = new Texture2D(Width / 2, 1, TextureFormat.RFloat, false);
        _fftTexture = new Texture2D(Width / 2, 1, TextureFormat.RFloat, false);

        using (var input = TempJobMemory.New<float>(source))
        {
            using (var dft = new DftBuffer(Width))
            {
                dft.Transform(input);
                _dftTexture.LoadRawTextureData(dft.Spectrum);
            }

            using (var fft = new FftBuffer(Width))
            {
                fft.Transform(input);
                _fftTexture.LoadRawTextureData(fft.Spectrum);
            }
        }

        _dftTexture.Apply();
        _fftTexture.Apply();
    }

    void OnGUI()
    {
        if (!Event.current.type.Equals(EventType.Repaint)) return;
        Graphics.DrawTexture(new Rect(10, 10, Width / 2, 16), _dftTexture);
        Graphics.DrawTexture(new Rect(10, 38, Width / 2, 16), _fftTexture);
    }
}

