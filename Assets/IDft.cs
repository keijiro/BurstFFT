using Unity.Collections;

public interface IDft
{
    NativeArray<float> Spectrum { get; }
    void Transform(NativeArray<float> input);
}
