using System.Collections.Generic;
using System.Linq;
using Unity.Collections;

public static class TempJobMemory
{
    public static NativeArray<T> New<T>(IEnumerable<T> e) where T : unmanaged
      => new NativeArray<T>(e.ToArray(), Allocator.TempJob);

    public static NativeArray<T> New<T>(int size) where T : unmanaged
      => new NativeArray<T>(size, Allocator.TempJob,
                            NativeArrayOptions.UninitializedMemory);
}

public static class PersistentMemory
{
    public static NativeArray<T> New<T>(IEnumerable<T> e) where T : unmanaged
      => new NativeArray<T>(e.ToArray(), Allocator.Persistent);

    public static NativeArray<T> New<T>(int size) where T : unmanaged
      => new NativeArray<T>(size, Allocator.Persistent,
                            NativeArrayOptions.UninitializedMemory);
}
