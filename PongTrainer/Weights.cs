namespace BrainPong;

/// <summary>
/// Saves and loads a network's weights as a small binary file.
/// File layout: a 4-byte int (the number of weights), then that many 4-byte floats.
/// </summary>
internal static class Weights
{
    /// <summary>
    /// Writes the weights to <paramref name="path"/>. It writes to a temporary file first and then renames it, so a
    /// crash or a viewer reading mid-save never sees a half-written file.
    /// </summary>
    public static void Save(string path, float[] weights)
    {
        string tmp = path + ".tmp";

        using var w = new BinaryWriter(File.Create(tmp));

        w.Write(weights.Length); // first data item in the count of weights, used to confirm input matches NN size

        foreach (var v in weights)
        {
            w.Write(v);
        }

        w.Close();

        File.Move(tmp, path, overwrite: true);
    }

    /// <summary>
    /// Reads weights from <paramref name="path"/>. <paramref name="expected"/> is how many weights the network needs;
    /// if the file has a different count (e.g. it was saved from a different network shape) this throws
    /// rather than quietly loading nonsense.
    /// </summary>
    public static float[] Load(string path, int expected)
    {
        using var r = new BinaryReader(File.OpenRead(path));

        // first value should be number of items in the weights.
        int n = r.ReadInt32();

        if (n != expected) throw new InvalidDataException($"weights have {n} params, network needs {expected}");
        
        var t = new float[n];
        for (int i = 0; i < n; i++)
        {
            t[i] = r.ReadSingle();
        }
        
        return t;
    }
}
