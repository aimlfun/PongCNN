namespace BrainPong;

/// <summary>
/// Initialises weights for a layer. He initialisation keeps the signal size steady through relu layers.
/// </summary>
internal static class WeightInit
{
    /// <summary>
    /// He initialisation: weights ~ N(0, 2/fanIn). Keeps the signal size steady through relu layers.
    /// Remember RELU = x<0?0:x. So if you chain neuron 1, with neuron 2, the output will get larger and larger. By doing this
    /// it ensures it doesn't grow wildly.
    /// </summary>
    /// <param name="weights"></param>
    /// <param name="offsetInWeights"></param>
    /// <param name="count"></param>
    /// <param name="fanIn"></param>
    /// <param name="rng"></param>
    public static void He(float[] weights, int offsetInWeights, int count, int fanIn, Random rng)
    {
        double std = Math.Sqrt(2.0 / fanIn);

        for (int index = 0; index < count; index++)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();

            weights[offsetInWeights + index] = (float)(std * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }
    }
}
