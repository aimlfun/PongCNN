namespace BrainPong;

/// <summary>
/// Result of QNet.Explain: how much each connection pushed one action's Q-value up (+) or down (-) for one state.
/// Used by the viewer's "Why?" window to show which parts of the network drove a decision.
/// </summary>
internal sealed class Attribution
{
    /// <summary>[frame][conv1 unit]: how much one input frame's pixels contributed through each first-layer filter.</summary>
    public float[][] InToC1 = [];

    /// <summary>[conv1 unit][conv2 unit]: contribution flowing from each first-layer filter to each second-layer filter.</summary>
    public float[][] C1ToC2 = [];

    /// <summary>[conv2 unit]: contribution of each second-layer filter to the chosen action's Q-value (the dense layer in between is folded in).</summary>
    public float[] C2ToQ = [];

    /// <summary>One value per input pixel (all frames): how much that pixel mattered. Draw it as a heat map.</summary>
    public float[] Saliency = [];

    /// <summary>Which action's Q-value this explains (0, 1 or 2 = the three paddle moves).</summary>
    public int Action;
}
