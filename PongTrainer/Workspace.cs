namespace BrainPong;

/// <summary>
/// Per-sample scratch: the activation of every layer (needed again on the way back) and the gradients flowing down.
/// Each worker thread owns its own Workspace, because the network's weights are shared but these buffers are written
/// to during every forward/backward pass, so sharing them across threads would corrupt results.
/// </summary>
internal sealed class Workspace
{
    /// <summary>The network's input: the stacked game frames, flattened into one array. Fill this before calling Forward.</summary>
    public float[] Input = [];
    public float[][] A = [];     // A[l] = output of layer l
    public float[][] D = [];     // D[l] = dLoss/dOutput of layer l
}
