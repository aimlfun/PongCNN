using static System.Runtime.InteropServices.JavaScript.JSType;

namespace BrainPong;

/// <summary>
/// The Q-network: Q(state) gives one value per action. The weights are NOT stored here but in a 
/// flat theta array passed in, so the same code runs the online net, the frozen target net, and a gradient check.
/// </summary>
internal sealed class QNet
{
    /// <summary>
    /// The specification for this Q-netowrk.
    /// </summary>
    public readonly NetSpec Spec;

    /// <summary>
    /// Stores the layers for this Q-network.
    /// </summary>
    private readonly ILayer[] _layers;

    /// <summary>
    /// Number of params.
    /// </summary>
    public readonly int ParamCount;

    public int InputSize => _layers[0].InSize;

    /// <summary>
    /// How many actions the Q-network supports.
    /// </summary>
    public int Actions => Spec.Actions;

    /// <summary>
    /// How many layers are in the Q-network.
    /// </summary>
    public int LayerCount => _layers.Length;

    /// <summary>
    /// Number of parameters in the first k layers (they sit first in theta). 3 = conv1 + conv2 + dense = the "trunk" that is shared with the pretraining net.
    /// </summary>
    public int ParamsUpTo(int k)
    {
        int n = 0;

        for (int i = 0; i < k; i++)
        {
            n += _layers[i].ParamCount;
        }

        return n;
    }

    /// <summary>
    /// Constructor. Initialises the network structure and computes the offsets of each layer's parameters in the flat theta array.
    /// </summary>
    /// <param name="spec"></param>
    public QNet(NetSpec spec)
    {
        Spec = spec;

        // The conv layers are the "trunk" of the network, shared with the pretraining net.

        // 6400->2592->1024->128->3 = stay,up,down [argmax()]
        // CNN ->CNN ->NN  ->NN ->    ACTION

        // input: 6400 } 4 frames x 40px x 40px
        // param-count: 1160
        Conv2D convolution1 = new(
            inputChannels: spec.Frames,  // 4 frames
            inputHeight: spec.Size,      // 40px } display
            inputWidth: spec.Size,       // 40px }
            outputChannels: spec.C1,     // 8 
            kernelSize: spec.KernelSize1,// 6 (6x6 kernel)
            stride: spec.S1,             // 2
            relu: true);                 // relu (0..+ve only)

        // => output: 8 channels of 18x18 (2592)

        // input: 8 channels of (18x18) = 2592
        // param-count 2064
        Conv2D convolution2 = new(
            inputChannels: spec.C1,                 // 8
            inputHeight: convolution1.OutputHeight, // 18
            inputWidth: convolution1.OutputWidth,   // 18
            outputChannels: spec.C2,                // 16
            kernelSize: spec.KernelSize2,           // 4
            stride: spec.S2,                        // 2
            relu: true);                            // rel (0..+ve only)

        // => output: 1024

        // The dense layers are the "head" that outputs Q-values.
        // "flatten" is free: the conv output is already one flat array, the dense layer just reads it as a vector
        // (dense means: every input is connected to every output)
        Dense f1 = new(
            inSize: convolution2.OutSize,     // in: 1024 from convolution network 2
            outSize: spec.Fc,                 // out: 128
            relu: true);                      // relu

        // linear head: Q-values can be any real number
        Dense f2 = new(
            inSize: spec.Fc,        // in: 128
            outSize: spec.Actions,  // out:  3
            relu: false);           // no relu

        // The layers are stored in order, so the offsets of their parameters in theta (weight) can be computed.
        // The first 3 layers are the trunk, shared with the pretraining net.
        // The last layer is the head, which is not shared.
        _layers = [convolution1, convolution2, f1, f2];

        int off = 0;

        foreach (var l in _layers)
        {
            l.Offset = off;  // the offset of this layer's parameters in theta
            off += l.ParamCount;
        }

        ParamCount = off; // total number of parameters in the network
    }

    /// <summary>
    /// Initialises a new theta array with random weights. The same code runs the online net, 
    /// the frozen target net, and a gradient check, so the weights are not stored in the QNet object itself.
    /// </summary>
    /// <param name="seed"></param>
    /// <returns></returns>
    public float[] NewParams(int seed)
    {
        float[] weights = new float[ParamCount];

#pragma warning disable S2245 // PRG is intentional, so it can be repeated.
        Random rng = new(seed);
#pragma warning restore S2245

        foreach (var l in _layers)
        {
            l.Init(weights, rng);
        }

        return weights;
    }

    /// <summary>
    /// Initialises a new workspace for a single sample. The workspace stores the activations 
    /// of each layer and the gradients flowing down.
    /// </summary>
    /// <returns></returns>
    public Workspace NewWorkspace()
    {
        Workspace workspace = new()
        {
            Input = new float[InputSize],
            A = new float[_layers.Length][],
            D = new float[_layers.Length][]
        };

        for (int layer = 0; layer < _layers.Length; layer++)
        {
            workspace.A[layer] = new float[_layers[layer].OutSize];
            workspace.D[layer] = new float[_layers[layer].OutSize];
        }

        return workspace;
    }

    /// <summary>
    /// Forward pass for the state already placed in workspace.Input. Returns the Q-values (workspace.A[last]).
    /// </summary>
    public float[] Forward(float[] weights, Workspace workspace)
    {
        var x = workspace.Input;

        for (int i = 0; i < _layers.Length; i++)
        {
            _layers[i].Forward(weights, x, workspace.A[i]);
            x = workspace.A[i];
        }

        return workspace.A[^1];
    }

    /// <summary>
    /// Backward pass. dOut = dLoss/dQ (one number per action; zero for actions not taken). Adds dLoss/dTheta into grad.
    /// Must follow a Forward on the same workspace (it reuses the stored activations).
    /// </summary>
    public void Backward(float[] weights, float[] grad, Workspace workspace, float[] dOut, int stopAt = 0)
    {
        int last = _layers.Length - 1;

        Array.Copy(dOut, workspace.D[last], dOut.Length);

        // stopAt > 0: layers below it are frozen, so nothing below needs a gradient (and the chain rule stops there)
        for (int i = last; i >= stopAt; i--)
        {
            var x = i == 0 ? workspace.Input : workspace.A[i - 1];

            _layers[i].Backward(weights, grad, x, workspace.A[i], workspace.D[i], i == 0 || i == stopAt ? null : workspace.D[i - 1]);
        }
    }

    /// <summary>
    /// The conv1 filters, for the viewer: [C1][Frames][K1*K1]. 
    /// Shows what the network learned to look for.
    /// </summary>
    public float[][][] Conv1Filters(float[] theta)
    {
        var cnn = (Conv2D)_layers[0];
        int kernelSize = Spec.KernelSize1;
        var r = new float[Spec.C1][][];

        for (int co = 0; co < Spec.C1; co++)
        {
            r[co] = new float[Spec.Frames][];

            for (int frameIndex = 0; frameIndex < Spec.Frames; frameIndex++)
            {
                r[co][frameIndex] = new float[kernelSize * kernelSize];
                Array.Copy(theta, cnn.Offset + (co * Spec.Frames + frameIndex) * kernelSize * kernelSize, r[co][frameIndex], 0, kernelSize * kernelSize);
            }
        }

        return r;
    }

    /// <summary>
    /// Activations of conv1 for the viewer: [C1][H*W] for the state in ws.Input (call after Forward).
    /// </summary>
    public (float[][] maps, int h, int w) Conv1Maps(Workspace ws)
    {
        var cnn = (Conv2D)_layers[0];
        var maps = new float[cnn.OutputChannels][];
        int n = cnn.OutputHeight * cnn.OutputWidth;

        for (int i = 0; i < maps.Length; i++)
        {
            maps[i] = new float[n];

            Array.Copy(ws.A[0], i * n, maps[i], 0, n);
        }

        return (maps, cnn.OutputHeight, cnn.OutputWidth);
    }

    private float[]? _explainScratch;

    /// <summary>
    /// Splits the Q-value of one action into contributions along the network's connections, for the viewer.
    /// Call after Forward on the same workspace. Every number is "how much did this connection push Q(action) up (+) or down (-)
    /// for THIS state": activation x weight x (how much the rest of the network cares about the receiving unit).
    ///   InToC1[frame][unit]   : a frame's pixels  -> a conv1 unit
    ///   C1ToC2[unit1][unit2]  : a conv1 unit      -> a conv2 unit
    ///   C2ToQ[unit2]          : a conv2 unit      -> the action's Q (the dense layers in between are folded into one gradient)
    ///   Saliency[frame*area+p]: the same idea per input pixel
    /// Biases are ignored, so the pieces add up to roughly (not exactly) Q.
    /// </summary>
    public Attribution Explain(float[] weights, Workspace ws, int action)
    {
        var c1 = (Conv2D)_layers[0];
        var c2 = (Conv2D)_layers[1];

        // gradients of Q[action] with respect to the output of every layer, walking back from the output
        var scratch = _explainScratch ??= new float[ParamCount];
        Array.Clear(scratch);

        var dQ = new float[Spec.Actions]; dQ[action] = 1;
        var dFc = new float[_layers[2].OutSize];
        var dC2 = new float[_layers[1].OutSize];
        var dC1 = new float[_layers[0].OutSize];
        var dIn = new float[InputSize];

        _layers[3].Backward(weights, scratch, ws.A[2], ws.A[3], dQ, dFc);
        _layers[2].Backward(weights, scratch, ws.A[1], ws.A[2], dFc, dC2);
        _layers[1].Backward(weights, scratch, ws.A[0], ws.A[1], dC2, dC1);
        _layers[0].Backward(weights, scratch, ws.Input, ws.A[0], dC1, dIn);

        int F = Spec.Frames, size = Spec.Size, k1 = Spec.KernelSize1, s1 = Spec.S1, k2 = Spec.KernelSize2, s2 = Spec.S2;
        int h1 = c1.OutputHeight, w1 = c1.OutputWidth, h2 = c2.OutputHeight, w2 = c2.OutputWidth;

        var a = new Attribution
        {
            InToC1 = NewGrid(F, Spec.C1),
            C1ToC2 = NewGrid(Spec.C1, Spec.C2),
            C2ToQ = new float[Spec.C2],
            Saliency = new float[InputSize],
            Action = action,
        };

        // input frames -> conv1 units
        for (int i = 0; i < Spec.C1; i++)
        {
            for (int oy = 0; oy < h1; oy++)
            {
                for (int ox = 0; ox < w1; ox++)
                {
                    int p = (i * h1 + oy) * w1 + ox;

                    if (ws.A[0][p] <= 0) continue;                 // relu gate: unit was off here, nothing flowed through it

                    float g = dC1[p];

                    for (int f = 0; f < F; f++)
                    {
                        for (int ky = 0; ky < k1; ky++)
                        {
                            for (int kx = 0; kx < k1; kx++)
                            {
                                float w = weights[c1.Offset + ((i * F + f) * k1 + ky) * k1 + kx];
                                float x = ws.Input[(f * size + oy * s1 + ky) * size + ox * s1 + kx];

                                a.InToC1[f][i] += g * w * x;
                            }
                        }
                    }
                }
            }
        }

        // conv1 units -> conv2 units
        for (int j = 0; j < Spec.C2; j++)
        {
            for (int oy = 0; oy < h2; oy++)
            {
                for (int ox = 0; ox < w2; ox++)
                {
                    int p = (j * h2 + oy) * w2 + ox;

                    if (ws.A[1][p] <= 0) continue;

                    float g = dC2[p];
                    a.C2ToQ[j] += g * ws.A[1][p];                   // conv2 unit -> Q: activation x gradient

                    for (int i = 0; i < Spec.C1; i++)
                    {
                        for (int ky = 0; ky < k2; ky++)
                        {
                            for (int kx = 0; kx < k2; kx++)
                            {
                                float w = weights[c2.Offset + ((j * Spec.C1 + i) * k2 + ky) * k2 + kx];
                                float x = ws.A[0][(i * h1 + oy * s2 + ky) * w1 + ox * s2 + kx];

                                a.C1ToC2[i][j] += g * w * x;
                            }
                        }
                    }
                }
            }
        }

        for (int p = 0; p < a.Saliency.Length; p++)
        {
            a.Saliency[p] = dIn[p] * ws.Input[p];
        }

        return a;
    }

    /// <summary>
    /// Creates new float(r,c).
    /// </summary>
    /// <param name="rows"></param>
    /// <param name="cols"></param>
    /// <returns></returns>
    private static float[][] NewGrid(int rows, int cols)
    {
        var g = new float[rows][];

        for (int r = 0; r < rows; r++)
        {
            g[r] = new float[cols];
        }

        return g;
    }

    /// <summary>
    /// Activations of conv2 for the viewer: [C2][H*W] for the state in ws.Input (call after Forward).
    /// </summary>
    public (float[][] maps, int h, int w) Conv2Maps(Workspace ws)
    {
        var c = (Conv2D)_layers[1];
        var maps = new float[c.OutputChannels][];
        int n = c.OutputHeight * c.OutputWidth;

        for (int i = 0; i < maps.Length; i++) { maps[i] = new float[n]; Array.Copy(ws.A[1], i * n, maps[i], 0, n); }

        return (maps, c.OutputHeight, c.OutputWidth);
    }
}