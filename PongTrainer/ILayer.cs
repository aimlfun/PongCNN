namespace BrainPong;

// ---------------------------------------------------------------------------------------------
// The two building blocks of the Q-network, written out by hand so every step of the chain rule
// is visible. All parameters of the whole network live in ONE flat float[] ("theta"/"weights"); a layer is
// just a view (offsets) into it. A matching flat array "grad" holds dLoss/dTheta.
//
// Each layer does   y = relu( W * x + b )   (the last layer skips the relu)
//   Forward : compute y from x.
//   Backward: given dy = dLoss/dy (what the layers above handed down), produce
//               dLoss/dW  and dLoss/db  (added into grad)   <- what learning needs
//               dLoss/dx                (written to dx)     <- what the layer BELOW needs
//             That last line IS the chain rule: dLoss/dx = (dy/dx)^T * dLoss/dy.
// ---------------------------------------------------------------------------------------------
internal interface ILayer
{
    int InSize { get; }
    int OutSize { get; }
    int ParamCount { get; }
    int Offset { get; set; }                       // where this layer's parameters start in theta
    void Init(float[] weights, Random rng);

    /// <summary>
    /// Calculate NN output with forward pass.
    /// </summary>
    /// <param name="weights">Weights</param>
    /// <param name="inputs">Inputs</param>
    /// <param name="outputs">Outputs</param>
    void Forward(float[] weights, float[] inputs, float[] outputs);

    /// <summary>
    /// Backpropagate NN.
    /// </summary>
    /// <param name="weights"></param>
    /// <param name="grad"></param>
    /// <param name="inputs"></param>
    /// <param name="outputs"></param>
    /// <param name="dy"></param>
    /// <param name="dx"></param>
    void Backward(float[] weights, float[] grad, float[] inputs, float[] outputs, float[] dy, float[]? dx);
}
