namespace BrainPong;

/// <summary>
/// Fully connected layer: every output is a weighted sum of every input.
/// </summary>
internal sealed class Dense(int inSize, int outSize, bool relu) : ILayer
{
    public int InSize => inSize;
    public int OutSize => outSize;
    public int ParamCount => outSize * inSize + outSize;      // weights then biases
    public int Offset { get; set; }

    public void Init(float[] weights, Random rng)
    {
        WeightInit.He(weights, Offset, outSize * inSize, inSize, rng);
        Array.Clear(weights, Offset + outSize * inSize, outSize);
    }

    // Terminology
    // ~~~~~~~~~~~
    // theta = weights
    // x     = inputs
    // y     = outputs

    /// <summary>
    /// Forward pass: compute the output from the inputs. output = relu( weight * input + b ) or just weight * input + b if relu is false.
    /// </summary>
    /// <param name="theta">weight</param>
    /// <param name="inputs">input</param>
    /// <param name="outputs">output</param>
    public void Forward(float[] weights, float[] inputs, float[] outputs)
    {
        // the data is in a wide array. We use the offset to access it.
        int biasOffset = Offset + outSize * inSize;

        for (int outputIndex = 0; outputIndex < outSize; outputIndex++)
        {
            float s = weights[biasOffset + outputIndex]; // the bias (rather than 0, and add bias)

            int weightIndex = Offset + outputIndex * inSize;

            for (int inputIndex = 0; inputIndex < inSize; inputIndex++)
            {
                // s = w * x
                s += weights[weightIndex + inputIndex] * inputs[inputIndex];
            }

            // y = relu(s) | y = s
            outputs[outputIndex] = relu && s < 0 ? 0 : s;
        }
    }

    /// <summary>
    /// Backward pass: compute the gradients of the loss with respect to the weights, biases, and input.
    /// </summary>
    /// <param name="weights">weights</param>
    /// <param name="grad"></param>
    /// <param name="inputs">inputs</param>
    /// <param name="outputs">outputs</param>
    /// <param name="dy"></param>
    /// <param name="dx"></param>
    public void Backward(float[] weights, float[] grad, float[] inputs, float[] outputs, float[] dy, float[]? dx)
    {
        int biasOffset = Offset + outSize * inSize;
        
        if (dx != null) Array.Clear(dx, 0, inSize);
        
        for (int outputIndex = 0; outputIndex < outSize; outputIndex++)
        {
            // relu gate: where the unit was off (y == 0) no gradient flows through it
            float g = relu && outputs[outputIndex] <= 0 ? 0 : dy[outputIndex];
            
            if (g == 0) continue; // if 0, it won't change the assignments that follow
            
            grad[biasOffset + outputIndex] += g;                // dL/db = g
        
            int weightIndex = Offset + outputIndex * inSize;
            
            for (int inputIndex = 0; inputIndex < inSize; inputIndex++)
            {
                grad[weightIndex + inputIndex] += g * inputs[inputIndex];   // dL/dW[o,i] = g * inputs[i]
                dx?[inputIndex] += g * weights[weightIndex + inputIndex];   // dL/dx[i]  += g * W[o,i]
            }
        }
    }
}
