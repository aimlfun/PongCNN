namespace BrainPong;

/// <summary>
/// 2-D convolution, no padding. Input is [Cin, H, W], kernel [Cout, Cin, K, K], output [Cout, Ho, Wo].
/// One output pixel = a K x K x Cin patch of the input dotted with one filter. The SAME filter weights are
/// reused at every position, so the gradient for a weight is the SUM of its gradient over all positions.
/// </summary>
/// <remarks>
/// Constructor. 
/// The input is [inputChannels][inputHeight][inputWidth], 
/// The kernel is [outputChannels][inputChannels][kernelSize][kernelSize]
/// The output is [outputChannels][Ho][Wo], where Ho = (inputHeight-KernelSize)/ stride +1, and similar for Wo...
/// </remarks>
/// <param name="inputChannels">Number of input channels</param>
/// <param name="inputHeight">Height of the input</param>
/// <param name="inputWidth">Width of the input</param>
/// <param name="outputChannels">Number of output channels</param>
/// <param name="kernelSize">Kernel size</param>
/// <param name="stride">Stride</param>
/// <param name="relu">Whether to apply ReLU activation</param>
internal sealed class Conv2D(int inputChannels, int inputHeight, int inputWidth, int outputChannels, int kernelSize, int stride, bool relu) : ILayer
{
    private readonly int _inputChannelCount = inputChannels, _inputHeight = inputHeight, _inputWidth = inputWidth, _outputChannelCount = outputChannels, _kernelSize = kernelSize, _stride = stride, _outputHeight = (inputHeight - kernelSize) / stride + 1, _outputWidth = (inputWidth - kernelSize) / stride + 1;

    /// <summary>
    /// Indicates whether to apply ReLU activation after the convolution. If true, negative values are set to zero.
    /// </summary>
    private readonly bool _relu = relu;

    /// <summary>
    /// Size of input (channel x width x height).
    /// </summary>
    public int InSize => _inputChannelCount * _inputHeight * _inputWidth;

    /// <summary>
    /// Size of output (channnel x width x height).
    /// </summary>
    public int OutSize => _outputChannelCount * _outputHeight * _outputWidth;

    /// <summary>
    /// Number of parameters.
    /// </summary>
    public int ParamCount => _outputChannelCount * _inputChannelCount * _kernelSize * _kernelSize + _outputChannelCount;

    /// <summary>
    /// 
    /// </summary>
    public int Offset { get; set; }

    /// <summary>
    /// Number of output channels.
    /// </summary>
    public int OutputChannels => _outputChannelCount;

    /// <summary>
    /// Size of output height.
    /// </summary>
    public int OutputHeight => _outputHeight;

    /// <summary>
    /// Size of output width.
    /// </summary>
    public int OutputWidth => _outputWidth;

    /// <summary>
    /// Initialises a new theta array with random weights. The same code runs the online net,
    /// </summary>
    /// <param name="weights"></param>
    /// <param name="rng"></param>
    public void Init(float[] weights, Random rng)
    {
        int n = _outputChannelCount * _inputChannelCount * _kernelSize * _kernelSize;

        WeightInit.He(weights, Offset, n, _inputChannelCount * _kernelSize * _kernelSize, rng);

        Array.Clear(weights, Offset + n, _outputChannelCount);
    }

    /// <summary>
    /// Forward pass: compute the output y from the input x. 
    /// The input is [Cin, H, W], the kernel is [Cout, Cin, K, K], the output is [Cout, Ho, Wo].
    /// </summary>
    /// <param name="weights"></param>
    /// <param name="inputs"></param>
    /// <param name="outputs"></param>
    public void Forward(float[] weights, float[] inputs, float[] outputs)
    {
        int biaseOffset = Offset + _outputChannelCount * _inputChannelCount * _kernelSize * _kernelSize;

        for (int outputChannelIndex = 0; outputChannelIndex < _outputChannelCount; outputChannelIndex++)
        {
            for (int outputY = 0; outputY < _outputHeight; outputY++)
            {
                for (int outputX = 0; outputX < _outputWidth; outputX++)
                {
                    float sum = weights[biaseOffset + outputChannelIndex];

                    for (int inputChannelIndex = 0; inputChannelIndex < _inputChannelCount; inputChannelIndex++)
                    {
                        for (int kernelY = 0; kernelY < _kernelSize; kernelY++)
                        {
                            int weightIndex = Offset + ((outputChannelIndex * _inputChannelCount + inputChannelIndex) * _kernelSize + kernelY) * _kernelSize;
                            int inputIndex = (inputChannelIndex * _inputHeight + outputY * _stride + kernelY) * _inputWidth + outputX * _stride;

                            for (int kernelX = 0; kernelX < _kernelSize; kernelX++)
                            {
                                // sum += W[co, ci, ky, kx] * x[ci, oy*stride + ky, ox*stride + kx]
                                // W (weights) are stored in theta as a flat array, x is also a flat array.
                                sum += weights[weightIndex + kernelX] * inputs[inputIndex + kernelX];
                            }
                        }
                    }

                    // Apply ReLU activation if specified. If relu is true and the sum is negative, set the output to 0.
                    outputs[(outputChannelIndex * _outputHeight + outputY) * _outputWidth + outputX] = _relu && sum < 0 ? 0 : sum;
                }
            }
        }
    }

    /// <summary>
    /// Backward pass: compute the gradients of the loss with respect to the weights, biases, and input.
    /// </summary>
    /// <param name="weights"></param>
    /// <param name="grad"></param>
    /// <param name="inputs"></param>
    /// <param name="outputs"></param>
    /// <param name="dy"></param>
    /// <param name="dx"></param>
    public void Backward(float[] weights, float[] grad, float[] inputs, float[] outputs, float[] dy, float[]? dx)
    {
        int biasOffset = Offset + _outputChannelCount * _inputChannelCount * _kernelSize * _kernelSize;

        if (dx != null) Array.Clear(dx, 0, InSize);

        for (int outputChannelIndex = 0; outputChannelIndex < _outputChannelCount; outputChannelIndex++)
        {
            for (int outputY = 0; outputY < _outputHeight; outputY++)
            {
                for (int outputX = 0; outputX < _outputWidth; outputX++)
                {
                    int yIndex = (outputChannelIndex * _outputHeight + outputY) * _outputWidth + outputX;

                    float g = _relu && outputs[yIndex] <= 0 ? 0 : dy[yIndex];

                    if (g == 0) continue; // 0 means nothing to add to gradient, or dx.

                    grad[biasOffset + outputChannelIndex] += g;

                    for (int inputChannelIndex = 0; inputChannelIndex < _inputChannelCount; inputChannelIndex++)
                    {
                        for (int kernelY = 0; kernelY < _kernelSize; kernelY++)
                        {
                            int weightIndex = Offset + ((outputChannelIndex * _inputChannelCount + inputChannelIndex) * _kernelSize + kernelY) * _kernelSize;
                            int inputIndex = (inputChannelIndex * _inputHeight + outputY * _stride + kernelY) * _inputWidth + outputX * _stride;

                            for (int kernelX = 0; kernelX < _kernelSize; kernelX++)
                            {
                                grad[weightIndex + kernelX] += g * inputs[inputIndex + kernelX];  // weight gradient: summed over positions

                                dx?[inputIndex + kernelX] += g * weights[weightIndex + kernelX];  // pixel gradient: summed over filters/positions that saw it
                            }
                        }
                    }
                }
            }
        }
    }
}