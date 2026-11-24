namespace BrainPong;

/// <summary>
/// Proves the hand-written backward pass is right: for random weights and a random state, compare dLoss/dTheta from
/// Backward() against the numerical slope (L(theta+h) - L(theta-h)) / 2h, one weight at a time.
/// </summary>
/// <remarks>
/// Why this exists: backpropagation is easy to get subtly wrong, and a wrong gradient doesn't crash, it just trains
/// badly. This test uses a tiny network so it runs in moments. Run it with <c>BrainPong gradcheck</c> after any change
/// to Layers.cs or QNet.cs.
/// </remarks>
internal static class GradCheck
{
    /// <summary>
    /// Runs the check and prints the result. Returns true if the analytic and numerical gradients agree.
    /// </summary>
    public static bool Run()
    {
        // A deliberately small network (12x12 input, 2 frames, few filters): the check nudges every weight twice.
        var spec = new NetSpec(Size: 12, Frames: 2, C1: 3, KernelSize1: 4, S1: 2, C2: 4, KernelSize2: 3, S2: 1, Fc: 10, Actions: 3);
        var net = new QNet(spec);
        var weights = net.NewParams(3);
        
        var rng = new Random(4); // intentionally pseudo-random with chosen seed for reproducibility
        
        // small nonzero biases and a dense random input so every relu has a chance to be both on and off
        for (int i = 0; i < weights.Length; i++) weights[i] += (float)(rng.NextDouble() - 0.5) * 0.05f;
        
        var ws = net.NewWorkspace();
        
        for (int i = 0; i < ws.Input.Length; i++) ws.Input[i] = (float)rng.NextDouble();
        
        // Pretend the network was asked for action 1's Q-value and the "right answer" was 0.7.
        const int action = 1; 
        const float target = 0.7f;

        // The loss being differentiated: half the squared error between Q(action) and the target.
        double Loss(float[] th)
        {
            var q = net.Forward(th, ws);
            double d = q[action] - target;
            return 0.5 * d * d;
        }

        // analytic gradient through our Backward
        var q0 = net.Forward(weights, ws);

        var dOut = new float[3];
        dOut[action] = q0[action] - target;      // d(0.5 d^2)/dq = d
        
        var grad = new float[weights.Length];
        
        net.Backward(weights, grad, ws, dOut);

        // Numerical gradient: for each weight, nudge it up and then down by h and see how the loss changes.
        // The slope (lp - lm) / 2h is the gradient. It's slow, which is why we only use it as a test.
        const float h = 1e-2f;
        double worst = 0; int checkedN = 0, bad = 0;

        for (int i = 0; i < weights.Length; i++)
        {
            float keep = weights[i];

            // upper
            weights[i] = keep + h; 
            double lp = Loss(weights);

            // lower
            weights[i] = keep - h; 
            double lm = Loss(weights);

            weights[i] = keep;
            double num = (lp - lm) / (2 * h), ana = grad[i];
            
            if (Math.Abs(num) < 1e-6 && Math.Abs(ana) < 1e-6) continue; // dead weight, both zero

            // Relative error between the two gradients; a few % is expected from float rounding and the finite step h.
            double rel = Math.Abs(num - ana) / Math.Max(1e-4, Math.Abs(num) + Math.Abs(ana));
            checkedN++; 
            
            worst = Math.Max(worst, rel);
        
            if (rel > 2e-2) bad++;
        }
        
        Console.WriteLine($"gradient check: {weights.Length} weights, {checkedN} with signal, worst relative error {worst:E2}, {bad} above 2e-2");
        Console.WriteLine(bad == 0 ? "PASS: backprop matches the numerical gradient" : "FAIL");
        return bad == 0;
    }
}
