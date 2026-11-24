namespace BrainPong;

/// <summary>
/// Learn the trunk WITHOUT any reward. The network sees the usual 4-frame stack and must predict the NEXT frame,
/// one prediction per action (same trick as the Q head: three outputs, the loss only touches the action actually taken).
/// To predict where the ball will be, the trunk has to represent where it is and where it is heading; nothing tells it
/// that: the pressure comes only from "make the next picture right". The trunk (conv1, conv2, dense) has exactly the
/// layout of the Q network, so its weights can be copied straight into a DQN.
/// </summary>
internal sealed class PredictTrainer
{
    /// <summary>
    /// How many layers make up the "trunk" (conv1, conv2, dense). These are the layers shared with the DQN network;
    /// everything after them is the output "head", which is different for prediction (pixels) and DQN (Q-values).
    /// </summary>
    public const int TrunkLayers = 3;

    public readonly QNet Net;
    public readonly PretrainConfig Cfg;

    /// <summary>
    /// All the network's weights (theta is the conventional maths name for them).
    /// </summary>
    public float[] Weights;

    private readonly int area = PongEnv.Size * PongEnv.Size;    // pixels in one frame

    private readonly Random rng;

    // Adam optimiser state (running averages of the gradient and squared gradient, update count) and the
    // gradient summed over the current minibatch. Same scheme as DqnTrainer.Adam.
    private readonly float[] adamM, adamV, grad;
    private int adamT;

    /// <summary>How many of the weights belong to the trunk. Only these are saved to pong_enc.bin and reused by the DQN.</summary>
    public int TrunkParams => Net.ParamsUpTo(TrunkLayers);

    public PredictTrainer(PretrainConfig cfg)
    {
        Cfg = cfg; rng = new Random(cfg.Seed);
        var spec = new NetSpec();
        
        Net = new QNet(spec with { Actions = spec.Actions * area });     // head = 3 * 1600 outputs instead of 3 Q-values
        
        Weights = Net.NewParams(cfg.Seed);
        
        adamM = new float[Net.ParamCount]; adamV = new float[Net.ParamCount]; grad = new float[Net.ParamCount];
    }

    /// <summary>
    /// Random play with STICKY actions (held for 1-10 steps) so the paddle visits the edges 
    /// too; uniform random keeps it near the middle.
    /// </summary>
    private static ReplayBuffer Collect(int steps, int seed, int frames)
    {
        var env = new PongEnv(seed * 104729);
        var r = new Random(seed + 17);

        // Sized to hold every step, so nothing is overwritten. Reuses the DQN's replay buffer for storage and sampling.
        var buf = new ReplayBuffer(steps, frames, PongEnv.Size * PongEnv.Size);
        var cur = (byte[])env.Reset().Clone();

        bool isStart = true;
        int hold = 0, a = 0;   // hold = how many more steps to repeat the current action; a = the current action

        for (int i = 0; i < steps; i++)
        {
            // Pick a new random action only when the previous one has been held long enough.
            if (hold-- <= 0)
            {
                a = r.Next(3); 
                hold = r.Next(1, 11);
            }
            
            int age = env.PointSteps;
            var next = env.Step(a, out float rew, out bool done);
        
            buf.Add(cur, a, rew, done, isStart, age);
            isStart = false;

            if (done)
            {
                cur = (byte[])env.Reset().Clone();
                isStart = true;
            }
            else
            {
                cur = (byte[])next.Clone();
            }
        }

        return buf;
    }

    /// <summary>
    /// Weighted squared error of one prediction against the true next frame;
    /// fills dOut (only the slice of the taken action) when asked.
    /// </summary>
    /// <param name="pred">The network's output: one full predicted frame per action, laid end to end.</param>
    /// <param name="a">The action that was actually taken; only that action's predicted frame is scored.</param>
    /// <param name="lastIn">The most recent frame the network was shown (used to find where things just were).</param>
    /// <param name="truth">The real next frame.</param>
    /// <param name="dOut">If not null, receives the gradient of the loss with respect to the output (needed for backprop).</param>
    /// <param name="scale">Multiplier applied to the gradient; 1/batch size, so the batch gradients average rather than add up.</param>
    /// <returns>The loss for this sample.</returns>
    private float PixelLoss(float[] pred, int a, float[] lastIn, float[] truth, float[]? dOut, float scale)
    {
        float loss = 0; int o = a * area;    // o = where the taken action's predicted frame starts in the output

        for (int p = 0; p < area; p++)
        {
            // Pixels with something on them (now or in the last frame) matter ActiveWeight times more than blank background.
            float w = (truth[p] > 0 || lastIn[p] > 0) ? Cfg.ActiveWeight : 1f;
            float d = pred[o + p] - truth[p];
            loss += 0.5f * w * d * d;

            dOut?[o + p] = w * d * scale;
        }

        return loss;
    }

    /// <summary>
    /// Train the trunk to predict the next frame from the current stack of frames, using a random-play dataset.
    /// </summary>
    /// <param name="onStats">Called every LogEvery batches with a progress report; return true to stop training early.</param>
    public void Train(Func<PretrainStats, bool> onStats)
    {
        // F = frames per input stack; last = offset of the newest frame inside the flattened input array.
        int F = Net.Spec.Frames, last = (F - 1) * area;

        // Two separate datasets from different seeds: one to train on, one held back purely to measure progress.
        var train = Collect(Cfg.CollectSteps, Cfg.Seed, F);
        var valBuf = Collect(20_000, Cfg.Seed + 1000, F);

        // As in DqnTrainer: a minibatch is split across CPU cores, each with its own scratch space and gradient array.
        int workers = Math.Max(1, Math.Min(Environment.ProcessorCount, 8)), B = Cfg.Batch;
        var ws = new Workspace[workers]; 
        var wg = new float[workers][]; 
        var sB = new float[workers][]; 
        var nB = new float[workers][];
        
        for (int w = 0; w < workers; w++) { 
            ws[w] = Net.NewWorkspace(); 
            wg[w] = new float[Net.ParamCount]; sB[w] = new float[Net.InputSize]; nB[w] = new float[Net.InputSize]; }

        // fixed validation set: loss of the model vs. the lazy answer "the next frame equals the last one"
        (float model, float copy) Validate()
        {
            var r = new Random(7); 
            float m = 0, c = 0; 
            int n = 0;
            var s = sB[0]; var nx = nB[0]; var next = new float[area]; var lastIn = new float[area];
            
            while (n < 500)
            {
                if (!valBuf.Sample(r, s, nx, out int a, out _, out bool done) || done) continue;

                Array.Copy(s, ws[0].Input, s.Length);
                var o = Net.Forward(Weights, ws[0]);
                Array.Copy(nx, last, next, 0, area); Array.Copy(s, last, lastIn, 0, area);
                m += PixelLoss(o, a, lastIn, next, null, 0);
                
                // copy baseline: predict the last frame itself, same weighting
                float cl = 0;

                for (int p = 0; p < area; p++)
                {
                    float w = (next[p] > 0 || lastIn[p] > 0) ? Cfg.ActiveWeight : 1f; 
                    float d = lastIn[p] - next[p]; 
                    
                    cl += 0.5f * w * d * d;
                }

                c += cl; n++;
            }

            return (m / n, c / n);
        }

        float lossAcc = 0; int lossN = 0;   // running training loss and sample count, averaged and reset at each log

        // One iteration = one minibatch = one weight update.
        for (int batch = 1; batch <= Cfg.Batches; batch++)
        {
            // One seed per sample, drawn on the main thread so the run is reproducible whatever the thread timing.
            var seeds = new int[B]; for (int i = 0; i < B; i++) seeds[i] = rng.Next();
            var ls = new float[workers]; var cnt = new int[workers];
            
            Parallel.For(0, workers, w =>
            {
                Array.Clear(wg[w]);
                var dOut = new float[Net.Actions]; var next = new float[area]; var lastIn = new float[area];
        
                for (int b = w; b < B; b += workers)
                {
                    if (!train.Sample(new Random(seeds[b]), sB[w], nB[w], out int a, out _, out bool done) || done) continue;
                
                    Array.Copy(sB[w], ws[w].Input, sB[w].Length);
            
                    var o = Net.Forward(Weights, ws[w]);
                    
                    Array.Copy(nB[w], last, next, 0, area); Array.Copy(sB[w], last, lastIn, 0, area);
                    Array.Clear(dOut);
                    
                    ls[w] += PixelLoss(o, a, lastIn, next, dOut, 1f / B);       // dLoss/dOutput for the 1600 pixels of the taken action
                    cnt[w]++;
                    
                    Net.Backward(Weights, wg[w], ws[w], dOut);                    // same chain rule as the DQN, just a wider head
                }
            });

            // Add up the gradients from all workers into one, then take a single optimiser step.
            Array.Clear(grad);

            for (int w = 0; w < workers; w++)
            {
                for (int i = 0; i < grad.Length; i++)
                {
                    grad[i] += wg[w][i];
                }
            }

            Adam();
            lossAcc += ls.Sum(); lossN += cnt.Sum();

            if (batch % Cfg.LogEvery == 0)
            {
                var (vm, vc) = Validate();
                var st = new PretrainStats(batch, lossN == 0 ? 0 : lossAcc / lossN, vm, vc);
                
                lossAcc = 0; 
                lossN = 0;

                if (onStats(st)) return;
            }
        }
    }

    /// <summary>
    /// Adjust the parameters using Adam. The gradients are assumed to be accumulated in grad[].
    /// See DqnTrainer.Adam for a step-by-step explanation; this is the same, but updates every weight (nothing is frozen).
    /// </summary>
    private void Adam()
    {
        adamT++;
        const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        float c1 = 1 - MathF.Pow(b1, adamT), c2 = 1 - MathF.Pow(b2, adamT);
        
        for (int i = 0; i < Weights.Length; i++)
        {
            float g = grad[i];

            adamM[i] = b1 * adamM[i] + (1 - b1) * g;
            adamV[i] = b2 * adamV[i] + (1 - b2) * g * g;
            Weights[i] -= Cfg.Lr * (adamM[i] / c1) / (MathF.Sqrt(adamV[i] / c2) + eps);
        }
    }
}
