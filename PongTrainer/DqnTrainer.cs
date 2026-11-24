namespace BrainPong;

/// <summary>
/// DQN, exactly the 2015 recipe at small scale:
///   act  : epsilon-greedy on Q(state)
///   store: every step into the replay buffer
///   learn: every few steps take a random minibatch, regress Q(s,a) toward  y = r + gamma * max_a' Qtarget(s',a')
///   sync : every few thousand steps copy the online weights into the frozen target net
/// </summary>
internal sealed class DqnTrainer
{
    public readonly QNet Net;
    public readonly DqnConfig Cfg;

    /// <summary>
    /// Online weights (being trained).
    /// </summary>
    public float[] Weights;

    /// <summary>
    /// Frozen copy used only to compute the regression targets.
    /// </summary>
    public float[] TargetWeights;

    /// <summary>
    /// Adam optimiser state, one entry per weight: _adamM = running average of the gradient (momentum),
    /// </summary>
    private readonly float[] _adamM;
    
    /// <summary>
    /// _adamV = running average of the squared gradient (used to scale each weight's step). 
    /// </summary>
    private readonly float[] _adamV;

    /// <summary>
    /// _adamT counts updates.
    /// </summary>
    private int _adamT;

    /// <summary>
    /// Remembers past (state, action, reward, next state) steps so training can sample them in random order.
    /// </summary>
    private readonly ReplayBuffer _replayBuffer;

    /// <summary>
    /// Used for a pseudo random number generator.
    /// </summary>
    private readonly Random _rng;

    /// <summary>
    /// One workspace + gradient buffer per worker thread: a minibatch is split across cores.
    /// </summary>
    private readonly int _workers;

    // Scratch buffers for the three forward passes needed per sample: the online net on the current state (_wsOnline),
    // the target net on the next state (_wsTarget), and the online net on the next state, used only by Double DQN (_wsSelect).
    private readonly Workspace[] _wsOnline, _wsTarget, _wsSelect;

    /// <summary>
    /// Learning rate for the current step (it can decay over the run, see DqnConfig.LrEndFrac).
    /// </summary>
    private float _curLr;

    /// <summary>
    /// Each worker adds its share of the gradient into its own array; they are summed into _grad afterwards.
    /// </summary>
    private readonly float[][] _workerGrad;

    // Per-worker input buffers that a sampled (current state, next state) pair is copied into.
    private readonly float[][] _sBuf, _nBuf;

    /// <summary>
    /// The combined gradient for the whole minibatch.
    /// </summary>
    private readonly float[] _grad;
    private readonly int _frozen;                 // parameters [0, frozen) are never updated

    public DqnTrainer(NetSpec spec, DqnConfig cfg)
    {
        Cfg = cfg; Net = new QNet(spec);

        _rng = new Random(cfg.Seed); // we intentionally use a PRNG

        Weights = Net.NewParams(cfg.Seed);
        TargetWeights = (float[])Weights.Clone();

        _frozen = Net.ParamsUpTo(cfg.FreezeLayers);

        // settings for ADAM optimiser.
        _adamM = new float[Net.ParamCount];
        _adamV = new float[Net.ParamCount];
        _grad = new float[Net.ParamCount];

        _replayBuffer = new ReplayBuffer(cfg.ReplayCapacity, spec.Frames, spec.Size * spec.Size);

        // scale the workers according to the compute.
        _workers = Math.Max(1, Environment.ProcessorCount);

        _wsOnline = new Workspace[_workers];
        _wsTarget = new Workspace[_workers];
        _wsSelect = new Workspace[_workers];

        _curLr = cfg.Lr;
        _workerGrad = new float[_workers][]; // done in parallel using workers

        _sBuf = new float[_workers][];
        _nBuf = new float[_workers][];

        InitialiseWorkspaceForWorkers();
    }

    /// <summary>
    /// Initialises workspaces for the workers.
    /// </summary>
    private void InitialiseWorkspaceForWorkers()
    {
        for (int w = 0; w < _workers; w++)
        {
            _wsOnline[w] = Net.NewWorkspace();
            _wsTarget[w] = Net.NewWorkspace();
            _wsSelect[w] = Net.NewWorkspace();

            _workerGrad[w] = new float[Net.ParamCount];

            _sBuf[w] = new float[Net.InputSize]; 
            _nBuf[w] = new float[Net.InputSize];
        }
    }

    /// <summary>
    /// Put pretrained trunk weights (the first `trunk.Length` parameters) into the online and target nets.
    /// </summary>
    public void LoadTrunk(float[] trunk)
    {
        Array.Copy(trunk, Weights, trunk.Length); 
        Array.Copy(trunk, TargetWeights, trunk.Length);
    }

    /// <summary>
    /// Returns the index of the highest value in the array. Given the network's Q-values (one score per action),
    /// that index is the action the network thinks is best. Ties go to the lowest index.
    /// </summary>
    /// <param name="actions">The Q-value for each action (here: 0 = stay, 1 and 2 = move up/down).</param>
    /// <returns>The index of the best-scoring action.</returns>
    public static int Argmax(float[] actions)
    {
        int action = 0; // stay
     
        // pick the action with the *highest* score
        for (int i = 1; i < actions.Length; i++)
        {
            if (actions[i] > actions[action])
            {
                action = i;
            }
        }

        return action;
    }

    /// <summary>
    /// One minibatch gradient step. Returns (mean Huber loss, mean Q of the taken actions).
    /// </summary>
    private (float loss, float meanQ) LearnStep()
    {
        int B = Cfg.Batch;
        
        // one seed per sample, drawn on the main thread so the run is reproducible; workers then read the (unchanging) buffer
        var seeds = new int[B];

        for (int i = 0; i < B; i++)
        {
            seeds[i] = _rng.Next();
        }

        float[] lossSum = new float[_workers], qSum = new float[_workers];

        Parallel.For(0, _workers, w =>
        {
            Array.Clear(_workerGrad[w]);
            var dOut = new float[Net.Actions];

            for (int b = w; b < B; b += _workers)
            {
                // --- 1. get a random past step -------------------------------------------------
                float[] s = _sBuf[w];
                float[] n = _nBuf[w];

                bool ok = _replayBuffer.Sample(new Random(seeds[b]), s, n, Cfg.NStep, Cfg.Gamma, out int a, out float r, out bool terminal, out float disc);
                if (!ok) continue;

                // --- 2. the regression TARGET, from the FROZEN net (a constant: no gradient flows into it)
                // r is the (n-step) discounted reward sum; disc = gamma^n is how much the bootstrap value is worth after n steps
                float y = r;
                
                if (!terminal)
                {
                    Array.Copy(n, _wsTarget[w].Input, n.Length);
                    var qn = Net.Forward(TargetWeights, _wsTarget[w]);
                    int best = Argmax(qn);                                   // plain DQN: target net picks AND scores (max)
                    
                    if (Cfg.DoubleDqn)
                    {
                        Array.Copy(n, _wsSelect[w].Input, n.Length);
                        best = Argmax(Net.Forward(Weights, _wsSelect[w]));      // Double DQN: online net picks, target net scores
                    }

                    // qn comes from the target.
                    // best comes from Weights if DDQN, or qn if DQN.
                    
                    y += disc * qn[best];
                }

                // --- 3. the PREDICTION, from the online net, for the action actually taken ------
                Array.Copy(s, _wsOnline[w].Input, s.Length);

                var q = Net.Forward(Weights, _wsOnline[w]);
                float delta = q[a] - y;                                   // TD error (delta)
                float ad = Math.Abs(delta);                               // absolute delta
                
                // Huber loss is a regression loss function that acts like squared error for small residuals and
                // linear (absolute) error for large residuals, controlled by a threshold parameter δ. It combines
                // the smooth, differentiable behavior of Mean Squared Error (MSE) near zero with the outlier
                // resistance of Mean Absolute Error (MAE).
                lossSum[w] += ad <= 1 ? 0.5f * delta * delta : ad - 0.5f;   // Huber loss
                qSum[w] += q[a];

                // --- 4. dLoss/dQ: Huber gradient (= delta clipped to [-1,1]), only on the taken action's output.
                // The other two outputs get 0: this sample says nothing about them. Divided by B for the batch mean.
                Array.Clear(dOut);
                dOut[a] = Math.Clamp(delta, -1f, 1f) / B; // delta is -1/B..+1/B

                // --- 5. the chain rule, from that single output back to every weight ------------
                Net.Backward(Weights, _workerGrad[w], _wsOnline[w], dOut, Cfg.FreezeLayers);
            }
        });

        Array.Clear(_grad);

        for (int w = 0; w < _workers; w++)
        {
            var g = _workerGrad[w];
            for (int i = 0; i < _grad.Length; i++)
            {
                _grad[i] += g[i];
            }
        }
        
        Adam();

        return (lossSum.Sum() / B, qSum.Sum() / B);
    }

    /// <summary>
    /// Adam: SGD with a per-weight step size. Same job as the RMSProp DeepMind used.
    /// Weights with consistently large gradients take smaller steps, rarely-updated ones take larger steps.
    /// Reads the batch gradient from _grad and nudges Weights; frozen layers are skipped.
    /// </summary>
    private void Adam()
    {
        _adamT++;

        // b1/b2: how slowly the two running averages forget (standard Adam values). eps avoids dividing by zero.
        const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;

        // Bias correction: the running averages start at zero, so early on they are too small. Dividing by
        // (1 - b^t) cancels that; the correction fades to 1 as training goes on.
        float c1 = 1 - MathF.Pow(b1, _adamT);
        float c2 = 1 - MathF.Pow(b2, _adamT);

        // Start at _frozen: parameters before it belong to frozen layers and are never changed.
        for (int i = _frozen; i < Weights.Length; i++)
        {
            float g = _grad[i];

            _adamM[i] = b1 * _adamM[i] + (1 - b1) * g;
            _adamV[i] = b2 * _adamV[i] + (1 - b2) * g * g;

            // step = learning rate * (smoothed gradient) / (typical gradient size)
            Weights[i] -=_curLr * (_adamM[i] / c1) / (MathF.Sqrt(_adamV[i] / c2) + eps);
        }
    }

    /// <summary>
    /// Run the whole training. onStats is called every LogEvery steps; return true from it to stop.
    /// </summary>
    public void Train(Func<TrainStats, bool> onStats)
    {
        var env = new PongEnv(Cfg.Seed * 7919) { ContinueOnTimeout = Cfg.TruncateTimeouts };
        int timeoutsWin = 0;
        var ws = Net.NewWorkspace();

        // area = pixels in one frame; F = how many frames are stacked as the network's input.
        // A single frame can't show which way the ball is moving, so the network is given the last F frames.
        int area = PongEnv.Size * PongEnv.Size, F = Net.Spec.Frames;

        // hist[0] is the oldest frame, hist[F-1] the newest.
        var hist = new byte[F][]; for (int i = 0; i < F; i++) hist[i] = new byte[area];

        // Start a new point: reset the game and fill the whole history with the first frame (there is no past yet).
        void NewPoint()
        {
            var f = env.Reset();

            for (int i = 0; i < F; i++)
            {
                Array.Copy(f, hist[i], area);
            }
        }

        // Add the newest frame to the history and drop the oldest. Shifts the references down by one and
        // recycles the oldest array for the new frame, so no memory is allocated per step.
        void Push(byte[] f)
        {
            var old = hist[0];
            
            for (int i = 0; i < F - 1; i++)
                hist[i] = hist[i + 1];
            
            hist[F - 1] = old;

            Array.Copy(f, old, area);
        }

        // Copy the F stacked frames, oldest first, into the network's input array.
        void FillInput()
        {
            for (int i = 0; i < F; i++)
            {
                for (int p = 0; p < area; p++)
                {
                    ws.Input[i * area + p] = hist[i][p];
                }
            }
        }

        NewPoint();
        
        bool isStart = true;                                // true for the first step of a point (stored in the buffer)
        var recent = new Queue<(float r, int len)>();       // result + length of the last 300 finished points, for the win rate
        float lossAcc = 0, qAcc = 0; int learnN = 0, points = 0;   // running totals, averaged and reset at each log
        var clock = System.Diagnostics.Stopwatch.StartNew();
        int lastLogStep = 0; double lastLogT = 0;           // used to work out steps per second

        // The main loop. Each iteration is one game step: choose an action, play it, remember it, maybe learn.
        for (int step = 1; step <= Cfg.TotalSteps; step++)
        {
            // Linearly shrink the learning rate over the run (no change when LrEndFrac = 1).
            _curLr =Cfg.Lr * (1 - (1 - Cfg.LrEndFrac) * step / Cfg.TotalSteps);

            // Epsilon-greedy: with probability eps take a random move (explore), otherwise the network's best move (exploit).
            // eps falls linearly from EpsStart to EpsEnd over EpsDecaySteps, then stays at EpsEnd.
            float eps = Math.Max(Cfg.EpsEnd, Cfg.EpsStart - (Cfg.EpsStart - Cfg.EpsEnd) * step / Cfg.EpsDecaySteps); 
            int a;

            if (_rng.NextDouble() < eps)
            {
                a = _rng.Next(Net.Actions); // random action
            }
            else
            {
                FillInput(); // provide the AI the data
                a = Argmax(Net.Forward(Weights, ws)); // ask the NN what action
            }

            var cur = (byte[])hist[F - 1].Clone(); // the frame the decision was based on
            int ageInPoint = env.PointSteps;

            // ball moves, paddle moves based on the action
            // r = reward (+1 won the point, -1 lost it, 0 otherwise); done = the point has ended.
            var next = env.Step(a, out float r, out bool done);

            // Remember what happened. The buffer stores the frame the decision was made on, not the next one,
            // and rebuilds the "next state" later from the following entry.
            _replayBuffer.Add(cur, a, r, done, isStart, ageInPoint);
            isStart = false;
            
            if (env.TimedOut) timeoutsWin++; // it enjoyed batting the ball, but lacked the desire/skill to"win" 

            if (done)
            {
                if (r != 0)
                {
                    recent.Enqueue((r, env.PointSteps)); 
                    points++; 
                    
                    if (recent.Count > 300) recent.Dequeue();
                }

                NewPoint(); isStart = true;
            }
            else
            {
                Push(next);
            }

            // Learn: once the buffer has enough data, do one gradient update every TrainEvery steps.
            if (step >= Cfg.LearnStart && step % Cfg.TrainEvery == 0)
            {
                var (l, q) = LearnStep(); 
                lossAcc += l; 
                qAcc += q; 
                learnN++;
            }

            if (step % Cfg.TargetSync == 0) Array.Copy(Weights, TargetWeights, Weights.Length);   // refresh the frozen target net

            if (step % Cfg.LogEvery == 0)
            {
                double t = clock.Elapsed.TotalSeconds;
                float win = recent.Count == 0 ? 0 : recent.Count(e => e.r > 0) / (float)recent.Count;
                float len = recent.Count == 0 ? 0 : (float)recent.Average(e => e.len);

                var st = new TrainStats(
                    Step: step, 
                    Eps: eps, 
                    Loss: learnN == 0 ? 0 : lossAcc / learnN, MeanQ: learnN == 0 ? 0 : qAcc / learnN, 
                    WinRate: win, 
                    AvgRally: len, 
                    Points: points,
                    StepsPerSec: (step - lastLogStep) / Math.Max(1e-9, t - lastLogT), 
                    BufReward: _replayBuffer.RewardFrac, 
                    BufLong: _replayBuffer.LongFrac, 
                    Timeouts: timeoutsWin);

                timeoutsWin = 0; 
                lossAcc = qAcc = 0; 
                learnN = 0; 
                lastLogStep = step; 
                lastLogT = t;

                if (onStats(st)) return;
            }
        }
    }

    /// <summary>
    /// Play `points` points greedily (tiny epsilon) with the given weights. Returns (point win rate, mean rally length).
    /// </summary>
    /// <param name="net">The neural network to evaluate.</param>
    /// <param name="theta">The weights for the neural network.</param>
    /// <param name="points">The number of points to play.</param>
    /// <param name="seed">The random seed for reproducibility.</param>
    /// <param name="eps">The epsilon value for epsilon-greedy action selection.</param>
    /// <returns>A tuple containing the win rate, mean rally length, and timeout rate.</returns>
    public static (float win, float rally, float timeouts) Evaluate(QNet net, float[] theta, int points, int seed, float eps = 0.02f)
    {
        var env = new PongEnv(seed); 
        var rng = new Random(seed + 1); 
        var ws = net.NewWorkspace();

        int area = PongEnv.Size * PongEnv.Size, F = net.Spec.Frames;
        
        var hist = new byte[F][];
        for (int i = 0; i < F; i++)
        {
            hist[i] = new byte[area];
        }

        int won = 0, lost = 0, lenSum = 0, n = 0, timeouts = 0;
        
        while (n < points)
        {
            var f = env.Reset();
            
            for (int i = 0; i < F; i++)
            {
                Array.Copy(f, hist[i], area);
            }

            bool done = false;
        
            while (!done)
            {
                for (int i = 0; i < F; i++)
                {
                    for (int p = 0; p < area; p++)
                    {
                        ws.Input[i * area + p] = hist[i][p]; // fill the workspace input with the last F frames
                    }
                }

                // pick an action, random or from the nn
                int action = rng.NextDouble() < eps ? rng.Next(net.Actions) : Argmax(net.Forward(theta, ws));
                var nf = env.Step(action, out float r, out done);
                
                var old = hist[0];

                for (int i = 0; i < F - 1; i++)  // shift the history down
                {
                    hist[i] = hist[i + 1];
                }
                
                hist[F - 1] = old; // reuse the oldest array to avoid allocations

                Array.Copy(nf, old, area);

                if (done)
                {
                    if (r > 0)
                        won++;
                    else if (r < 0)
                        lost++;

                    if (env.TimedOut) timeouts++; 
                    
                    lenSum += env.Steps; 
                    n++;
                }
            }
        }

        return (won / (float)Math.Max(1, won + lost), lenSum / (float)n, timeouts / (float)n);
    }
}
