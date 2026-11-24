using System.Globalization;

namespace BrainPong;

/*

 BrainPong: Pong from pixels with a hand-written CNN + DQN (the 2013/2015 DeepMind recipe), no ML library.

 BrainPong gradcheck                      numerically verify the hand-written backprop (run this first)
 BrainPong train [steps] [outDir] [seed]  train; writes pong_dqn.bin (weights) and pong_log.csv into outDir
 BrainPong eval [weights.bin] [points]    play greedy points against the built-in opponent
 BrainPong random [points]                baseline: a random player
 BrainPong pretrain [outDir] [seed] [batches] [collectSteps]   learn the trunk by predicting the next frame (no reward); writes pong_enc.bin
 
        train options: freeze=3 (trunk frozen, only the Q head learns), trunk=<pong_enc.bin> (start from a pretrained trunk)
 
The defaults are plain DQN (no double, no lr decay, 100k replay, no trunc, n-step 1). r7 comes from these switches (BrainPong/Program.cs:32-49):

The best is a 1.5M step run with double DQN, learning rate decay, 300k replay buffer, truncation of timeouts, and n-step=5. 

BrainPong.exe train 1500000 r7_nstep 2 double lrdecay buf=300000 trunc nstep=5 v2

The arguments, in order:
- train is the mode.
- 1500000 is the number of steps.
- r7_nstep is the output folder. It gets pong_dqn.bin, pong_log.csv, pong_eval.csv, pong_best.bin and the snapshots.
- 2 is the seed.
- double lrdecay buf=300000 trunc nstep=5 are the switches.
- v2 adds:
    - Dead zone: paddles cannot reach the top 3 px (Atari-style). Top only; the bottom is not restricted.
    - Ball speed: +3% per paddle hit with a cap of 3.0. v1 is +4% with a cap of 1.8.
    - Timeout: a timeout counts as -1 for the agent. In v1 it is 0.

- Time: expect about 2.5 hours (r7 took roughly that on my laptop). The seed is deterministic, so a rerun with seed 2 should reproduce r7's numbers.
- Checking the result: BrainPong.exe eval r7_nstep\pong_dqn.bin 500 should give about 89% win and 9% timeouts. You can watch it with BrainPongViewer.exe r7_nstep.
*/
internal static class Program
{
    /// <summary>
    /// Entry point. The first argument picks the mode (see the guide at the top of this file); the rest are that
    /// mode's options.
    /// </summary>
    /// <param name="args">Command-line arguments.</param>
    /// <returns>0 on success, 1 if the gradient check fails. This becomes the process exit code.</returns>
    private static int Main(string[] args)
    {
        string mode = args.Length > 0 ? args[0] : "help";

        switch (mode)
        {
            case "gradcheck": return GradCheck.Run() ? 0 : 1;
            case "train": Train(args); return 0;
            case "eval": Eval(args); return 0;
            case "pretrain": Pretrain(args); return 0;
            case "random": RandomBaseline(args.Length > 1 ? int.Parse(args[1]) : 500); return 0;
            default:
                Console.WriteLine("modes: gradcheck | train [steps] [outDir] [seed] | eval [weights.bin] [points] | random [points]");
                return 0;
        }
    }

    /// <summary>
    /// The "train" mode: parses the command line into a <see cref="DqnConfig"/>, builds the trainer, runs it and
    /// finishes with a greedy evaluation. Usage: train [steps] [outDir] [seed] [switches...]
    /// </summary>
    /// <param name="args">Command-line arguments (args[0] is "train").</param>
    private static void Train(string[] args)
    {
        int steps = args.Length > 1 ? int.Parse(args[1]) : 1_500_000;
        string dir = args.Length > 2 ? args[2] : ".";
        int seed = args.Length > 3 ? int.Parse(args[3]) : 1;

        Directory.CreateDirectory(dir);

        // optional switches after the seed:  double  lrdecay  buf=300000   (each changes ONE thing, so runs can be compared)
        var opts = args.Skip(4).ToHashSet();
        if (opts.Contains("v2")) PongEnv.GameVersion2 = true;

        File.WriteAllText(Path.Combine(dir, "pong_env.txt"), PongEnv.GameVersion2 ? "v2" : "v1");   // the viewer reads this so it plays by the same rules

        int buf = opts.Select(o => o.StartsWith("buf=") ? int.Parse(o[4..]) : 0).FirstOrDefault(v => v > 0, 100_000);
        var cfg = new DqnConfig
        {
            TotalSteps = steps,
            Seed = seed,
            EpsDecaySteps = Math.Min(300_000, steps / 3),
            DoubleDqn = opts.Contains("double"),
            LrEndFrac = opts.Contains("lrdecay") ? 0.1f : 1f,
            ReplayCapacity = buf,
            TruncateTimeouts = opts.Contains("trunc"),
            NStep = opts.Select(o => o.StartsWith("nstep=") ? int.Parse(o[6..]) : 0).FirstOrDefault(v => v > 0, 1),
            FreezeLayers = opts.Select(o => o.StartsWith("freeze=") ? int.Parse(o[7..]) : 0).FirstOrDefault(v => v > 0, 0)
        };

        string? trunkPath = opts.FirstOrDefault(o => o.StartsWith("trunk="))?[6..];

        Console.WriteLine($"double={cfg.DoubleDqn} lrEndFrac={cfg.LrEndFrac} replay={cfg.ReplayCapacity:N0} trunc={cfg.TruncateTimeouts} nstep={cfg.NStep}");

        float bestWin = -1; string evalCsv = Path.Combine(dir, "pong_eval.csv");
        File.WriteAllText(evalCsv, "step,greedyWin,rally\n");

        var tr = new DqnTrainer(new NetSpec(), cfg);

        if (trunkPath != null) tr.LoadTrunk(Weights.Load(trunkPath, tr.Net.ParamsUpTo(PredictTrainer.TrunkLayers)));

        string trunkLabel = trunkPath ?? (cfg.FreezeLayers > 0 ? "RANDOM (control)" : "-");
        Console.WriteLine($"freeze={cfg.FreezeLayers} trunk={trunkLabel}");
        Console.WriteLine($"params={tr.Net.ParamCount:N0}  steps={steps:N0}  workers={Environment.ProcessorCount}  seed={seed}");

        string csv = Path.Combine(dir, "pong_log.csv");
        File.WriteAllText(csv, "step,eps,loss,meanQ,winRate,avgRally,points,stepsPerSec,bufReward,bufLong,timeouts\n");
        Console.WriteLine("   step     eps    loss   meanQ  win%(last300 pts)  rally  steps/s  buf:reward% long%  timeouts");
        
        bestWin = TrainPongAI(dir, bestWin, evalCsv, tr, csv);

        var (win, rally, to) = DqnTrainer.Evaluate(tr.Net, tr.Weights, 500, 12345);

        Console.WriteLine($"final greedy eval over 500 points: win {win * 100:F1}%  mean rally {rally:F1} frames  timeouts {to * 100:F0}%");
    }

    /// <summary>
    /// Runs the training loop and handles everything that happens at each progress report: print a line, append to the
    /// CSV log, save the latest weights, keep periodic snapshots, and track the best greedy-evaluated weights.
    /// </summary>
    /// <param name="dir">Output folder.</param>
    /// <param name="bestWin">Best greedy win rate seen so far (-1 = none yet).</param>
    /// <param name="evalCsv">Path of the CSV that records the greedy evaluations.</param>
    /// <param name="tr">The trainer to run.</param>
    /// <param name="csv">Path of the CSV that records the training statistics.</param>
    /// <returns>The best greedy win rate reached during the run.</returns>
    private static float TrainPongAI(string dir, float bestWin, string evalCsv, DqnTrainer tr, string csv)
    {
        tr.Train(s =>
        {
            Console.WriteLine($"{s.Step,8} {s.Eps,7:F3} {s.Loss,7:F4} {s.MeanQ,7:F3} {s.WinRate * 100,12:F1}%      {s.AvgRally,6:F1} {s.StepsPerSec,7:F0}   {s.BufReward * 100,6:F2}% {s.BufLong * 100,5:F1}%  {s.Timeouts,6}");
            File.AppendAllText(csv, string.Create(CultureInfo.InvariantCulture,
                $"{s.Step},{s.Eps},{s.Loss},{s.MeanQ},{s.WinRate},{s.AvgRally},{s.Points},{s.StepsPerSec},{s.BufReward},{s.BufLong},{s.Timeouts}\n"));

            // Always keep the latest weights, so a run can be watched or resumed from the viewer while it trains.
            Weights.Save(Path.Combine(dir, "pong_dqn.bin"), tr.Weights);

            // save every 100k
            if (s.Step % 100_000 == 0)
            {
                Weights.Save(Path.Combine(dir, $"pong_dqn_{s.Step / 1000}k.bin"), tr.Weights);   // history, never overwritten
            }

            // every 50k steps, run a greedy evaluation
            if (s.Step % 50_000 == 0)
            {
                // the training win% is measured WITH exploration noise and on a moving window; this is a clean greedy test.
                var (gw, gr, gt) = DqnTrainer.Evaluate(tr.Net, tr.Weights, 150, 4242);

                Console.WriteLine($"         greedy eval (150 pts): win {gw * 100:F1}%  rally {gr:F0}  timeouts {gt * 100:F0}%");
                File.AppendAllText(evalCsv, string.Create(CultureInfo.InvariantCulture, $"{s.Step},{gw},{gr}\n"));

                // if it's better, save it.
                if (gw > bestWin)
                {
                    bestWin = gw; 
                    Weights.Save(Path.Combine(dir, "pong_best.bin"), tr.Weights); 
                    Console.WriteLine("         new best -> pong_best.bin");
                }
            }

            return false;   // false = carry on training (return true to stop early)
        });

        return bestWin;
    }

    /// <summary>
    /// The "pretrain" mode: teaches the network's feature-extracting layers to predict the next game frame, with no
    /// reward involved. The result (pong_enc.bin) can then be loaded into a DQN run with trunk=pong_enc.bin.
    /// Usage: pretrain [outDir] [seed] [batches] [collectSteps]
    /// </summary>
    /// <param name="args">Command-line arguments (args[0] is "pretrain").</param>
    private static void Pretrain(string[] args)
    {
        string dir = args.Length > 1 ? args[1] : ".";
        
        var cfg = new PretrainConfig { Seed = args.Length > 2 ? int.Parse(args[2]) : 1, Batches = args.Length > 3 ? int.Parse(args[3]) : 60_000,
            CollectSteps = args.Length > 4 ? int.Parse(args[4]) : 200_000 };
        
        Directory.CreateDirectory(dir);

        var pt = new PredictTrainer(cfg);
        Console.WriteLine($"pretrain: params={pt.Net.ParamCount:N0} (trunk {pt.TrunkParams:N0})  collect={cfg.CollectSteps:N0}  batches={cfg.Batches:N0}  seed={cfg.Seed}");
        Console.WriteLine("   batch   train-loss   val-loss   copy-last-frame-loss   improvement over copy");
    
        string csv = Path.Combine(dir, "pretrain_log.csv");
        File.WriteAllText(csv, "batch,loss,valLoss,copyLoss\n");

        pt.Train(s =>
        {
            Console.WriteLine($"{s.Batch,8} {s.Loss,12:F3} {s.ValLoss,10:F3} {s.CopyLoss,22:F3} {(1 - s.ValLoss / s.CopyLoss) * 100,18:F1}%");

            File.AppendAllText(csv, string.Create(CultureInfo.InvariantCulture, $"{s.Batch},{s.Loss},{s.ValLoss},{s.CopyLoss}\n"));
            
            Weights.Save(Path.Combine(dir, "pong_enc.bin"), pt.Weights[..pt.TrunkParams]);
            
            // keep a snapshot every 20k batches
            if (s.Batch % 20_000 == 0) Weights.Save(Path.Combine(dir, $"pong_enc_{s.Batch / 1000}k.bin"), pt.Weights[..pt.TrunkParams]);
            
            return false;
        });
    }

    /// <summary>
    /// Evaluate the AI to see how well it has learnt: load saved weights and play greedily (almost no random moves).
    /// Usage: eval [weights.bin] [points] [v2]
    /// </summary>
    /// <param name="args">Command-line arguments (args[0] is "eval").</param>
    private static void Eval(string[] args)
    {
        string path = args.Length > 1 ? args[1] : "pong_dqn.bin";
        int points = args.Length > 2 ? int.Parse(args[2]) : 500;

        // Use the same game rules the weights were trained on: "v2" if asked for on the command line, or if the
        // pong_env.txt written by 'train' (next to the weights file) says so. Otherwise the original v1 rules.
        PongEnv.GameVersion2 = args.Skip(3).Contains("v2") || (File.Exists(Path.Combine(Path.GetDirectoryName(Path.GetFullPath(path))!, "pong_env.txt")) && File.ReadAllText(Path.Combine(Path.GetDirectoryName(Path.GetFullPath(path))!, "pong_env.txt")).Trim() == "v2");
        
        var net = new QNet(new NetSpec());
        var theta = Weights.Load(path, net.ParamCount);
        var (win, rally, to) = DqnTrainer.Evaluate(net, theta, points, 999);
        
        Console.WriteLine($"{points} points: win {win * 100:F1}%  mean rally {rally:F1} frames  timeouts {to * 100:F0}%");
    }

    /// <summary>
    /// Runs a random baseline for a fixed number of points: a player that picks up/down/stay at random.
    /// This is the score to beat; a trained agent should be far above it.
    /// </summary>
    /// <param name="points">How many points to play.</param>
    private static void RandomBaseline(int points)
    {
        var env = new PongEnv(5); 
        var rng = new Random(6);

        int won = 0, lost = 0, n = 0;
        
        while (n < points)
        {
            env.Reset(); bool done = false;
            while (!done) { env.Step(rng.Next(3), out float r, out done); if (done) { if (r > 0) won++; else if (r < 0) lost++; n++; } }
        }
   
        Console.WriteLine($"random player, {points} points: win {won * 100.0 / Math.Max(1, won + lost):F1}% (decided points only)");
    }
}
