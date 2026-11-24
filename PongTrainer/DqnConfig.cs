namespace BrainPong;

/// <summary>
/// All the tunable settings (hyperparameters) for DQN training, with the classic-DQN values as defaults.
/// It is an immutable record: create one with the settings you want to change, e.g.
/// <c>new DqnConfig { DoubleDqn = true, NStep = 5 }</c>, and everything else keeps its default.
/// "Step" below always means one frame of the game (one action taken), not one gradient update.
/// </summary>
internal sealed record DqnConfig
{
    // ---- Run length and replay memory -----------------------------------------------------------------------------

    /// <summary>
    /// Total game steps to train for. At the default this takes a couple of hours on a laptop.
    /// </summary>
    public int TotalSteps { get; init; } = 1_500_000;

    /// <summary>
    /// How many past steps the replay buffer remembers. Once full, the oldest are overwritten.
    /// </summary>
    public int ReplayCapacity { get; init; } = 100_000;

    /// <summary>
    /// How many remembered steps are sampled for each gradient update.
    /// </summary>
    public int Batch { get; init; } = 32;

    /// <summary>
    /// Don't learn until this many steps are stored, so early updates aren't made from a handful of samples.
    /// </summary>
    public int LearnStart { get; init; } = 5_000;          // fill the buffer with random play first

    /// <summary>
    /// Play this many steps between gradient updates (4 = one update per 4 frames).
    /// </summary>
    public int TrainEvery { get; init; } = 4;              // one gradient step per this many env steps

    /// <summary>
    /// How often (in steps) the frozen "target" network is overwritten with the online network.
    /// The target gives stable regression targets; refreshing it too often makes training chase its own tail.
    /// </summary>
    public int TargetSync { get; init; } = 5_000;          // env steps between copies online -> target

    /// <summary>
    /// Discount factor: how much a reward one step in the future is worth today (0.99 = almost fully).
    /// Closer to 1 means the agent plans further ahead.
    /// </summary>
    public float Gamma { get; init; } = 0.99f;

    /// <summary>
    /// Learning rate: the size of each Adam weight update. Too high is unstable, too low is slow.
    /// </summary>
    public float Lr { get; init; } = 2.5e-4f;

    // ---- Exploration (epsilon-greedy) -----------------------------------------------------------------------------
    // With probability epsilon the agent takes a random action instead of the one it thinks is best.
    // Epsilon starts high (explore) and falls linearly to a low value (mostly exploit what it has learnt).

    /// <summary>
    /// Starting epsilon. 1.0 = every action is random at the beginning.
    /// </summary>
    public float EpsStart { get; init; } = 1.0f;

    /// <summary>
    /// Epsilon floor, held for the rest of training (5% random actions).
    /// </summary>
    public float EpsEnd { get; init; } = 0.05f;

    /// <summary>
    /// Number of steps over which epsilon falls from EpsStart to EpsEnd.
    /// </summary>
    public int EpsDecaySteps { get; init; } = 300_000;

    // ---- Bookkeeping ----------------------------------------------------------------------------------------------

    /// <summary>
    /// Seed for every random number generator, so a run can be reproduced exactly.
    /// </summary>
    public int Seed { get; init; } = 1;

    /// <summary>
    /// How often (in steps) to report progress, write the CSV log and save the weights.
    /// </summary>
    public int LogEvery { get; init; } = 10_000;

    // ---- Optional improvements over plain DQN (each one is a separate switch so runs can be compared) -------------

    /// <summary>
    /// Double DQN. Plain DQN uses the target net to both pick and score the best next action, which over-estimates
    /// values. Double DQN lets the online net pick and the target net score.
    /// </summary>
    public bool DoubleDqn { get; init; } = false;          // online net CHOOSES the next action, target net SCORES it

    /// <summary>
    /// Learning rate at the end of training, as a fraction of <see cref="Lr"/>. It falls linearly to that value.
    /// 1 = no decay, 0.1 = ends at a tenth of the starting rate.
    /// </summary>
    public float LrEndFrac { get; init; } = 1.0f;          // learning rate decays linearly to Lr * this by the last step (1 = no decay)

    /// <summary>
    /// Treat a rally that hits the time limit as "cut short" rather than "game over". The agent then still counts the
    /// future value of that state instead of believing it is worth nothing.
    /// </summary>
    public bool TruncateTimeouts { get; init; } = false;   // fix 1: a rally timeout is a truncation (bootstrap), not a terminal

    /// <summary>
    /// N-step returns: add up the next N rewards before estimating the rest. Spreads the reward back along the rally
    /// faster than 1-step. 1 = classic DQN.
    /// </summary>
    public int NStep { get; init; } = 1;                   // fix 2: n-step returns (1 = classic DQN)

    /// <summary>
    /// Transfer learning: the first k layers keep their weights and are never updated. 3 = the whole feature-extracting
    /// "trunk" (conv1, conv2, dense) is frozen and only the final Q-value layer learns. Used with a pretrained trunk.
    /// </summary>
    public int FreezeLayers { get; init; } = 0;            // the first k layers keep their weights (3 = whole trunk, only the Q head learns)
}
