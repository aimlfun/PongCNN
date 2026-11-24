namespace BrainPong;

/// <summary>
/// Settings for the reward-free pretraining step (<see cref="PredictTrainer"/>), where the network learns
/// to predict the next frame of the game. Create one with <c>new PretrainConfig { Batches = 10_000 }</c> etc.
/// </summary>
internal sealed record PretrainConfig
{
    /// <summary>How many steps of random play to record as the training data.</summary>
    public int CollectSteps { get; init; } = 200_000;      // random-play steps to learn from (no reward is ever used)

    /// <summary>How many gradient updates to run in total.</summary>
    public int Batches { get; init; } = 60_000;

    /// <summary>Samples per gradient update.</summary>
    public int Batch { get; init; } = 32;

    /// <summary>Learning rate for the Adam optimiser.</summary>
    public float Lr { get; init; } = 5e-4f;

    /// <summary>
    /// Pixels are mostly empty background, so a network could score well by predicting "all black". To stop that,
    /// errors on pixels where the ball or paddle is (or just was) are multiplied by this weight. Background counts as 1.
    /// </summary>
    public float ActiveWeight { get; init; } = 20f;        // pixels where something is, or just was, count this much (background = 1)

    /// <summary>Seed for the random number generators, so the run is reproducible.</summary>
    public int Seed { get; init; } = 1;

    /// <summary>Report progress and save the weights every this many batches.</summary>
    public int LogEvery { get; init; } = 2_000;
}
