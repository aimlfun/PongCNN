namespace BrainPong;

/// <summary>
/// A progress report from DQN training, produced every <see cref="DqnConfig.LogEvery"/> steps.
/// It is printed to the console, appended to pong_log.csv, and read by the viewer.
/// </summary>
/// <param name="Step">Game steps (frames) played so far.</param>
/// <param name="Eps">Epsilon: the current probability of taking a random move instead of the network's choice.</param>
/// <param name="Loss">Average Huber loss over the gradient updates since the last report (how wrong the Q predictions are).</param>
/// <param name="MeanQ">Average predicted Q-value of the actions taken. Roughly "how good the agent thinks things are"; a runaway rise suggests over-estimation.</param>
/// <param name="WinRate">Fraction of the last 300 finished points that the agent won. Measured while still exploring randomly, so it understates the greedy play.</param>
/// <param name="AvgRally">Average length (in steps) of those last 300 points.</param>
/// <param name="Points">Total points finished so far (won or lost).</param>
/// <param name="StepsPerSec">Training speed since the last report.</param>
/// <param name="BufReward">Fraction of the replay buffer holding a real reward (+1/-1). Reward is rare in Pong, which is why learning is slow.</param>
/// <param name="BufLong">Fraction of the replay buffer from 100+ steps into a rally. A high value means the buffer is full of near-identical frames.</param>
/// <param name="Timeouts">Points since the last report that hit the time limit without anyone scoring: the agent kept returning the ball but didn't win.</param>
internal sealed record TrainStats(
    int Step, float Eps, float Loss, float MeanQ, float WinRate, float AvgRally, int Points, double StepsPerSec,
    float BufReward = 0, float BufLong = 0, int Timeouts = 0);
