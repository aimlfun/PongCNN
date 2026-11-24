# BrainPong: Pong from pixels, CNN + Q-learning from scratch

A teaching build of the DeepMind DQN recipe. No ML library: the convolution, backprop, Adam, replay buffer and
target network are all plain C# you can read top to bottom.

## How the CNN and RL connect (the one idea)

Q-learning wants a table `Q(state, action)` = "expected future reward if I do `action` in `state`". Pixels have far
too many states for a table, so the **CNN is the table**:

    input : 4 stacked frames (40x40, 0/1)       <- stacking is how it sees velocity
    output: 3 numbers = Q(state, stay), Q(state, up), Q(state, down)

RL supplies a **label**; the CNN is trained to regress onto it (this is ordinary supervised learning):

    y    = r + gamma * max_a' Q_target(s', a')      the target (treated as a CONSTANT)
    loss = huber( Q(s, a_taken) - y )               only the taken action's output has an error

Backprop starts at that one output unit and the chain rule carries the error back through dense -> conv2 -> conv1.
See `DqnTrainer.LearnStep` (steps 1-5, commented) and `Layers.cs` (each `Backward` is one link of the chain).

Why the two tricks:
* **Replay** (`ReplayBuffer.cs`): consecutive frames are near-identical; SGD on them in order drifts one way. Random
  past steps look closer to independent samples.
* **Target net** (`DqnTrainer.Target`): `y` contains Q itself. If it came from the net being updated, every update
  would move its own labels. A frozen copy, refreshed every `TargetSync` steps, keeps the label steady in between.

## Files

| file | what |
|---|---|
| `Layers.cs` | `Dense`, `Conv2D`: forward + hand-written backward |
| `QNet.cs` | the 4-layer network, one flat weight array (`theta`) |
| `PongEnv.cs` | the game; agent sees only a 40x40 frame; 1 episode = 1 point, reward +1/-1 |
| `ReplayBuffer.cs` | ring buffer, rebuilds 4-frame stacks on demand |
| `DqnTrainer.cs` | epsilon-greedy act, store, minibatch learn, target sync |
| `PredictTrainer.cs` | reward-free pretraining of the trunk by next-frame prediction |
| `Program.cs` | `gradcheck`, `train`, `eval`, `random` modes + the numerical gradient check |

## Run (to TRAIN it)

    dotnet run -c Release -- train 1500000 r7_nstep 2 double lrdecay buf=300000 trunc nstep=5 v2 # recommended. ~2.4 h on a 12-thread laptop; writes out/pong_dqn.bin + pong_log.csv
    dotnet run -c Release -- gradcheck                  # proves the backprop (compare analytic vs numerical gradient)
    dotnet run -c Release -- random 500                 # baseline ~9% of points won
    dotnet run -c Release -- eval out/pong_dqn.bin

## Features found without reward

In plain DQN the conv filters are shaped ONLY by the TD loss, i.e. by whatever predicts Q. Here the trunk (conv1, conv2,
dense 128) learns first from a different question, with no reward at all: "given the last 4 frames and my action, what
does the next frame look like?" (`pretrain`). To get that right it must represent where the ball is and where it is going.
Then the trunk is frozen and DQN learns only the final 128 -> 3 layer on top (`train ... freeze=3 trunk=pong_enc.bin`).
The control (`freeze=3` with no `trunk=`) uses a frozen RANDOM trunk: if pretrained beats random, the prediction task really
produced useful features.

    dotnet run -c Release -- pretrain pre 1 40000 200000           # writes pre/pong_enc.bin, prints val loss vs "copy the last frame"
    dotnet run -c Release -- train 1500000 s1 2 double lrdecay buf=300000 trunc freeze=3 trunk=pre/pong_enc.bin

## Exercises (the learning part)

(1) In `gradcheck`, break one line of `Conv2D.Backward` (e.g. drop the `+=` on `dx`) and watch the check fail.
(2) Set `TargetSync = 1` (no frozen target) and compare the loss / Q curves.
(3) Set `ReplayCapacity = 32` (almost no replay) and compare.
(4) Look at meanQ in the log: it should creep toward the discounted value of a won point.
(5) change `ActiveWeight` to 1 and watch the model collapse to "predict black"; 
(6) compare val-loss with the copy-last-frame baseline: the gap is what the network learned about MOTION; 
(7) freeze=2 (only dense + head learn).
