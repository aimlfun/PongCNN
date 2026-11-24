# PongCNN - "DeepMind" level AI game of Pong

After watching an AI documentary where they talked about DeepMind playing Pong. Having already made one without a CNN that learns quickly, I thought let's play with a CNN.

This Pong uses double DQN, CNN, RL like DeepMind's but smaller. It works, just as well.

The screen is 40x40 px.

CNN  - >CNN  -> NN   -> NN  -> ACTION
6400 -> 2592 -> 1024 -> 128 -> 3 = stay,up,down [argmax()]

i.e.
CNN1: 4 frames x (40px x 40px) kernel 6x6 stride 2 => 8 output channels (18x18) ReLU
|
CNN2: 8 x (18 x 18) kernel 4x4 stride 2 => 16 channels (8x8) ReLU
|
Dense (all inputs connected to all outputs) in: 1024 out: 128 relReLU
|
Dense (all inputs connected to all outputs) in: 128 out: 3 (no ReLU)
|
ACTIONS

---

# Before running

By default the trainer writes the trained neural network to **C:\temp\r7_nstep**, and unsurprisingly the viewer looks in that folder. So if you don't have a **c:\temp**, either create it or change the launch settings to another folder.

# PongTrainer

Run this app to train the AI. Depending on your hardware, it may take 2 hours to complete 1,500,000 steps. Every 10,000 it outputs progress.

Pong will be play *badly* until it has trained 70-100,000 steps.

Training happens without displaying games, as a console app.

# PongViewer

Run this app to see it play games. Click "Why" to see what makes the layers pick up or down.

**Note: the viewer doesn't train, it's a view that loads the data populated by the trainer.**