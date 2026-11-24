namespace BrainPong;

/// <summary>
/// Minimal Pong on a 40x40 pixel field. 
/// The agent controls the RIGHT paddle and sees ONLY the rendered frame (0/1 pixels): no ball coordinates, no velocities. 
/// One episode is ONE POINT: reward +1 if the opponent misses, -1 if the agent misses, 0 on a rally timeout.
/// Actions: 0 = stay, 1 = up, 2 = down.
/// </summary>
internal sealed class PongEnv
{
    #region CONSTANTS
    /// <summary>
    /// How many pixels wide and tall the frame is. The paddles and ball are drawn as filled rectangles, 
    /// so the frame is a 2D array of 0/1 pixels. The agent sees only this frame, not the underlying coordinates or velocities.
    /// </summary>
    public const int Size = 40;

    /// <summary>
    /// Paddle height, paddle width, and ball size in pixels. The paddles are drawn as filled rectangles, so the agent sees them as a 2D array of 0/1 pixels.
    /// </summary>
    public const int PaddleHeight = 7, PaddleWidth = 2, BallSize = 2;

    /// <summary>
    /// X coordinates of the paddle faces. The left paddle occupies x in [2,4), the right paddle occupies x in [36,38).
    /// </summary>
    private const float LeftFace = 4f;   // right edge of the left paddle (it occupies x in [2,4))
    private const float RightFace = 36f; // left edge of the right paddle (it occupies x in [36,38))

    /// <summary>
    /// Paddle pixels per step.
    /// </summary>
    private const float AgentSpeedPX = 2f;

    /// <summary>
    /// Slower than steep shots, so the opponent can be beaten by angle.
    /// </summary>
    private const float OpponentSpeedPX = 0.8f;

    /// <summary>
    /// Max rally length. A rally that hits this limit is a draw in v1, a loss for the agent in v2.
    /// </summary>
    public const int MaxRally = 400;

    /// <summary>
    /// v2 rules (opt-in, so old runs/snapshots stay testable under the rules they were trained with):
    ///  - dead zone: neither paddle can reach the top DeadTop pixels (as in Atari Pong), so a ball skimming the top wall is unreachable
    ///  - ball speeds up 3% per paddle hit up to 3.0 (v1: 4% up to 1.8, which stops growing after ~15 hits)
    ///  - a rally that hits MaxRally is a LOSS for the agent (-1), not a free draw: stalling no longer costs nothing
    /// </summary>
    public static bool GameVersion2;

    /// <summary>
    /// The top DeadTop pixels are unreachable by either paddle. This is a v2 rule, so the agent must learn to avoid the top wall.
    /// </summary>
    public const float DeadTop = 3f;

    /// <summary>
    /// Neural network says don't move paddle.
    /// </summary>
    private const int ACTION_STAY = 0;

    /// <summary>
    /// Neural network says move paddle UP.
    /// </summary>
    private const int ACTION_UP = 1;

    /// <summary>
    /// Neural network says move paddle down.
    /// </summary>
    private const int ACTION_DOWN = 2;
    #endregion CONSTANTS

    /// <summary>
    /// The last step of the episode was a timeout (the rally clock hit MaxRally). This is raised for 
    /// one step so it can be counted, but the episode may continue if ContinueOnTimeout is true.
    /// </summary>
    public bool TimedOut;

    /// <summary>
    /// TRAINING ONLY. A timeout is a truncation (the clock ran out, nothing about the position was decided), so with this
    /// on the point simply goes on: the rally clock restarts, done stays false, reward 0. The learner then bootstraps
    /// from the real next state instead of being told "this position is worth 0 / -1 forever". TimedOut is still
    /// raised for the one step so it can be counted.
    /// </summary>
    public bool ContinueOnTimeout;

    /// <summary>
    /// Pseudo-random number generator for the environment. The seed is passed in the constructor so that runs can be repeated.
    /// </summary>
    private readonly Random rng;

    /// <summary>
    /// Position and velocity of the ball, and position of the paddles. The agent sees only the Frame array, not these coordinates.
    /// </summary>
    public float BallX, BallY, Vx, Vy;
    public float LeftY, RightY; // top edge of each paddle

    /// <summary>
    /// Rally clock (restarts on a continued timeout).
    /// </summary>
    public int Steps;

    /// <summary>
    /// Steps since the point started (never restarts).
    /// </summary>
    public int PointSteps;

    /// <summary>
    /// The current frame of the game, a 40x40 array of 0/1 pixels. The agent sees only this frame, 
    /// not the underlying coordinates or velocities.
    /// </summary>
    public readonly byte[] Frame = new byte[Size * Size];

    /// <summary>
    /// Constructor. The seed is used to initialize the random number generator so that runs can be repeated.
    /// </summary>
    /// <param name="seed"></param>
    public PongEnv(int seed)
    {
        rng = new Random(seed);  // intentionally not using Random.Shared so that runs can be repeated
    }

    /// <summary>
    /// Resets the environment to a new random state. The ball is placed in 
    /// the middle with a random angle, and the paddles are centered.
    /// Frame is drawn.
    /// </summary>
    /// <returns></returns>
    public byte[] Reset()
    {
        // ball in the middle with a random angle, paddles centered
        BallX = Size / 2f - 1;
        BallY = Size / 2f - 1 + (float)(rng.NextDouble() * 10 - 5);

        Vx = rng.Next(2) == 0 ? 1f : -1f; // random left or right
        Vy = (float)(rng.NextDouble() * 1.6 - 0.8); // random vertical +/- 0.8

        // center the paddles vertically
        LeftY = RightY = (Size - PaddleHeight) / 2f;

        // reset the rally clock and point clock, and render the new frame
        Steps = 0;
        PointSteps = 0;

        Render();

        return Frame;
    }

    /// <summary>
    /// Step the environment by one frame. The agent's action is applied, the opponent moves, 
    /// the ball moves, and the new frame is rendered. Returns the new frame, the reward, 
    /// and whether the episode is done.
    /// </summary>
    /// <param name="action">action to agent paddle</param>
    /// <param name="reward"></param>
    /// <param name="done"></param>
    /// <returns></returns>
    public byte[] Step(int action, out float reward, out bool done)
    {
        reward = 0;
        done = false;
        Steps++;
        PointSteps++;
        TimedOut = false;

        MoveAgentPaddle(action);

        MoveOpponentPaddle();

        MoveBall();

        DetectWinLose(ref reward, ref done);

        Render();
        return Frame;
    }

    /// <summary>
    /// Look at where the ball is, and detect a win/lose.
    /// </summary>
    /// <param name="reward"></param>
    /// <param name="done"></param>
    private void DetectWinLose(ref float reward, ref bool done)
    {
        // did anyone win?
        if (BallX + BallSize > Size)
        {
            reward = -1; // got past the agent
            done = true;
        }
        else if (BallX < 0)
        {
            reward = +1; // got past the opponent
            done = true;
        }
        else if (Steps >= MaxRally) // taking too long to win?
        {
            TimedOut = true;

            if (ContinueOnTimeout)
                Steps = 0; // truncation: the point goes on
            else
            {
                done = true;

                if (GameVersion2) reward = -1; // endless rally: v1 = draw, v2 = loss
            }
        }
    }

    /// <summary>
    /// The opponents paddle moves the bat when the ball is coming towards it. When it's not, it returns
    /// to the centre. Some bounces will catch it out, as the ball is quicker.
    /// </summary>
    private void MoveOpponentPaddle()
    {
        // opponent: tracks the ball only while it is coming toward it, otherwise drifts to the middle
        float target = Vx < 0 ? BallY + BallSize / 2f - PaddleHeight / 2f : (Size - PaddleHeight) / 2f;
        float d = Math.Clamp(target - LeftY, -OpponentSpeedPX, OpponentSpeedPX);

        float minY = GameVersion2 ? DeadTop : 0f;
        LeftY = Math.Clamp(LeftY + d, minY, Size - PaddleHeight);
    }

    /// <summary>
    /// Moves the (right) agent paddle based on the action.
    /// </summary>
    /// <param name="action"></param>
    private void MoveAgentPaddle(int action)
    {
        // agent paddle
        switch (action)
        {
            case ACTION_UP:
                RightY -= AgentSpeedPX;
                break;

            case ACTION_DOWN:
                RightY += AgentSpeedPX;
                break;

            case ACTION_STAY:
                break;
        }

        float minY = GameVersion2 ? DeadTop : 0f;

        // stop the paddle moving outside the defined bounds. The NN might ask to move up or down even though
        // the paddle can't
        RightY = Math.Clamp(RightY, minY, Size - PaddleHeight);
    }

    /// <summary>
    /// Add velocity to ball.
    /// </summary>
    private void MoveBall()
    {
        float prevX = BallX;
        BallX += Vx;
        BallY += Vy;

        // handle bouncing

        // walls hit?
        if (BallY < 0)
        {
            BallY = -BallY; Vy = -Vy;
        }

        if (BallY > Size - BallSize)
        {
            BallY = 2 * (Size - BallSize) - BallY; 
            Vy = -Vy;
        }

        // paddle hit?
        if (Vx > 0 && prevX + BallSize <= RightFace && BallX + BallSize > RightFace)
        {
            if (BallY + BallSize > RightY && BallY < RightY + PaddleHeight)
            {
                Bounce(RightY, -1, ref BallX, RightFace - BallSize);
            }
        }
        else if (Vx < 0 && prevX >= LeftFace && BallX < LeftFace)
        {
            if (BallY + BallSize > LeftY && BallY < LeftY + PaddleHeight)
            {
                Bounce(LeftY, +1, ref BallX, LeftFace);
            }
        }
    }

    /// <summary>
    /// Bounce the ball off a paddle. The paddle's position is used to compute the return angle, 
    /// and the ball's x is set to the paddle face.
    /// </summary>
    /// <param name="paddleY"></param>
    /// <param name="dir"></param>
    /// <param name="x"></param>
    /// <param name="newX"></param>
    private void Bounce(float paddleY, int dir, ref float x, float newX)
    {
        // Where the ball hits the paddle decides the return angle: the edge sends it steeply, the middle straight.
        float offset = (BallY + BallSize / 2f - (paddleY + PaddleHeight / 2f)) / (PaddleHeight / 2f);   // -1..1

        Vx = dir * (GameVersion2 ? Math.Min(Math.Abs(Vx) * 1.03f, 3.0f) : Math.Min(Math.Abs(Vx) * 1.04f, 1.8f));   // v2: +3% per hit, high cap, so rallies speed up
        Vy = Math.Clamp(offset * 1.3f, -1.5f, 1.5f);

        x = newX;
    }

    /// <summary>
    /// Renders the current state of the game into the Frame array: 0 = black, 1 = white. 
    /// The ball and paddles are drawn as filled rectangles.
    /// </summary>
    private void Render()
    {
        Array.Clear(Frame); // this is our "video display" - a "frame".

        // draw the paddles and ball as filled rectangles
        Rect(2, (int)MathF.Round(LeftY), PaddleWidth, PaddleHeight);
        Rect(36, (int)MathF.Round(RightY), PaddleWidth, PaddleHeight);

        // draw the ball as a filled rectangle, rounding to the nearest pixel
        Rect((int)MathF.Round(BallX), (int)MathF.Round(BallY), BallSize, BallSize);
    }

    /// <summary>
    /// Draws a filled rectangle into the Frame array. Clipped to the frame boundaries, 
    /// so it can be used for the ball even if it is partially off-screen.
    /// </summary>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <param name="w"></param>
    /// <param name="h"></param>
    private void Rect(int x, int y, int w, int h)
    {
        for (int yy = y; yy < y + h; yy++)
        {
            for (int xx = x; xx < x + w; xx++)
            {
                if (xx >= 0 && xx < Size && yy >= 0 && yy < Size)
                {
                    Frame[yy * Size + xx] = 1;
                }
            }
        }
    }
}