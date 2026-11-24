using System.Drawing.Drawing2D;
using System.Globalization;
using BrainPong;

namespace BrainPongViewer;

/// <summary>
/// Double-buffered surface that calls a draw delegate.
/// </summary>
internal sealed class Canvas : Control
{
    public Action<Graphics, Rectangle>? Draw;

    public Canvas()
    {
        SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.UserPaint | ControlStyles.OptimizedDoubleBuffer | ControlStyles.ResizeRedraw, true);
        BackColor = Color.FromArgb(24, 26, 32);
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        e.Graphics.Clear(BackColor);
        e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
        Draw?.Invoke(e.Graphics, ClientRectangle);
    }
}

internal sealed class MainForm : Form
{
    private const int RowTop = 56, RowPitch = 6 * 7 + 6;   // shared by filters and maps so row i lines up in both

    private static readonly Color Fg = Color.FromArgb(220, 224, 232), Dim = Color.FromArgb(130, 136, 150);

    private static readonly Color Pos = Color.FromArgb(90, 170, 255), Neg = Color.FromArgb(255, 110, 90), Accent = Color.FromArgb(120, 220, 140);

    /// <summary>
    /// Used to draw the white pixels of the 40x40 frame as the agent sees it. The agent sees 
    /// only 0/1 pixels, so we draw 1s as white and 0s as black.
    /// </summary>
    private static readonly SolidBrush s_whiteBrush = new(Color.FromArgb(235, 235, 240));

    /// <summary>
    /// A ball heading away from the agent (towards the opponent) is drawn grey: the agent has nothing to decide about it.
    /// </summary>
    private static readonly SolidBrush s_awayBrush = new(Color.FromArgb(105, 108, 118));

    /// <summary>
    /// Used to draw the dead zone above the paddles if v2: the strip of pixels no paddle can reach. This is a red zone.
    /// </summary>
    private static readonly SolidBrush s_deadZoneBrush = new(Color.FromArgb(70, 255, 110, 90));

    private readonly QNet net = new(new NetSpec());
    
    private readonly Workspace workSpace;

    private float[] theta;

    private string dir;

    private DateTime weightsStamp = DateTime.MinValue;

    private string weightsInfo = "untrained (random weights)";

    // the game being watched
    private PongEnv env = new(1);
    
    private readonly byte[][] hist;

    private readonly int area = PongEnv.Size * PongEnv.Size;

    private float[] q = new float[3];

    private int lastAction;

    private int won, lost, seedCounter = 1;

    private bool paused;

    private int speed = 1;
    private int slowDiv = 1;   // >1 = slow motion: step the game once every slowDiv ticks
    private int slowCount;
    private bool stepOnce;

    /// <summary>
    /// Learning curve. Each entry is (step, win%, mean Q, epsilon). The curve is hot-reloaded from pong_log.csv while a training run keeps saving it.
    /// </summary>
    private List<(int step, float win, float q, float eps)> curve = [];

    private DateTime curveStamp = DateTime.MinValue;

    private readonly Canvas game = new(), filters = new(), maps = new(), maps2 = new(), chart = new();

    /// <summary>
    /// which conv1 unit (row) the mouse is over, in either panel
    /// </summary>
    private int hover = -1;

    /// <summary>
    /// The "why" popup, if open.
    /// </summary>
    private FlowForm? flow;

    private readonly Label status = new();
    
    private readonly System.Windows.Forms.Timer tick = new() { Interval = 33 };

    public MainForm(string folder)
    {
        dir = folder;
        Text = "BrainPong: DQN watching itself play Pong from pixels";
        BackColor = Color.FromArgb(24, 26, 32);
        ClientSize = new Size(1180, 760);
        MinimumSize = new Size(1000, 700);

        workSpace = net.NewWorkspace();
        theta = net.NewParams(1);

        hist = new byte[net.Spec.Frames][];

        for (int i = 0; i < hist.Length; i++) hist[i] = new byte[area];

        // top bar with buttons and status
        var bar = new FlowLayoutPanel { Dock = DockStyle.Top, Height = 40, Padding = new Padding(6, 6, 6, 0), BackColor = Color.FromArgb(34, 36, 44) };
        bar.Controls.Add(Btn("Pause / Resume", () => paused = !paused));
        bar.Controls.Add(Btn("Speed: slow/x1/x4/x16", CycleSpeed));
        bar.Controls.Add(Btn("Step 1 frame", () => { paused = true; stepOnce = true; }));
        bar.Controls.Add(Btn("New match", NewMatch));
        bar.Controls.Add(Btn("Open folder...", OpenFolder));
        bar.Controls.Add(Btn("Why? (network flow)", OpenFlow));

        status.AutoSize = true; status.ForeColor = Fg; status.Padding = new Padding(10, 7, 0, 0);
        bar.Controls.Add(status);
        Controls.Add(bar);

        var root = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, RowCount = 2, Padding = new Padding(6) };
        root.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 460));
        root.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        root.RowStyles.Add(new RowStyle(SizeType.Percent, 60));
        root.RowStyles.Add(new RowStyle(SizeType.Percent, 40));

        foreach (var c in new[] { game, filters, maps, maps2, chart }) c.Dock = DockStyle.Fill;
        game.Draw = DrawGame; filters.Draw = DrawFilters; maps.Draw = DrawMaps; maps2.Draw = DrawMaps2; chart.Draw = DrawChart;

        // The conv1 filters and maps panels are small. We want to be able to see them properly.
        // Each share the same hover row, so the mouse can be over either one        
        foreach (var c in new[] { filters, maps })
        {
            c.MouseMove += (_, e) => hover = RowAt(e.Y);
            c.MouseLeave += (_, _) => hover = -1;
        }

        var right = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, RowCount = 1 };
        right.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 270));
        right.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        right.Controls.Add(filters, 0, 0); right.Controls.Add(maps, 1, 0);

        root.Controls.Add(game, 0, 0); root.SetRowSpan(game, 2);
        root.Controls.Add(right, 1, 0);
        // bottom row: conv2 feature maps beside the learning curve
        var bottom = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, RowCount = 1 };
        bottom.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 270));
        bottom.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        bottom.Controls.Add(maps2, 0, 0); bottom.Controls.Add(chart, 1, 0);
        root.Controls.Add(bottom, 1, 1);
        Controls.Add(root); root.BringToFront();

        // show the initial state of the game and the network (random weights) before the first tick
        ReloadCurve(); 
        ReloadWeights(); 
        NewMatch();

        // tick every 33ms (30fps) to update the game, reload weights and curve, and redraw
        tick.Tick += (_, _) => Tick();
        tick.Start();
    }

    /// <summary>
    /// Creates a button with a given text and click action. The button is styled to match the app's theme.
    /// </summary>
    /// <param name="text"></param>
    /// <param name="a"></param>
    /// <returns></returns>
    private static Button Btn(string text, Action a)
    {
        var b = new Button { Text = text, AutoSize = true, FlatStyle = FlatStyle.Flat, ForeColor = Fg, BackColor = Color.FromArgb(52, 56, 68) };
        b.Click += (_, _) => a();
        return b;
    }

    /// <summary>
    /// Opens a folder browser dialog to select a new folder containing pong_dqn.bin and pong_log.csv.
    /// </summary>
    private void OpenFolder()
    {
        using var d = new FolderBrowserDialog { SelectedPath = dir };

        if (d.ShowDialog() != DialogResult.OK) return;
        dir = d.SelectedPath; weightsStamp = DateTime.MinValue; curveStamp = DateTime.MinValue;

        ReloadCurve(); ReloadWeights(); NewMatch();
    }

    /// <summary>
    /// Opens the popup that explains the current decision as a flow through the network. Non-modal, so the game keeps running.
    /// </summary>
    private void OpenFlow()
    {
        if (flow is { IsDisposed: false }) { flow.Activate(); return; }

        flow = new FlowForm(net, workSpace, () => theta, () => q, () => lastAction);
        flow.FormClosed += (_, _) => flow = null;
        flow.Show(this);
    }

    /// <summary>
    /// Resets the match: the win/loss counters are reset and a new point is started.
    /// </summary>
    private void NewMatch() { won = lost = 0; StartPoint(); }

    /// <summary>
    /// Resets the environment to start a new point. The agent and opponent paddles are reset, 
    /// the ball is reset, and the history of frames is filled with the first frame.
    /// </summary>
    private void StartPoint()
    {
        env = new PongEnv(++seedCounter);
        var f = env.Reset();

        // fill the history with the first frame so the network sees a full 4-frame input
        for (int i = 0; i < hist.Length; i++) Array.Copy(f, hist[i], area);
    }

    // ------------------------------------------------------------------ simulation

    /// <summary>
    /// The main loop: reloads the weights and curve every second, steps the game at the current speed, and redraws all panels.
    /// </summary>
    private void CycleSpeed()
    {
        // order: x1 -> x4 -> x16 -> slow (1/8, ~4 frames/s) -> very slow (1/20, ~1.5 frames/s) -> x1
        slowCount = 0;
        if (slowDiv == 20) { slowDiv = 1; speed = 1; }
        else if (slowDiv == 8) slowDiv = 20;
        else if (speed == 16) { speed = 1; slowDiv = 8; }
        else speed = speed == 1 ? 4 : 16;
    }

    private string SpeedLabel() => slowDiv > 1 ? $"1/{slowDiv}" : $"x{speed}";

    /// <summary>
    /// Moves the ball, and bats (steps the game).
    /// </summary>
    private void Tick()
    {
        if ((DateTime.Now.Second & 1) == 0 && tick.Tag is not true)
        {
            tick.Tag = true;
            ReloadCurve();
            ReloadWeights();
        }
        else if ((DateTime.Now.Second & 1) == 1)
        {
            tick.Tag = false;
        }

        // if not paused, it steps the game forward by one frame, applying the agent's action and moving the opponent and ball
        if (!paused)
        {
            if (slowDiv > 1)
            {
                if (++slowCount >= slowDiv)
                {
                    slowCount = 0; 
                    StepGame();
                }
            }
            else
            {
                for (int s = 0; s < speed; s++)
                {
                    StepGame();
                }
            }
        }
        else if (stepOnce)
        {
            stepOnce = false; 
            StepGame();
        }

        RedrawEverything();

        status.Text = $"{weightsInfo}   |   match: agent {won} - {lost} opponent   |   speed {SpeedLabel()}{(paused ? "  [paused]" : "")}";
    }

    /// <summary>
    /// redraw all panels: the game, the conv1 filters, the conv1 maps, and the learning curve
    /// </summary>
    private void RedrawEverything()
    {
        game.Invalidate();
        filters.Invalidate();
        maps.Invalidate();
        maps2.Invalidate();
        chart.Invalidate();
        flow?.Redraw();
    }

    /// <summary>
    /// Steps the game forward by one frame. The network looks at the last 4 frames and prices each action, then the best action is taken.
    /// </summary>
    private void StepGame()
    {
        int F = hist.Length;

        CopyLast4FramesIntoWorkspaceInput(F);

        q = (float[])net.Forward(theta, workSpace).Clone(); // the network looks at the 4 frames and prices each action

        lastAction = DqnTrainer.Argmax(q);           // greedy: the best-priced action

        // step the game forward by one frame, applying the agent's action and moving the opponent and ball
        var nf = env.Step(lastAction, out float r, out bool done);
        
        RotateHistory(F, nf);

        if (done)
        {
            // the point is over: count the result and start a new point
            if (r > 0) won++; else if (r < 0) lost++;

            StartPoint();
        }
    }

    /// <summary>
    /// Rotates the history of frames: t-2 becomes t-3, t-1 becomes t-2, now becomes t-1, and the new frame becomes now.
    /// </summary>
    /// <param name="F"></param>
    /// <param name="nf"></param>
    private void RotateHistory(int F, byte[] nf)
    {
        byte[] old = hist[0];

        for (int i = 0; i < F - 1; i++)
        {
            hist[i] = hist[i + 1];
        }

        hist[F - 1] = old;

        Array.Copy(nf, old, area);
    }

    /// <summary>
    /// Copies the last 4 frames into the workspace input, in the order t-3, t-2, t-1, now. 
    /// The workspace input is a 1D array of 6400 floats (4 frames of 40x40 pixels).
    /// </summary>
    /// <param name="F"></param>
    private void CopyLast4FramesIntoWorkspaceInput(int F)
    {
        for (int i = 0; i < F; i++)
        {
            for (int p = 0; p < area; p++)
            {
                workSpace.Input[i * area + p] = hist[i][p];
            }
        }
    }

    // ------------------------------------------------------------------ weights + curve (hot reload)

    /// <summary>
    /// Reloads the rules from pong_env.txt if it has changed. The rules control whether the game is v1 or v2.
    /// </summary>
    private void ReloadRules()
    {
        try
        {
            string f = Path.Combine(dir, "pong_env.txt");

            bool v2 = File.Exists(f) && File.ReadAllText(f).Trim() == "v2";

            if (v2 != PongEnv.GameVersion2) { PongEnv.GameVersion2 = v2; NewMatch(); }
        }
        catch (IOException) 
        { 
            // file read error, might be being updated. Ignoring is fine.
        }
    }

    /// <summary>
    /// Reloads the weights from pong_dqn.bin if it has changed. 
    /// The weights are used to price the actions.
    /// </summary>
    private void ReloadWeights()
    {
        ReloadRules();

        string path = Path.Combine(dir, "pong_dqn.bin");

        try
        {
            if (!File.Exists(path))
            {
                weightsInfo = $"no pong_dqn.bin in {dir}: random weights - run the trainer.";
                return;
            }

            var stamp = File.GetLastWriteTimeUtc(path);

            if (stamp == weightsStamp) return;

            theta = Weights.Load(path, net.ParamCount);
            weightsStamp = stamp;
            weightsInfo = $"weights saved {stamp.ToLocalTime():HH:mm:ss}" + (curve.Count > 0 ? $" (step {curve[^1].step:N0})" : "");
        }
        catch (IOException) { /* writer is mid-save; try again next second */ }
        catch (InvalidDataException ex) { weightsInfo = ex.Message; }
    }

    /// <summary>
    /// Reloads the learning curve from pong_log.csv if it has changed. 
    /// The curve is a list of (step, win%, mean Q, epsilon) tuples.
    /// </summary>
    private void ReloadCurve()
    {
        string path = Path.Combine(dir, "pong_log.csv");
        try
        {
            if (!File.Exists(path))
            {
                curve = [];
                return;
            }

            var stamp = File.GetLastWriteTimeUtc(path);

            if (stamp == curveStamp) return;

            var list = new List<(int, float, float, float)>();

            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            using var sr = new StreamReader(fs);

            sr.ReadLine(); // skip the header line
            string? line;

            while ((line = sr.ReadLine()) != null)
            {
                var p = line.Split(',');
                if (p.Length < 5) continue;

                list.Add((int.Parse(p[0], CultureInfo.InvariantCulture), float.Parse(p[4], CultureInfo.InvariantCulture),
                          float.Parse(p[3], CultureInfo.InvariantCulture), float.Parse(p[1], CultureInfo.InvariantCulture)));
            }

            curve = list; curveStamp = stamp;
        }
        catch (IOException) { }
        catch (FormatException) { }
    }

    // ------------------------------------------------------------------ drawing

    /// <summary>
    /// Write text to the graphics surface. Uses Segoe UI, anti-aliased, with a given color and size.
    /// </summary>
    /// <param name="g"></param>
    /// <param name="s"></param>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <param name="c"></param>
    /// <param name="size"></param>
    /// <param name="st"></param>
    private static void Text2(Graphics g, string s, float x, float y, Color c, float size = 9f, FontStyle st = FontStyle.Regular)
    {
        using var f = new Font("Segoe UI", size, st); using var b = new SolidBrush(c);
        g.DrawString(s, f, b, x, y);
    }

    /// <summary>
    /// Draws the game state as the agent sees it: the 40x40 pixel frame, the three Q-values, and the dead zone if v2.
    /// </summary>
    /// <param name="g"></param>
    /// <param name="r"></param>
    private void DrawGame(Graphics g, Rectangle r)
    {
        Text2(g, "WHAT THE AGENT SEES (40x40 pixels, nothing else)", 6, 4, Dim, 8.5f, FontStyle.Bold);

        int cell = 10, ox = 12, oy = 26;
        g.InterpolationMode = InterpolationMode.NearestNeighbor;
        g.FillRectangle(Brushes.Black, ox, oy, 40 * cell, 40 * cell);

        DrawVideoScreenWithBatsAndBall(g, cell, ox, oy);

        if (PongEnv.GameVersion2) PaintDeadZoneAboveBats(g, cell, ox, oy);

        g.DrawRectangle(Pens.DimGray, ox, oy, 40 * cell, 40 * cell);

        Text2(g, "opponent", ox + 4, oy + 40 * cell + 2, Dim, 8);
        Text2(g, "agent (right paddle)", ox + 40 * cell - 110, oy + 40 * cell + 2, Accent, 8);
        DrawTheQValues(g, cell, ox, oy);
    }

    /// <summary>
    /// Draws the three Q-values for the current state: the expected future reward of each action.
    /// The bars are colored green if this was the last action taken, blue if positive, and red if negative. 
    /// The magnitude of the value controls the length of the bar.
    /// </summary>
    /// <param name="g"></param>
    /// <param name="cell"></param>
    /// <param name="ox"></param>
    /// <param name="oy"></param>
    private void DrawTheQValues(Graphics g, int cell, int ox, int oy)
    {
        // the three Q-values: what the network believes each action is worth
        int by = oy + 40 * cell + 26;
        Text2(g, "Q(state, action): expected future reward of each move", 6, by - 4, Dim, 8.5f, FontStyle.Bold);

        string[] names = ["stay", "up", "down"];
        float lim = Math.Max(0.3f, q.Max(Math.Abs));

        for (int a = 0; a < 3; a++)
        {
            int y = by + 18 + a * 22, mid = ox + 220, half = 170;
            // the action name and a vertical line separating it from the bar
            Text2(g, names[a], ox, y - 2, a == lastAction ? Accent : Fg, 9, a == lastAction ? FontStyle.Bold : FontStyle.Regular);
            g.DrawLine(Pens.DimGray, mid, y - 2, mid, y + 16);

            float w = q[a] / lim * half;

            // the bar: green if this was the last action, blue if positive, red if negative
            using var br = new SolidBrush(a == lastAction ? Accent : (q[a] >= 0 ? Pos : Neg));
            g.FillRectangle(br, w >= 0 ? mid : mid + w, y, Math.Abs(w), 14);
            Text2(g, q[a].ToString("+0.000;-0.000", CultureInfo.InvariantCulture), mid + (w >= 0 ? w + 4 : 4), y - 2, Fg, 8);
        }
    }

    /// <summary>
    /// Draws a dead zone above the paddles if v2: the strip of pixels no paddle can reach. This is a red zone.
    /// The point is that a ball skimming the top wall is unreachable, enabling a win.
    /// </summary>
    /// <param name="g"></param>
    /// <param name="cell"></param>
    /// <param name="ox"></param>
    /// <param name="oy"></param>
    private static void PaintDeadZoneAboveBats(Graphics g, int cell, int ox, int oy)
    {
        // draw the dead zone if v2: the strip of pixels no paddle can reach (a red zone)

        g.FillRectangle(s_deadZoneBrush, ox + 2 * cell, oy, 2 * cell, (int)(PongEnv.DeadTop * cell));
        g.FillRectangle(s_deadZoneBrush, ox + 36 * cell, oy, 2 * cell, (int)(PongEnv.DeadTop * cell));
    }

    /// <summary>
    /// Draw the 40x40 pixel frame as the agent sees it: 
    /// - white pixels are 1, 
    /// - black pixels are 0
    /// The frame is stored as a 1D array of 1600 bytes, so we loop over y and x to draw each pixel.
    /// </summary>
    /// <param name="g"></param>
    /// <param name="cell"></param>
    /// <param name="ox"></param>
    /// <param name="oy"></param>
    private void DrawVideoScreenWithBatsAndBall(Graphics g, int cell, int ox, int oy)
    {
        for (int y = 0; y < 40; y++)
        {
            for (int x = 0; x < 40; x++)
            {
                if (env.Frame[y * 40 + x] != 0)
                {
                    // columns 4..35 are the ball (paddles occupy x<4 and x>=36); the agent is the right-hand paddle, so Vx > 0 = coming towards it
                    bool isBall = x >= 4 && x < 36;
                    g.FillRectangle(isBall && env.Vx <= 0 ? s_awayBrush : s_whiteBrush, ox + x * cell, oy + y * cell, cell, cell);
                }
            }
        }
    }

    /// <summary>
    /// Draws a color that diverges from gray to blue for positive values and red for negative values. 
    /// The magnitude of the value controls the saturation.
    /// </summary>
    /// <param name="v"></param>
    /// <returns></returns>
    private static Color Diverge(float v)
    {
        v = Math.Clamp(v, -1f, 1f);
        var c = v >= 0 ? Pos : Neg; float t = Math.Abs(v);

        return Color.FromArgb((int)(30 + (c.R - 30) * t), (int)(30 + (c.G - 30) * t), (int)(34 + (c.B - 34) * t));
    }

    /// <summary>
    /// Returns the row index of the conv1 unit under the mouse, or -1 if the mouse is not over a row.
    /// </summary>
    /// <param name="y"></param>
    /// <returns></returns>
    private int RowAt(int y)
    {
        int i = (y - RowTop) / RowPitch;
        return y >= RowTop && i < net.Spec.C1 ? i : -1;
    }

    /// <summary>
    /// Draws the conv1 filters: each row is one unit, and the four frames are shown in a 2x2 grid. 
    /// The colour scale is set by the maximum absolute value of that filter alone.
    /// </summary>
    /// <param name="g"></param>
    /// <param name="r"></param>
    private void DrawFilters(Graphics g, Rectangle r)
    {
        Text2(g, "CONV1 FILTERS (8 x 4 frames)", 6, 4, Dim, 8.5f, FontStyle.Bold);
        Text2(g, "blue = raises the unit, red = lowers", 6, 20, Dim, 8);

        var fl = net.Conv1Filters(theta);
        float mx = 1e-6f; foreach (var f in fl) foreach (var c in f) foreach (var v in c) mx = Math.Max(mx, Math.Abs(v));
        int k = net.Spec.KernelSize1, px = 7, gap = 6, top = 56;

        g.InterpolationMode = InterpolationMode.NearestNeighbor;

        string[] cols = ["t-3", "t-2", "t-1", "now"];

        for (int ci = 0; ci < 4; ci++) Text2(g, cols[ci], 30 + ci * (k * px + gap), top - 14, Dim, 7.5f);

        if (hover >= 0) using (var hb = new SolidBrush(Color.FromArgb(40, 255, 255, 255))) g.FillRectangle(hb, 0, top + hover * (k * px + gap) - 3, r.Width, k * px + gap);

        for (int co = 0; co < fl.Length; co++)
        {
            Text2(g, $"#{co}", 4, top + co * (k * px + gap) + 10, Dim, 8);

            for (int ci = 0; ci < fl[co].Length; ci++)
            {
                for (int y = 0; y < k; y++) for (int x = 0; x < k; x++)
                {
                    using var b = new SolidBrush(Diverge(fl[co][ci][y * k + x] / mx));
                    g.FillRectangle(b, 30 + ci * (k * px + gap) + x * px, top + co * (k * px + gap) + y * px, px, px);
                }
            }
        }
    }

    /// <summary>
    /// Draws the feature maps produced by the conv1 filters for the current state. Each row is one unit, 
    /// and the four frames are averaged together to produce a single map.
    /// </summary>
    /// <param name="g"></param>
    /// <param name="r"></param>
    private void DrawMaps(Graphics g, Rectangle r)
    {
        Text2(g, "CONV1 FEATURE MAPS for the current state", 6, 4, Dim, 8.5f, FontStyle.Bold);
        Text2(g, "where each filter fires (bright = strong)", 6, 20, Dim, 8);

        var (m, h, w) = net.Conv1Maps(workSpace); // m = [C1][H*W] for the current state in ws.Input

        // one vertical column, same row pitch as the filters panel so map #i lines up with filter #i
        int k = net.Spec.KernelSize1, pitch = k * 7 + 6, top = 56, size = pitch - 6;

        g.InterpolationMode = InterpolationMode.NearestNeighbor;
        g.PixelOffsetMode = PixelOffsetMode.Half;

        if (hover >= 0) using (var hb = new SolidBrush(Color.FromArgb(40, 255, 255, 255))) g.FillRectangle(hb, 0, top + hover * pitch - 3, 80, pitch);

        for (int i = 0; i < m.Length; i++)
        {
            float mx = Math.Max(1e-6f, m[i].Max());
            Text2(g, $"#{i}", 4, top + i * pitch + 10, Dim, 8);

            using var bmp = new Bitmap(w, h);

            for (int y = 0; y < h; y++) for (int x = 0; x < w; x++)
            {
                int v = (int)(255 * Math.Clamp(m[i][y * w + x] / mx, 0, 1));
                bmp.SetPixel(x, y, Color.FromArgb(v, v, v));
            }

            g.DrawImage(bmp, new Rectangle(30, top + i * pitch, size, size));
        }

        DrawZoom(g, r, m, h, w);
    }

    /// <summary>
    /// Draws the conv2 feature maps for the current state: 16 units on a 4x4 grid. Conv2 reads conv1's maps,
    /// so these respond to combinations of conv1 features (e.g. ball moving towards a paddle).
    /// </summary>
    private void DrawMaps2(Graphics g, Rectangle r)
    {
        Text2(g, "CONV2 FEATURE MAPS (16 units)", 6, 4, Dim, 8.5f, FontStyle.Bold);

        var (m, h, w) = net.Conv2Maps(workSpace);

        int cols = 4, rows = (m.Length + cols - 1) / cols, top = 26, gap = 16;
        int size = Math.Min((r.Width - 12 - (cols - 1) * 6) / cols, (r.Height - top - 6 - rows * gap) / rows);

        if (size < 16) return;

        g.InterpolationMode = InterpolationMode.NearestNeighbor;
        g.PixelOffsetMode = PixelOffsetMode.Half;

        for (int i = 0; i < m.Length; i++)
        {
            int x0 = 6 + (i % cols) * (size + 6), y0 = top + (i / cols) * (size + gap);
            float mx = Math.Max(1e-6f, m[i].Max());

            using var bmp = new Bitmap(w, h);

            for (int y = 0; y < h; y++) for (int x = 0; x < w; x++)
            {
                int v = (int)(255 * Math.Clamp(m[i][y * w + x] / mx, 0, 1));
                bmp.SetPixel(x, y, Color.FromArgb(v, v, v));
            }

            g.DrawImage(bmp, new Rectangle(x0, y0 + gap - 2, size, size - 2));
            Text2(g, $"#{i}", x0, y0 - 2, Dim, 7.5f);
        }
    }

    /// <summary>
    /// Hover zoom: the filter under the mouse (its 4 frames) next to the map it produces, both enlarged.
    /// </summary>
    private void DrawZoom(Graphics g, Rectangle r, float[][] m, int h, int w)
    {
        int x0 = 100, y0 = 44;

        if (hover < 0)
        {
            Text2(g, "hover a row to enlarge that filter and its map", x0, y0, Dim, 8.5f);
            return;
        }

        int S = Math.Min(r.Height - y0 - 30, (r.Width - x0 - 30) / 2);

        if (S < 60) return;

        Text2(g, $"unit #{hover}", x0, y0 - 2, Accent, 9, FontStyle.Bold);

        int top = y0 + 20;
        g.InterpolationMode = InterpolationMode.NearestNeighbor; g.PixelOffsetMode = PixelOffsetMode.Half;

        // feature map (this unit's response over the picture)
        float mx = Math.Max(1e-6f, m[hover].Max());

        using (var bmp = new Bitmap(w, h))
        {
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    int v = (int)(255 * Math.Clamp(m[hover][y * w + x] / mx, 0, 1)); 
                    bmp.SetPixel(x, y, Color.FromArgb(v, v, v));
                }
            }

            g.DrawImage(bmp, new Rectangle(x0, top, S, S));
        }

        g.DrawRectangle(Pens.DimGray, x0, top, S, S);
        Text2(g, "feature map (bright = fires here)", x0, top + S + 2, Dim, 8);

        // the filter: 4 frames in a 2x2 grid, colour scale set by this filter alone
        var f = net.Conv1Filters(theta)[hover]; int k = net.Spec.KernelSize1;
        float fm = 1e-6f;
        foreach (var c in f)
        {
            foreach (var v in c)
            {
                fm = Math.Max(fm, Math.Abs(v));
            }
        }

        int fx = x0 + S + 20, half = S / 2;
        string[] cols = ["t-3", "t-2", "t-1", "now"];

        for (int ci = 0; ci < f.Length; ci++)
        {
            int cx = fx + (ci % 2) * half, cy = top + (ci / 2) * half, cell = (half - 6) / k;

            for (int y = 0; y < k; y++)
            {
                for (int x = 0; x < k; x++)
                {
                    using var b = new SolidBrush(Diverge(f[ci][y * k + x] / fm));
                    g.FillRectangle(b, cx + x * cell, cy + 14 + y * cell, cell, cell);
                }
            }

            Text2(g, cols[ci], cx, cy, Dim, 8);
        }

        Text2(g, "filter, 4 frames (scaled to this filter)", fx, top + S + 2, Dim, 8);
    }

    /// <summary>
    /// Draws the learning curve: the win rate of the last 300 points, the mean Q-value, and a horizontal line at 9% for a random player.
    /// </summary>
    /// <param name="g"></param>
    /// <param name="r"></param>
    private void DrawChart(Graphics g, Rectangle r)
    {
        Text2(g, "LEARNING CURVE (from pong_log.csv)", 6, 4, Dim, 8.5f, FontStyle.Bold);
        Text2(g, "win rate of last 300 points", 250, 4, Pos, 8); Text2(g, "mean Q", 400, 4, Accent, 8); Text2(g, "random player 9%", 470, 4, Dim, 8);

        var plot = new Rectangle(40, 28, r.Width - 56, r.Height - 52);

        using (var p = new Pen(Color.FromArgb(60, 64, 76)))
        {
            for (int i = 0; i <= 4; i++)
            {
                g.DrawLine(p, plot.Left, plot.Top + plot.Height * i / 4, plot.Right, plot.Top + plot.Height * i / 4);
            }
        }

        for (int i = 0; i <= 4; i++)
        {
            Text2(g, $"{100 - i * 25}%", 6, plot.Top + plot.Height * i / 4 - 7, Dim, 7.5f);
        }

        using (var dash = new Pen(Dim) { DashStyle = DashStyle.Dash })
        {
            g.DrawLine(dash, plot.Left, plot.Bottom - plot.Height * 0.094f, plot.Right, plot.Bottom - plot.Height * 0.094f);
        }

        if (curve.Count < 2)
        {
            Text2(g, "no log yet: run  BrainPong train ... out  and open that folder", plot.Left + 10, plot.Top + 10, Dim);
            return;
        }

        int maxStep = curve[^1].step;
        float qMax = Math.Max(0.5f, curve.Max(c => c.q));

        PointF[] wl = curve.Select(c => new PointF(plot.Left + plot.Width * (float)c.step / maxStep, plot.Bottom - plot.Height * Math.Clamp(c.win, 0, 1))).ToArray();
        PointF[] ql = curve.Select(c => new PointF(plot.Left + plot.Width * (float)c.step / maxStep, plot.Bottom - plot.Height * Math.Clamp(c.q / qMax, 0, 1))).ToArray();

        using (var pq = new Pen(Accent, 1.5f)) g.DrawLines(pq, ql);
        using (var pw = new Pen(Pos, 2f)) g.DrawLines(pw, wl);

        Text2(g, $"step {maxStep:N0}   win {curve[^1].win * 100:F1}%   Q {curve[^1].q:F2} (Q axis 0..{qMax:F2})", plot.Left, plot.Bottom + 4, Fg, 8);
    }
}