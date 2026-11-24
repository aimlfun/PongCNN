using System.Drawing.Drawing2D;
using System.Globalization;
using BrainPong;

namespace BrainPongViewer;

/// <summary>
/// Popup that answers "why did it pick that move?". It draws the network as a flow:
/// 4 input frames -> 8 conv1 units -> 16 conv2 units -> Q(action).
/// A line's thickness is how much that connection pushed the explained value, its colour the direction
/// (blue = pushed it up, orange = pushed it down). Each box is the picture that unit produced for the current state.
/// You can explain one Q-value or the difference between two (why up beats down), keep the scales fixed so frames are
/// comparable, and click a conv box to isolate the lines through it.
/// </summary>
internal sealed class FlowForm : Form
{
    private static readonly Color Fg = Color.FromArgb(220, 224, 232), Dim = Color.FromArgb(130, 136, 150);
    private static readonly Color Pos = Color.FromArgb(40, 150, 255), Neg = Color.FromArgb(255, 90, 20), Accent = Color.FromArgb(120, 220, 140);

    private readonly QNet net;
    private readonly Workspace ws;
    private readonly Func<float[]> theta;
    private readonly Func<float[]> q;
    private readonly Func<int> chosen;

    private readonly Canvas canvas = new() { Dock = DockStyle.Fill };
    private readonly ComboBox which = new() { DropDownStyle = ComboBoxStyle.DropDownList, Width = 170 };
    private readonly ComboBox view = new() { DropDownStyle = ComboBoxStyle.DropDownList, Width = 150 };
    private readonly CheckBox fixedScale = new() { Text = "Fixed scale (compare frames)", AutoSize = true, Checked = true, Padding = new Padding(10, 2, 0, 0) };

    // running maxima used when "fixed scale" is on, so a faint frame looks faint instead of being stretched to full strength
    private float hSal, hIn, hC1, hC2;
    private readonly float[] hMap1 = new float[64], hMap2 = new float[64];

    // Q history, one entry per new game frame (detected by the main window handing over a new Q array)
    private const int HistMax = 150;
    private readonly List<(float[] q, int act, float off, bool toward)> hist = [];   // toward = ball moved towards the agent (right) since the previous frame   // off = ball y minus agent paddle centre y (NaN if either is not visible)
    private float[]? lastQ;

    // clicked conv unit: layer 0 = none, 1 = conv1, 2 = conv2
    private int selLayer, selIdx;
    private Rectangle[] c1Rects = [], c2Rects = [];

    /// <summary>
    /// Smallest line (as a fraction of the strongest in its stage) that is still drawn. Keeps 128 conv1->conv2 lines readable.
    /// </summary>
    private const float Cutoff = 0.06f;

    public FlowForm(QNet net, Workspace ws, Func<float[]> theta, Func<float[]> q, Func<int> chosen)
    {
        this.net = net; this.ws = ws; this.theta = theta; this.q = q; this.chosen = chosen;

        Text = "BrainPong: why did it pick that move? (contribution flow)";
        BackColor = Color.FromArgb(24, 26, 32);
        ClientSize = new Size(1180, 800);
        MinimumSize = new Size(900, 640);

        var bar = new FlowLayoutPanel { Dock = DockStyle.Top, Height = 36, Padding = new Padding(6, 6, 6, 0), BackColor = Color.FromArgb(34, 36, 44) };
        bar.Controls.Add(new Label { Text = "Explain:", AutoSize = true, ForeColor = Fg, Padding = new Padding(0, 4, 0, 0) });
        which.Items.AddRange(["the chosen move", "stay", "up", "down", "chosen vs runner-up", "up vs down", "up vs stay", "down vs stay"]);
        which.SelectedIndex = 0;
        which.SelectedIndexChanged += (_, _) => ResetScale();
        bar.Controls.Add(which);

        bar.Controls.Add(new Label { Text = "Input view:", AutoSize = true, ForeColor = Fg, Padding = new Padding(10, 4, 0, 0) });
        view.Items.AddRange(["Overlay (fading trail)", "4 separate frames"]);
        view.SelectedIndex = 0;
        bar.Controls.Add(view);

        fixedScale.ForeColor = Fg;
        fixedScale.CheckedChanged += (_, _) => ResetScale();
        bar.Controls.Add(fixedScale);

        var reset = new Button { Text = "Reset scale", AutoSize = true, ForeColor = Fg, BackColor = Color.FromArgb(54, 57, 68), FlatStyle = FlatStyle.Flat };
        reset.Click += (_, _) => ResetScale();
        bar.Controls.Add(reset);
        Controls.Add(bar);

        canvas.MouseDown += OnCanvasMouseDown;
        canvas.Draw = DrawAll;
        Controls.Add(canvas);
        canvas.BringToFront();
    }

    /// <summary>
    /// Called by the main window every tick.
    /// </summary>
    public void Redraw() => canvas.Invalidate();

    private void ResetScale()
    {
        hSal = hIn = hC1 = hC2 = 0;
        Array.Clear(hMap1); Array.Clear(hMap2);
    }

    /// <summary>
    /// With fixed scale on, the scale only grows (running maximum); otherwise each frame is scaled to itself.
    /// </summary>
    private float Held(ref float held, float cur) => fixedScale.Checked ? (held = Math.Max(held, cur)) : cur;

    private void OnCanvasMouseDown(object? sender, MouseEventArgs e)
    {
        int layer = 0, idx = 0;

        for (int i = 0; i < c1Rects.Length; i++) if (c1Rects[i].Contains(e.Location)) { layer = 1; idx = i; }
        for (int j = 0; j < c2Rects.Length; j++) if (c2Rects[j].Contains(e.Location)) { layer = 2; idx = j; }

        if (layer == 0 || (layer == selLayer && idx == selIdx)) { selLayer = 0; selIdx = 0; }
        else { selLayer = layer; selIdx = idx; }

        canvas.Invalidate();
    }

    /// <summary>
    /// Which action(s) to explain: (a, -1) = Q(a); (a, b) = Q(a) - Q(b), i.e. why a beats b.
    /// </summary>
    private (int a, int b) Pair(float[] qv)
    {
        int m = which.SelectedIndex, ch = chosen();

        switch (m)
        {
            case 0: return (ch, -1);
            case 1: case 2: case 3: return (m - 1, -1);
            case 4:
                int best = -1;
                for (int i = 0; i < qv.Length; i++) if (i != ch && (best < 0 || qv[i] > qv[best])) best = i;
                return (ch, best);
            case 5: return (1, 2);
            case 6: return (1, 0);
            default: return (2, 0);
        }
    }

    /// <summary>
    /// Attribution is linear in the output gradient, so explaining Q(a) - Q(b) is exactly the difference of the two explanations
    /// (and the output biases cancel).
    /// </summary>
    private static Attribution Diff(Attribution x, Attribution y)
    {
        static float[][] G(float[][] p, float[][] q) => p.Select((row, i) => row.Select((v, j) => v - q[i][j]).ToArray()).ToArray();

        return new Attribution
        {
            InToC1 = G(x.InToC1, y.InToC1),
            C1ToC2 = G(x.C1ToC2, y.C1ToC2),
            C2ToQ = x.C2ToQ.Zip(y.C2ToQ, (p, q) => p - q).ToArray(),
            Saliency = x.Saliency.Zip(y.Saliency, (p, q) => p - q).ToArray(),
            Action = x.Action,
        };
    }

    /// <summary>
    /// Centre of the lit pixels of the ball (paddle = false) or the agent paddle (paddle = true) in frame f, or null if none.
    /// </summary>
    private (float x, float y)? Blob(int f, bool paddle)
    {
        int size = net.Spec.Size, area = size * size;
        float sx = 0, sy = 0; int n = 0;

        for (int p = 0; p < area; p++)
        {
            if (ws.Input[f * area + p] <= 0) continue;
            int x = p % size, y = p / size;
            bool isPad = x >= size - 4;

            if (isPad != paddle || (!paddle && x < 4)) continue;

            sx += x; sy += y; n++;
        }

        return n == 0 ? null : (sx / n, sy / n);
    }

    /// <summary>
    /// Plain-English reading of the current state, worked out from the frames and the saliency. Descriptive, not causal.
    /// The agent is the right-hand paddle (x >= size-4), the opponent the left-hand one (x &lt; 4), anything between is the ball.
    /// </summary>
    private string[] Summarise(Attribution attr, string what, float[] qv)
    {
        int F = net.Spec.Frames, size = net.Spec.Size, area = size * size;
        string[] names = ["stay", "up", "down"];

        (float x, float y)? Centroid(int f, bool paddle) => Blob(f, paddle);

        var ball = Centroid(F - 1, false);
        var pad = Centroid(F - 1, true);
        (float x, float y)? first = null;

        for (int f = 0; f < F - 1 && first == null; f++) first = Centroid(f, false);

        string l1;

        if (ball is null) l1 = "Ball: not visible as a separate blob in the newest frame (it is touching a paddle).";
        else
        {
            string dir = "";

            if (first is { } b0)
            {
                float dx = ball.Value.x - b0.x, dy = ball.Value.y - b0.y;
                dir = ", moving " + (Math.Abs(dx) < 0.5f ? "no sideways" : dx > 0 ? "TOWARD the agent (right)" : "AWAY to the opponent (left)")
                    + " and " + (Math.Abs(dy) < 0.5f ? "level" : dy > 0 ? "down" : "up");
            }

            string rel = "";

            if (pad is { } p0)
            {
                float d = ball.Value.y - p0.y;
                rel = Math.Abs(d) < 1.5f ? ". Ball is level with the agent paddle" : $". Ball is {Math.Abs(d):0} px {(d < 0 ? "above" : "below")} the agent paddle centre";
            }

            l1 = $"Ball at x={ball.Value.x:0}, y={ball.Value.y:0}{dir}{rel}.";
        }

        // net push by what each lit pixel belongs to, and how much each frame mattered
        float sb = 0, sp = 0, so = 0, tot = 1e-9f;
        var perFrame = new float[F];

        for (int f = 0; f < F; f++)
            for (int p = 0; p < area; p++)
            {
                float v = attr.Saliency[f * area + p];
                int x = p % size;

                if (x < 4) so += v; else if (x >= size - 4) sp += v; else sb += v;

                perFrame[f] += Math.Abs(v); tot += Math.Abs(v);
            }

        string frames = string.Join(", ", Enumerable.Range(0, F).Select(f => $"{(f == F - 1 ? "now" : "t-" + (F - 1 - f))} {100 * perFrame[f] / tot:0}%"));
        string l2 = $"Net push on {what}: ball pixels {sb:+0.00;-0.00}, agent paddle {sp:+0.00;-0.00}, opponent paddle {so:+0.00;-0.00}.  Weight by frame: {frames}.";

        int top = 0;
        for (int i = 1; i < qv.Length; i++) if (qv[i] > qv[top]) top = i;

        string l3 = "";

        if (ball is { } bl && pad is { } pd)
        {
            float d = bl.y - pd.y;
            int chase = Math.Abs(d) < 1.5f ? 0 : d < 0 ? 1 : 2;
            l3 = $"A simple chase rule (go to the ball's current height) says {names[chase]}; the network's highest Q is {names[top]}"
               + (chase == top ? " (agrees)." : " (differs: it is using the ball's motion / where it will be, not just its height).");
        }

        return [l1, l2, l3];
    }

    private static void Text2(Graphics g, string s, float x, float y, Color c, float size = 9f, FontStyle st = FontStyle.Regular)
    {
        using var f = new Font("Segoe UI", size, st); using var b = new SolidBrush(c);
        g.DrawString(s, f, b, x, y);
    }

    private static Color Diverge(float v, Color neutral)
    {
        v = Math.Clamp(v, -1f, 1f);
        var c = v >= 0 ? Pos : Neg; float t = Math.Abs(v);

        return Color.FromArgb((int)(neutral.R + (c.R - neutral.R) * t), (int)(neutral.G + (c.G - neutral.G) * t), (int)(neutral.B + (c.B - neutral.B) * t));
    }

    /// <summary>
    /// Grey picture of one unit's activations (bright = strong), scaled to <paramref name="mx"/>.
    /// </summary>
    private static void DrawMap(Graphics g, float[] m, int h, int w, Rectangle dest, float mx)
    {
        mx = Math.Max(1e-6f, mx);
        using var bmp = new Bitmap(w, h);

        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int v = (int)(255 * Math.Clamp(m[y * w + x] / mx, 0, 1));
                bmp.SetPixel(x, y, Color.FromArgb(v, v, v));
            }

        g.DrawImage(bmp, dest);
        g.DrawRectangle(Pens.DimGray, dest);
    }

    /// <summary>
    /// A curve from the right edge of one box to the left edge of the next. Thickness = |contribution|, colour = sign.
    /// </summary>
    private static void Edge(Graphics g, float x0, float y0, float x1, float y1, float c, float max, float cut)
    {
        float t = Math.Min(1f, Math.Abs(c) / max);

        if (t < cut) return;

        using var pen = new Pen(Color.FromArgb((int)(60 + 150 * t), c >= 0 ? Pos : Neg), 0.6f + 6f * t);
        float mx = (x0 + x1) / 2;
        g.DrawBezier(pen, x0, y0, mx, y0, mx, y1, x1, y1);
    }

    private static float MaxAbs(float[][] grid)
    {
        float m = 1e-9f;

        foreach (var row in grid) foreach (var v in row) m = Math.Max(m, Math.Abs(v));

        return m;
    }

    private void DrawAll(Graphics g, Rectangle r)
    {
        float[] qv = q();
        var (action, other) = Pair(qv);
        string[] names = ["stay", "up", "down"];

        if (!ReferenceEquals(qv, lastQ))
        {
            lastQ = qv;
            var bl = Blob(net.Spec.Frames - 1, false); var pd = Blob(net.Spec.Frames - 1, true);
            var bPrev = Blob(net.Spec.Frames - 2, false);
            bool toward = bl is { } bn && bPrev is { } bo ? bn.x > bo.x : hist.Count > 0 && hist[^1].toward;   // unknown -> keep the last direction
            hist.Add((qv, chosen(), bl is { } b && pd is { } p ? b.y - p.y : float.NaN, toward));
            if (hist.Count > HistMax) hist.RemoveAt(0);
        }

        var attr = net.Explain(theta(), ws, action);
        if (other >= 0) attr = Diff(attr, net.Explain(theta(), ws, other));

        var (m1, h1, w1) = net.Conv1Maps(ws);
        var (m2, h2, w2) = net.Conv2Maps(ws);

        int F = net.Spec.Frames, C1 = net.Spec.C1, C2 = net.Spec.C2, size = net.Spec.Size, area = size * size;

        string what = other >= 0 ? $"Q({names[action]}) - Q({names[other]})" : $"Q({names[action]})";
        float value = other >= 0 ? qv[action] - qv[other] : qv[action];

        Text2(g, $"Why {what} = {value:+0.000;-0.000}"
            + (other >= 0 ? $"   (what makes it prefer {names[action]} over {names[other]})" : action == chosen() ? "   (the move it took)" : ""), 8, 6, Accent, 11, FontStyle.Bold);
        Text2(g, "line thickness = how much that connection pushed this value;  blue = UP,  orange = DOWN.  Click a conv box to isolate its lines (click again to clear).", 8, 30, Dim, 8.5f);

        var lines = Summarise(attr, what, qv);
        for (int k = 0; k < lines.Length; k++) Text2(g, lines[k], 8, 48 + k * 16, k == 0 ? Fg : Dim, 9f);

        float top = 148, bottom = r.Height - 30, span = bottom - top;
        float[] xs = [r.Width * 0.03f, r.Width * 0.27f, r.Width * 0.55f, r.Width * 0.84f];
        bool overlay = view.SelectedIndex == 0;

        // ---- node boxes (positions first, so the lines can be drawn underneath) ----
        int fSize = (int)Math.Min(150, span / F - 14);
        int s1 = (int)Math.Min(64, span / C1 - 6);
        int s2 = (int)Math.Min(40, span / C2 - 4);
        int ovSize = (int)Math.Min(span - 40, xs[1] - xs[0] - 50);
        var ovBox = new Rectangle((int)xs[0], (int)(top + (span - ovSize) / 2f - 10), ovSize, ovSize);

        Rectangle FrameBox(int f) => new((int)xs[0], (int)(top + (f + 0.5f) * span / F - fSize / 2f), fSize, fSize);

        // where frame f's lines start: in overlay mode, spread down the right edge of the single box (top = oldest, bottom = newest)
        PointF Anchor(int f) => overlay
            ? new PointF(ovBox.Right, ovBox.Top + (f + 0.5f) * ovBox.Height / F)
            : new PointF(FrameBox(f).Right, FrameBox(f).Top + FrameBox(f).Height / 2f);
        Rectangle C1Box(int i) => new((int)xs[1], (int)(top + (i + 0.5f) * span / C1 - s1 / 2f), s1, s1);
        Rectangle C2Box(int j) => new((int)xs[2], (int)(top + (j + 0.5f) * span / C2 - s2 / 2f), s2, s2);
        var qBox = new Rectangle((int)xs[3], (int)top + 10, (int)Math.Min(200, r.Width - xs[3] - 10), 140);

        c1Rects = Enumerable.Range(0, C1).Select(C1Box).ToArray();
        c2Rects = Enumerable.Range(0, C2).Select(C2Box).ToArray();

        g.SmoothingMode = SmoothingMode.AntiAlias;

        // ---- lines (when a conv unit is selected: only its own lines, down to a tiny threshold) ----
        float max1 = Math.Max(1e-9f, Held(ref hIn, MaxAbs(attr.InToC1)));
        float max2 = Math.Max(1e-9f, Held(ref hC1, MaxAbs(attr.C1ToC2)));
        float max3 = Math.Max(1e-9f, Held(ref hC2, attr.C2ToQ.Max(Math.Abs)));
        float cut = selLayer == 0 ? Cutoff : 0.01f;
        int qRow = other >= 0 ? 3 : action;
        float qy = qBox.Top + 30 + qRow * 24;

        for (int f = 0; f < F; f++)
            for (int i = 0; i < C1; i++)
            {
                if (selLayer == 2 || (selLayer == 1 && i != selIdx)) continue;
                var a = Anchor(f); var b = C1Box(i);
                Edge(g, a.X, a.Y, b.Left, b.Top + b.Height / 2f, attr.InToC1[f][i], max1, cut);
            }

        for (int i = 0; i < C1; i++)
            for (int j = 0; j < C2; j++)
            {
                if ((selLayer == 1 && i != selIdx) || (selLayer == 2 && j != selIdx)) continue;
                var a = C1Box(i); var b = C2Box(j);
                Edge(g, a.Right, a.Top + a.Height / 2f, b.Left, b.Top + b.Height / 2f, attr.C1ToC2[i][j], max2, cut);
            }

        for (int j = 0; j < C2; j++)
        {
            if (selLayer == 1 || (selLayer == 2 && j != selIdx)) continue;
            var a = C2Box(j);
            Edge(g, a.Right, a.Top + a.Height / 2f, qBox.Left, qy, attr.C2ToQ[j], max3, cut);
        }

        // ---- input frames, tinted by how much each lit pixel pushed the value ----
        float smax = 1e-9f;
        foreach (var v in attr.Saliency) smax = Math.Max(smax, Math.Abs(v));
        smax = Math.Max(1e-9f, Held(ref hSal, smax));

        string[] cols = ["t-3", "t-2", "t-1", "now"];
        g.SmoothingMode = SmoothingMode.None;

        if (overlay)
        {
            // all frames on one picture: the newest lit frame at a pixel wins, older frames are progressively fainter,
            // so the ball's path reads as a fading trail ending at its current position
            using var ov = new Bitmap(size, size);
            var bg = Color.FromArgb(10, 10, 12);

            for (int p = 0; p < area; p++)
            {
                var px = bg;

                for (int f = F - 1; f >= 0; f--)
                {
                    if (ws.Input[f * area + p] <= 0) continue;

                    float s = attr.Saliency[f * area + p] / smax, a = Math.Abs(s), t = a < 0.02f ? 0f : Math.Max(0.45f, MathF.Cbrt(a));
                    var c = Diverge(Math.Sign(s) * t, Color.FromArgb(200, 200, 205));
                    float fade = MathF.Pow(0.62f, F - 1 - f);

                    px = Color.FromArgb((int)(bg.R + (c.R - bg.R) * fade), (int)(bg.G + (c.G - bg.G) * fade), (int)(bg.B + (c.B - bg.B) * fade));
                    break;
                }

                ov.SetPixel(p % size, p / size, px);
            }

            g.InterpolationMode = InterpolationMode.NearestNeighbor;
            g.PixelOffsetMode = PixelOffsetMode.Half;
            g.DrawImage(ov, ovBox);
            g.DrawRectangle(Pens.DimGray, ovBox);
            Text2(g, "t-3 ... now overlaid: older = fainter", ovBox.Left, ovBox.Top - 16, Dim, 7.5f);
            Text2(g, "dots on the right edge = where each frame's\nlines start (top = t-3, bottom = now)", ovBox.Left, ovBox.Bottom + 4, Dim, 7.5f);

            for (int f = 0; f < F; f++)
            {
                var pt = Anchor(f);
                g.SmoothingMode = SmoothingMode.AntiAlias;
                g.FillEllipse(Brushes.WhiteSmoke, pt.X - 3, pt.Y - 3, 6, 6);
                g.SmoothingMode = SmoothingMode.None;
            }
        }

        for (int f = 0; f < F && !overlay; f++)
        {
            var box = FrameBox(f);
            using var bmp = new Bitmap(size, size);

            for (int p = 0; p < area; p++)
            {
                bool on = ws.Input[f * area + p] > 0;
                float s = attr.Saliency[f * area + p] / smax;

                // strong tint: cube-root boosts weak pushes, and anything above noise gets at least 45% colour
                float a = Math.Abs(s), t = a < 0.02f ? 0f : Math.Max(0.45f, MathF.Cbrt(a));

                bmp.SetPixel(p % size, p / size, on ? Diverge(Math.Sign(s) * t, Color.FromArgb(200, 200, 205)) : Color.FromArgb(10, 10, 12));
            }

            g.InterpolationMode = InterpolationMode.NearestNeighbor;
            g.PixelOffsetMode = PixelOffsetMode.Half;
            g.DrawImage(bmp, box);
            g.DrawRectangle(Pens.DimGray, box);
            Text2(g, cols[f], box.Left, box.Top - 14, Dim, 7.5f);
        }

        // ---- conv1 and conv2 units ----
        for (int i = 0; i < C1; i++) { DrawMap(g, m1[i], h1, w1, C1Box(i), Held(ref hMap1[i], m1[i].Max())); Text2(g, $"#{i}", C1Box(i).Right + 3, C1Box(i).Top, Dim, 7); }
        for (int j = 0; j < C2; j++) { DrawMap(g, m2[j], h2, w2, C2Box(j), Held(ref hMap2[j], m2[j].Max())); Text2(g, $"#{j}", C2Box(j).Right + 3, C2Box(j).Top + 6, Dim, 7); }

        if (selLayer != 0)
        {
            using var sel = new Pen(Accent, 2.5f);
            var rc = selLayer == 1 ? C1Box(selIdx) : C2Box(selIdx);
            rc.Inflate(2, 2);
            g.DrawRectangle(sel, rc);
        }

        // ---- Q outputs ----
        g.SmoothingMode = SmoothingMode.AntiAlias;
        Text2(g, "Q-values", qBox.Left, qBox.Top, Dim, 8.5f, FontStyle.Bold);
        float qlim = Math.Max(0.3f, qv.Max(Math.Abs));

        for (int a = 0; a < 3; a++)
        {
            int y = qBox.Top + 24 + a * 24, mid = qBox.Left + 80, half = 60;
            float w = qv[a] / qlim * half;
            bool hi = a == action || a == other;
            Text2(g, names[a], qBox.Left, y, hi ? Accent : Fg, 9, hi ? FontStyle.Bold : FontStyle.Regular);
            g.DrawLine(Pens.DimGray, mid, y, mid, y + 16);
            using var br = new SolidBrush(qv[a] >= 0 ? Pos : Neg);
            g.FillRectangle(br, w >= 0 ? mid : mid + w, y + 1, Math.Abs(w), 14);
            Text2(g, qv[a].ToString("+0.00;-0.00", CultureInfo.InvariantCulture), mid + (w >= 0 ? w + 3 : 3), y, Fg, 7.5f);
        }

        if (other >= 0)
        {
            int y = qBox.Top + 24 + 3 * 24, mid = qBox.Left + 80, half = 60;
            float d = value / qlim * half;
            Text2(g, "difference", qBox.Left - 6, y, Accent, 8.5f, FontStyle.Bold);
            g.DrawLine(Pens.DimGray, mid, y, mid, y + 16);
            using var br = new SolidBrush(d >= 0 ? Pos : Neg);
            g.FillRectangle(br, d >= 0 ? mid : mid + d, y + 1, Math.Abs(d), 14);
            Text2(g, value.ToString("+0.00;-0.00", CultureInfo.InvariantCulture), mid + (d >= 0 ? d + 3 : 3), y, Fg, 7.5f);
        }

        // ---- chart 1: how good does it think the situation is? (the best Q; it falls towards -1 when the point is lost) ----
        var va = new Rectangle(qBox.Left, qBox.Bottom + 34, Math.Max(60, r.Width - qBox.Left - 12), 70);
        Text2(g, $"Best Q = expected result of this", va.Left, va.Top - 42, Dim, 8f, FontStyle.Bold);
        Text2(g, $"point now: +1 win, -1 lose ", va.Left, va.Top - 30, Dim, 7.5f, FontStyle.Bold);
        Text2(g, $"(last {HistMax} frames)", va.Left, va.Top - 18, Dim, 7.5f, FontStyle.Bold);
        g.FillRectangle(new SolidBrush(Color.FromArgb(30, 32, 40)), va);
        g.DrawRectangle(Pens.DimGray, va);

        float XOf(int i) => va.Left + 4 + i * (va.Width - 8f) / (HistMax - 1);
        float VY(float v) => va.Top + 4 + (1 - (Math.Clamp(v, -1.1f, 1.1f) + 1.1f) / 2.2f) * (va.Height - 8);
        int off = HistMax - hist.Count;
        using (var zero0 = new Pen(Color.FromArgb(70, 74, 86), 1f) { DashStyle = DashStyle.Dash }) g.DrawLine(zero0, va.Left, VY(0), va.Right, VY(0));
        Text2(g, "+1 win", va.Right - 40, va.Top, Dim, 7); Text2(g, "-1 lose", va.Right - 44, va.Bottom - 13, Dim, 7);

        if (hist.Count > 1)
        {
            using var vp = new Pen(Color.FromArgb(120, 190, 255), 2f);
            g.DrawLines(vp, hist.Select((h, i) => new PointF(XOf(off + i), VY(h.q.Max()))).ToArray());
        }

        // ---- chart 2: which move does it prefer? each Q minus the average of the three, auto-scaled so small gaps are visible ----
        var ch = new Rectangle(va.Left, va.Bottom + 38, va.Width, 110);
        float lim = Math.Max(0.05f, hist.Count == 0 ? 0.05f : hist.Max(h => h.q.Max(v => Math.Abs(v - h.q.Average()))));
        Text2(g, $"Preference: each Q minus ", ch.Left, ch.Top - 30, Dim, 7.5f, FontStyle.Bold);
        Text2(g, $"the average (range +/-{lim:0.00})", ch.Left, ch.Top - 18, Dim, 7.5f, FontStyle.Bold);
        g.FillRectangle(new SolidBrush(Color.FromArgb(30, 32, 40)), ch);
        g.DrawRectangle(Pens.DimGray, ch);

        float YOf(float v) => ch.Top + 6 + (1 - (v + lim) / (2 * lim)) * (ch.Height - 28);
        using (var zero = new Pen(Color.FromArgb(70, 74, 86), 1f) { DashStyle = DashStyle.Dash }) g.DrawLine(zero, ch.Left, YOf(0), ch.Right, YOf(0));

        // three clearly different hues (cyan / magenta / amber); stay is also dashed so it never relies on colour alone
        Color[] qc = [Color.FromArgb(255, 170, 40), Color.FromArgb(0, 225, 235), Color.FromArgb(240, 70, 220)];

        // draw the explained action last so it sits on top
        foreach (int a in new[] { 0, 1, 2 }.OrderBy(a => a == action))
        {
            if (hist.Count < 2) break;
            using var pen = new Pen(qc[a], a == action ? 2.6f : 1.6f) { DashStyle = a == 0 ? DashStyle.Dash : DashStyle.Solid };
            g.DrawLines(pen, hist.Select((h, i) => new PointF(XOf(off + i), YOf(h.q[a] - h.q.Average()))).ToArray());
        }

        // strip under the chart: which move was actually taken each frame
        for (int i = 0; i < hist.Count; i++)
        {
            using var br = new SolidBrush(qc[hist[i].act]);
            g.FillRectangle(br, XOf(off + i) - 1, ch.Bottom - 16, 2.5f, 10);
        }

        for (int a = 0; a < 3; a++)
        {
            using var br = new SolidBrush(qc[a]);
            g.FillRectangle(br, ch.Left + a * 62, ch.Bottom + 6, 10, 10);
            Text2(g, names[a], ch.Left + a * 62 + 13, ch.Bottom + 2, Fg, 8);
        }

        Text2(g, "bottom strip = move taken", ch.Left, ch.Bottom + 18, Dim, 7.5f);

        // ---- ball height relative to the agent paddle, on the same time axis as the Q chart ----
        var bc = new Rectangle(ch.Left, ch.Bottom + 58, ch.Width, 100);
        Text2(g, "Ball vs paddle px, same time axis", bc.Left, bc.Top - 18, Dim, 8f, FontStyle.Bold);
        g.FillRectangle(new SolidBrush(Color.FromArgb(30, 32, 40)), bc);
        g.DrawRectangle(Pens.DimGray, bc);

        const float OffLim = 20f;   // the court is 40 px tall
        float BY(float v) => bc.Top + 4 + ((Math.Clamp(v, -OffLim, OffLim) + OffLim) / (2 * OffLim)) * (bc.Height - 8);   // pixel y grows downwards: ball below paddle (+) plots LOWER
        using (var zero2 = new Pen(Color.FromArgb(120, 124, 136), 1f) { DashStyle = DashStyle.Dash }) g.DrawLine(zero2, bc.Left, BY(0), bc.Right, BY(0));
        // faint dotted lines where an edge hit happens: r7 appears to aim there on purpose to put an angle on the ball (hypothesis, see CHANGELOG)
        using (var edgePen = new Pen(Color.FromArgb(60, 64, 78), 1f) { DashStyle = DashStyle.Dot }) { g.DrawLine(edgePen, bc.Left, BY(2.8f), bc.Right, BY(2.8f)); g.DrawLine(edgePen, bc.Left, BY(-2.8f), bc.Right, BY(-2.8f)); }
        Text2(g, "dotted = paddle edge (+/-2.8 px)", bc.Left + 95, bc.Bottom - 15, Color.FromArgb(90, 94, 108), 7);
        Text2(g, "ball below paddle", bc.Left + 2, bc.Bottom - 15, Dim, 7);
        Text2(g, "ball above paddle", bc.Left + 2, bc.Top + 1, Dim, 7);
        Text2(g, "level", bc.Right - 34, BY(0) - 13, Dim, 7);

        // white while the ball comes towards the agent (it has a decision to make), grey while it goes away (it does not)
        using (var towardPen = new Pen(Color.FromArgb(240, 240, 245), 2.4f))
        using (var awayPen = new Pen(Color.FromArgb(105, 108, 118), 2f))
        {
            for (int i = 1; i < hist.Count; i++)
            {
                if (float.IsNaN(hist[i].off) || float.IsNaN(hist[i - 1].off)) continue;

                g.DrawLine(hist[i].toward ? towardPen : awayPen, XOf(off + i - 1), BY(hist[i - 1].off), XOf(off + i), BY(hist[i].off));
            }
        }

        Text2(g, "white = ball coming towards the agent, grey = going away", bc.Left, bc.Bottom + 3, Dim, 7.5f);

        // ---- column headings ----
        Text2(g, "INPUT (blue = pushed up, ", xs[0], top - 40, Dim, 8f, FontStyle.Bold);
        Text2(g, "orange = down, pale = no effect)", xs[0], top - 28, Dim, 8f, FontStyle.Bold);
        Text2(g, "CONV1: 8 detectors", xs[1], top - 28, Dim, 8f, FontStyle.Bold);
        Text2(g, "CONV2: 16 combiners", xs[2], top - 28, Dim, 8f, FontStyle.Bold);
        Text2(g, "OUTPUT", xs[3], top - 28, Dim, 8f, FontStyle.Bold);

        Text2(g, "Each stage has its own scale" + (fixedScale.Checked ? " (fixed: only grows, so frames are comparable; Reset scale to restart)" : " (re-fitted every frame)")
            + "; faint lines hidden. Biases ignored, so contributions add up to roughly Q.", 8, r.Height - 20, Dim, 7.5f);
    }
}