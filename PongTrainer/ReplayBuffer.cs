namespace BrainPong;

/// <summary>
/// Experience replay. Stores every step as (frame, action, reward, done) in a ring; a "state" (the 4-frame stack the
/// network sees) is rebuilt on demand from the last 4 frames of the same episode, so each frame is stored once.
/// 
/// Training samples RANDOM past steps: consecutive frames are nearly identical, and learning from them in order
/// would make the gradient steps all push the same way (strongly correlated data), which destabilises SGD.
/// </summary>
internal sealed class ReplayBuffer
{
    /// <summary>
    /// Capacity of buffer.
    /// </summary>
    private readonly int _cap;
    private readonly int _frames;
    private readonly int _area;

    // The actual data: a ring buffer of frames, actions, rewards, and done flags. The oldest entry is at _head - _count.

    private readonly byte[][] _frame;
    private readonly byte[] _action;
    private readonly float[] _reward;
    private readonly bool[] _done, _start;

    private int _head, _count;

    // Composition diagnostics: what is the buffer actually made of?

    /// <summary>
    /// steps since its point started, when this step was taken
    /// </summary>
    private readonly ushort[] _age;

    /// <summary>
    /// Entries that are a scored point / entries taken >= LongAge steps into a rally
    /// </summary>
    private int _rewardCnt, _longCnt;

    public const int LongAge = 100;

    public int Count => _count;

    /// <summary>
    /// Fraction of stored steps that end a point with a +1/-1 (the only steps that carry real reward).
    /// </summary>
    public float RewardFrac => _count == 0 ? 0 : _rewardCnt / (float)_count;

    /// <summary>
    /// Fraction of stored steps taken 100+ steps into a rally (near-identical "keep returning" frames).
    /// </summary>
    public float LongFrac => _count == 0 ? 0 : _longCnt / (float)_count;

    /// <summary>
    /// Constructor. Creates all the structures needed to make a replay buffer.
    /// </summary>
    /// <param name="capacity"></param>
    /// <param name="frames"></param>
    /// <param name="area"></param>
    public ReplayBuffer(int capacity, int frames, int area)
    {
        _cap = capacity;
        _frames = frames;
        _area = area;

        _frame = new byte[_cap][];
        _action = new byte[_cap];
        _reward = new float[_cap];
        _done = new bool[_cap];
        _start = new bool[_cap];

        for (int i = 0; i < _cap; i++)
        {
            _frame[i] = new byte[area];
        }

        _age = new ushort[_cap];
    }

    /// <summary>
    /// Record one step: the frame the agent saw, what it did, what came of it. stepInPoint = steps since the point began.
    /// </summary>
    public void Add(byte[] f, int a, float r, bool isDone, bool isStart, int stepInPoint = 0)
    {
        if (_count == _cap)
        {
            if (_reward[_head] != 0) _rewardCnt--;

            if (_age[_head] >= LongAge) _longCnt--;
        }

        // the entry being overwritten leaves

        Array.Copy(f, _frame[_head], _area);

        _action[_head] = (byte)a;
        _reward[_head] = r;
        _done[_head] = isDone;
        _start[_head] = isStart;

        _age[_head] = (ushort)Math.Min(stepInPoint, ushort.MaxValue);

        if (r != 0) _rewardCnt++;

        if (stepInPoint >= LongAge) _longCnt++;

        _head = (_head + 1) % _cap;

        if (_count < _cap) _count++;
    }

    /// <summary>
    /// Returns the physical index in the ring buffer of the logical index (0 = oldest, _count-1 = newest). The ring wraps around at _cap.
    /// </summary>
    /// <param name="logical"></param>
    /// <returns></returns>
    private int PhysicalIndexInRingBuffer(int logical) => ((_head - _count + _cap) % _cap + logical) % _cap;

    /// <summary>
    /// Fill dst with the stacked state ending at logical index i (oldest frame first). At an episode start the first frame is repeated.
    /// </summary>
    public void Stack(int i, float[] dst)
    {
        int j = i;

        bool stopped = false;

        for (int k = 0; k < _frames; k++)
        {
            var src = _frame[PhysicalIndexInRingBuffer(j)];
            int o = (_frames - 1 - k) * _area;

            for (int p = 0; p < _area; p++)
            {
                dst[o + p] = src[p];
            }

            if (!stopped)
            {
                if (_start[PhysicalIndexInRingBuffer(j)] || j == 0)
                {
                    stopped = true;
                }
                else
                {
                    j--;
                }
            }
        }
    }

    /// <summary>
    /// Pick a random transition. Returns false if the buffer is too small. For a terminal step the next state is not
    /// needed (its value is 0), so next is left untouched and isDone is true.
    /// </summary>
    public bool Sample(Random rng, float[] state, float[] next, out int a, out float r, out bool isDone)
        => Sample(rng, state, next, 1, 1f, out a, out r, out isDone, out _);

    /// <summary>
    /// N-step version: r = r_i + g*r_{i+1} + ... + g^(n-1)*r_{i+n-1}, cut short at the end of the point (then isDone, no bootstrap).
    /// Otherwise next = state n steps later and discount = g^n is what the bootstrap value must be multiplied by.
    /// n = 1 reproduces the plain one-step sample exactly (same rng draw, same values).
    /// </summary>
    public bool Sample(Random rng, float[] state, float[] next, int n, float gamma, out int a, out float r, out bool isDone, out float discount)
    {
        a = 0;
        r = 0;

        isDone = false;
        discount = gamma;

        if (_count < 10 + n) return false;

        int i = rng.Next(_count - n); // the newest n steps lack a full lookahead

        Stack(i, state);

        a = _action[PhysicalIndexInRingBuffer(i)];

        float g = 1f;

        for (int k = 0; k < n; k++)
        {
            int p = PhysicalIndexInRingBuffer(i + k);

            r += g * _reward[p]; 
            g *= gamma;

            if (_done[p])
            {
                isDone = true;
                break;
            }
        }

        discount = g; // gamma^(steps actually summed); only used when not terminal, where it is gamma^n

        if (!isDone) Stack(i + n, next);

        return true;
    }
}