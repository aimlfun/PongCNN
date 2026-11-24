namespace BrainPong;

/// <summary>
/// Shape of the network. Defaults: 4 stacked 40x40 frames -> conv -> conv -> dense -> 3 Q-values.
/// </summary>
internal sealed record NetSpec(int Size = 40, int Frames = 4, int C1 = 8, int KernelSize1 = 6, int S1 = 2,
                               int C2 = 16, int KernelSize2 = 4, int S2 = 2, int Fc = 128, int Actions = 3);
