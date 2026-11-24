namespace BrainPong;

/// <summary>
/// Progress report from pretraining. All losses are the weighted squared error per sample (lower is better).
/// </summary>
/// <param name="Batch">Which gradient update this report is for.</param>
/// <param name="Loss">Average error on the training samples since the last report.</param>
/// <param name="ValLoss">Error on a fixed set of samples the network never trains on. If this stays high while Loss falls, it is memorising.</param>
/// <param name="CopyLoss">The error of the lazy baseline "the next frame looks the same as the last one", on the same validation samples.
/// ValLoss must come in below this for the network to have learnt any real movement.</param>
internal sealed record PretrainStats(int Batch, float Loss, float ValLoss, float CopyLoss);
