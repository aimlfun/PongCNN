namespace BrainPongViewer;

// Watches a BrainPong DQN play. Usage: BrainPongViewer [folder with pong_dqn.bin and pong_log.csv]
// The weights file is hot-reloaded while a training run keeps saving it, so you can watch the agent improve.
internal static class Program
{
    [STAThread]
    private static void Main(string[] args)
    {
        ApplicationConfiguration.Initialize();
        Application.Run(new MainForm(args.Length > 0 ? args[0] : Directory.GetCurrentDirectory()));
    }
}
