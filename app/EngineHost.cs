using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text.Json.Nodes;

namespace Xdna1Chat;

/// <summary>Runs `python -m xdna1_llm.server` on a free localhost port and stops it with the app.</summary>
/// <remarks>The process goes into a Windows job object that kills it when the app's handle closes, so the engine
/// (and the NPU contexts it holds) never outlives the app, even if the app crashes.</remarks>
public sealed class EngineHost : IDisposable
{
    private readonly HttpClient http = new() { Timeout = TimeSpan.FromSeconds(5) };
    private readonly IntPtr job = CreateKillOnCloseJob();
    private readonly Queue<string> tail = new();
    private Process? proc;
    private StreamWriter? log;

    public string BaseUrl { get; private set; } = "";
    public string Model { get; private set; } = "";
    public string Device { get; private set; } = "";

    public static string DataDir { get; } =
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "xdna1-llm");

    public static string LogPath => Path.Combine(DataDir, "engine.log");

    /// <summary>The engine's Python: settings, $XDNA1_LLM_PYTHON, python\ beside the app (installer layout), or the
    /// engine\.venv of a source checkout above the app.</summary>
    public static string? FindPython(string? configured)
    {
        foreach (var p in new[] { configured, Environment.GetEnvironmentVariable("XDNA1_LLM_PYTHON") })
            if (!string.IsNullOrEmpty(p) && File.Exists(p)) return p;
        var bundled = Path.Combine(AppContext.BaseDirectory, "python", "python.exe");
        if (File.Exists(bundled)) return bundled;
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d != null; d = d.Parent)
        {
            var venv = Path.Combine(d.FullName, "engine", ".venv", "Scripts", "python.exe");
            if (File.Exists(venv)) return venv;
        }
        return null;
    }

    public async Task StartAsync(string python, string gguf, CancellationToken ct)
    {
        Stop();
        Directory.CreateDirectory(DataDir);
        log = new StreamWriter(LogPath, append: false) { AutoFlush = true };
        tail.Clear();

        int port = FreePort();
        var psi = new ProcessStartInfo(python)
        {
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        foreach (var a in new[] { "-m", "xdna1_llm.server", "--gguf", gguf, "--port", port.ToString() })
            psi.ArgumentList.Add(a);
        psi.Environment["PYTHONUNBUFFERED"] = "1";
        psi.Environment["PYTHONIOENCODING"] = "utf-8";
        var kernels = Path.Combine(AppContext.BaseDirectory, "kernels");  // portable/installer layout
        if (Directory.Exists(kernels) && string.IsNullOrEmpty(Environment.GetEnvironmentVariable("XDNA1_LLM_KERNELS")))
            psi.Environment["XDNA1_LLM_KERNELS"] = kernels;

        proc = Process.Start(psi) ?? throw new InvalidOperationException($"could not start {python}");
        AssignProcessToJobObject(job, proc.Handle);
        proc.OutputDataReceived += (_, e) => Append(e.Data);
        proc.ErrorDataReceived += (_, e) => Append(e.Data);
        proc.BeginOutputReadLine();
        proc.BeginErrorReadLine();

        BaseUrl = $"http://127.0.0.1:{port}";
        while (true)
        {
            ct.ThrowIfCancellationRequested();
            if (proc.HasExited)
                throw new InvalidOperationException($"the engine stopped (exit code {proc.ExitCode}):\n{LastLines()}");
            try
            {
                var health = JsonNode.Parse(await http.GetStringAsync(BaseUrl + "/health", ct));
                Model = (string?)health?["model"] ?? "";
                Device = (string?)health?["device"] ?? "";
                return;
            }
            catch (HttpRequestException) { }  // still loading: the server listens once the model is ready
            await Task.Delay(500, ct);
        }
    }

    public void Stop()
    {
        if (proc is { HasExited: false })
        {
            try { proc.Kill(entireProcessTree: true); } catch (InvalidOperationException) { }
        }
        proc?.Dispose();
        proc = null;
        log?.Dispose();
        log = null;
    }

    public string LastLines()
    {
        lock (tail) return string.Join("\n", tail);
    }

    private void Append(string? line)
    {
        if (line == null) return;
        lock (tail)
        {
            tail.Enqueue(line);
            while (tail.Count > 12) tail.Dequeue();
            try { log?.WriteLine(line); } catch (ObjectDisposedException) { }
        }
    }

    private static int FreePort()
    {
        var l = new TcpListener(IPAddress.Loopback, 0);
        l.Start();
        int port = ((IPEndPoint)l.LocalEndpoint).Port;
        l.Stop();
        return port;
    }

    public void Dispose()
    {
        Stop();
        CloseHandle(job);
        http.Dispose();
    }

    // Job object that kills its processes when the last handle to it closes (the app exiting, or crashing).
    private static IntPtr CreateKillOnCloseJob()
    {
        var h = CreateJobObject(IntPtr.Zero, null);
        var info = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
        info.BasicLimitInformation.LimitFlags = 0x2000;  // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        int size = Marshal.SizeOf<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>();
        var ptr = Marshal.AllocHGlobal(size);
        try
        {
            Marshal.StructureToPtr(info, ptr, false);
            SetInformationJobObject(h, 9 /* JobObjectExtendedLimitInformation */, ptr, (uint)size);
        }
        finally { Marshal.FreeHGlobal(ptr); }
        return h;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct JOBOBJECT_BASIC_LIMIT_INFORMATION
    {
        public long PerProcessUserTimeLimit, PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize, MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass, SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct IO_COUNTERS
    {
        public ulong ReadOperationCount, WriteOperationCount, OtherOperationCount;
        public ulong ReadTransferCount, WriteTransferCount, OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
        public IO_COUNTERS IoInfo;
        public UIntPtr ProcessMemoryLimit, JobMemoryLimit, PeakProcessMemoryUsed, PeakJobMemoryUsed;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
    private static extern IntPtr CreateJobObject(IntPtr attributes, string? name);

    [DllImport("kernel32.dll")]
    private static extern bool SetInformationJobObject(IntPtr job, int infoClass, IntPtr info, uint length);

    [DllImport("kernel32.dll")]
    private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

    [DllImport("kernel32.dll")]
    private static extern bool CloseHandle(IntPtr handle);
}
