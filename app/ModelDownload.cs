using System.Diagnostics;
using System.Net.Http.Headers;
using System.Security.Cryptography;

namespace Xdna1Chat;

/// <summary>A model the app can download: one file of the pinned Hugging Face revision.</summary>
public sealed record ModelInfo(string Name, string File, long Size, string Sha256, string Note);

public readonly record struct DownloadProgress(long Done, long Total, double BytesPerSecond);

/// <summary>Downloads LFM3-1.7B GGUFs from Hugging Face into %LOCALAPPDATA%\xdna1-llm\models. A pinned revision,
/// size and SHA-256 per file, so what runs is exactly what was tested; an interrupted download resumes.</summary>
public static class ModelDownload
{
    private const string Repo = "LiquidAI/LFM3-1.7B-GGUF";
    private const string Revision = "cd0c74c575b5c416f3011b5b4c298077f26ca8c2";

    public static readonly ModelInfo[] Models =
    [
        new("LFM3-1.7B Q4_K_M", "LFM3-1.7B-Q4_K_M.gguf", 1_099_627_808,
            "3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4", "Faster: about 14 tokens per second"),
        new("LFM3-1.7B Q6_K", "LFM3-1.7B-Q6_K.gguf", 1_427_638_304,
            "54e1bd947c77167823d45bf495f1c3e2d226ca7fb242f83cfa141ee3a6ab55c6", "Closer to full quality: about 12 tokens per second"),
    ];

    private static readonly HttpClient Http = CreateClient();

    public static string Dir => Path.Combine(EngineHost.DataDir, "models");

    public static string LocalPath(ModelInfo m) => Path.Combine(Dir, m.File);

    /// <summary>Downloaded already (judged by size; the hash was checked when it was downloaded).</summary>
    public static bool IsDownloaded(ModelInfo m) => new FileInfo(LocalPath(m)) is { Exists: true } f && f.Length == m.Size;

    public static async Task<string> DownloadAsync(ModelInfo m, IProgress<DownloadProgress> progress, CancellationToken ct)
    {
        var path = LocalPath(m);
        if (IsDownloaded(m)) return path;
        Directory.CreateDirectory(Dir);
        var part = path + ".part";
        long have = System.IO.File.Exists(part) ? new FileInfo(part).Length : 0;
        if (have > m.Size) { System.IO.File.Delete(part); have = 0; }
        long free = new DriveInfo(Path.GetPathRoot(Dir)!).AvailableFreeSpace;
        if (free < m.Size - have + (100L << 20))
            throw new IOException($"Not enough disk space: {Gb(m.Size - have)} needed on {Path.GetPathRoot(Dir)}, {Gb(free)} free.");

        using var sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        await using var file = new FileStream(part, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None, 1 << 20, useAsync: true);
        var buf = new byte[1 << 20];
        for (int n; have > 0 && (n = await file.ReadAsync(buf, ct)) > 0;)  // resuming: hash what is there
            sha.AppendData(buf, 0, n);

        using var req = new HttpRequestMessage(HttpMethod.Get, $"https://huggingface.co/{Repo}/resolve/{Revision}/{m.File}");
        if (have > 0) req.Headers.Range = new RangeHeaderValue(have, null);
        using var resp = await Http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct);
        if (resp.StatusCode == System.Net.HttpStatusCode.RequestedRangeNotSatisfiable)
            have = file.Length;  // already complete: verify below
        else
        {
            resp.EnsureSuccessStatusCode();
            if (have > 0 && resp.StatusCode != System.Net.HttpStatusCode.PartialContent)
            {
                have = 0;  // the server sent the whole file: start over
                file.SetLength(0);
                sha.GetHashAndReset();
            }
            file.Position = have;
            await using var body = await resp.Content.ReadAsStreamAsync(ct);
            var sw = Stopwatch.StartNew();
            long start = have, lastReport = 0;
            using var stall = CancellationTokenSource.CreateLinkedTokenSource(ct);
            while (true)
            {
                stall.CancelAfter(TimeSpan.FromSeconds(60));
                int n;
                try { n = await body.ReadAsync(buf, stall.Token); }
                catch (OperationCanceledException) when (!ct.IsCancellationRequested)
                {
                    throw new IOException("The download stalled (no data for a minute). Try again to resume.");
                }
                if (n == 0) break;
                await file.WriteAsync(buf.AsMemory(0, n), ct);
                sha.AppendData(buf, 0, n);
                have += n;
                if (sw.ElapsedMilliseconds - lastReport >= 200)
                {
                    lastReport = sw.ElapsedMilliseconds;
                    progress.Report(new(have, m.Size, (have - start) / Math.Max(sw.Elapsed.TotalSeconds, 0.001)));
                }
            }
        }
        progress.Report(new(have, m.Size, 0));
        await file.FlushAsync(ct);
        file.Close();

        var hash = Convert.ToHexString(sha.GetHashAndReset()).ToLowerInvariant();
        if (have != m.Size || hash != m.Sha256)
        {
            System.IO.File.Delete(part);
            throw new InvalidDataException($"{m.File} did not download correctly ({have:N0} of {m.Size:N0} bytes, checksum "
                + (hash == m.Sha256 ? "ok" : "mismatch") + "). Try again.");
        }
        System.IO.File.Move(part, path, overwrite: true);
        return path;
    }

    public static string Gb(long bytes) => $"{bytes / 1e9:0.0} GB";

    private static HttpClient CreateClient()
    {
        var http = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };  // cancelled by the caller, not by size
        http.DefaultRequestHeaders.UserAgent.ParseAdd("xdna1-chat/0.1");
        return http;
    }
}
