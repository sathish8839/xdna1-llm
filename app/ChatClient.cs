using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json.Nodes;

namespace Xdna1Chat;

/// <summary>Streams /v1/chat/completions as parsed SSE chunks. Cancelling drops the connection, which stops the
/// engine's generation.</summary>
public sealed class ChatClient(string baseUrl)
{
    private static readonly HttpClient Http = new() { Timeout = Timeout.InfiniteTimeSpan };

    public async IAsyncEnumerable<JsonObject> StreamAsync(JsonObject body, [EnumeratorCancellation] CancellationToken ct)
    {
        using var req = new HttpRequestMessage(HttpMethod.Post, baseUrl + "/v1/chat/completions")
        {
            Content = new StringContent(body.ToJsonString(), Encoding.UTF8, "application/json"),
        };
        using var resp = await Http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct);
        if (!resp.IsSuccessStatusCode)
        {
            var text = await resp.Content.ReadAsStringAsync(ct);
            string? msg = null;
            try { msg = (string?)JsonNode.Parse(text)?["error"]?["message"]; } catch (System.Text.Json.JsonException) { }
            throw new InvalidOperationException(msg ?? $"HTTP {(int)resp.StatusCode}: {text}");
        }
        using var reader = new StreamReader(await resp.Content.ReadAsStreamAsync(ct), Encoding.UTF8);
        while (await reader.ReadLineAsync(ct) is { } line)
        {
            if (!line.StartsWith("data: ", StringComparison.Ordinal)) continue;
            var data = line[6..];
            if (data == "[DONE]") yield break;
            if (JsonNode.Parse(data) is JsonObject chunk) yield return chunk;
        }
    }
}
