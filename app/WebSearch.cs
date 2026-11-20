using System.Net;
using System.Text;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace Xdna1Chat;

/// <summary>The `web_search` tool: offered to the model when web search is on, run here with DuckDuckGo's HTML
/// page, and answered with a tool message. The engine never touches the network.</summary>
public static partial class WebSearch
{
    public sealed record Result(string Title, string Url, string Snippet);

    private static readonly HttpClient Http = new() { Timeout = TimeSpan.FromSeconds(15) };

    public static JsonArray Tools() => new()
    {
        new JsonObject
        {
            ["type"] = "function",
            ["function"] = new JsonObject
            {
                ["name"] = "web_search",
                ["description"] = "Search the web for current or factual information. Returns titles, links and snippets.",
                ["parameters"] = new JsonObject
                {
                    ["type"] = "object",
                    ["properties"] = new JsonObject
                    {
                        ["query"] = new JsonObject { ["type"] = "string", ["description"] = "What to search for" },
                    },
                    ["required"] = new JsonArray("query"),
                },
            },
        },
    };

    public static async Task<List<Result>> SearchAsync(string query, CancellationToken ct, int max = 5)
    {
        using var req = new HttpRequestMessage(HttpMethod.Post, "https://html.duckduckgo.com/html/")
        {
            Content = new FormUrlEncodedContent(new Dictionary<string, string> { ["q"] = query }),
        };
        req.Headers.UserAgent.ParseAdd("Mozilla/5.0 (Windows NT 10.0; Win64; x64)");
        using var resp = await Http.SendAsync(req, ct);
        resp.EnsureSuccessStatusCode();
        var html = await resp.Content.ReadAsStringAsync(ct);
        var results = Parse(html, max);
        // A real empty result says so; anything else is DuckDuckGo turning away rapid requests for a while.
        if (results.Count == 0 && !html.Contains("no-results", StringComparison.Ordinal))
            throw new HttpRequestException("DuckDuckGo sent no result page; it limits rapid searches, try again shortly");
        return results;
    }

    public static List<Result> Parse(string html, int max)
    {
        var results = new List<Result>();
        foreach (var block in html.Split("class=\"result ").Skip(1))
        {
            int q = block.IndexOf('"');
            if (q < 0 || block[..q].Contains("result--ad")) continue;  // the rest of the class list: skip ads
            var a = TitleRe().Match(block);
            if (!a.Success) continue;
            var url = RealUrl(WebUtility.HtmlDecode(a.Groups[1].Value));
            if (url == null) continue;
            var snippet = SnippetRe().Match(block);
            results.Add(new Result(Clean(a.Groups[2].Value), url, snippet.Success ? Clean(snippet.Groups[1].Value) : ""));
            if (results.Count == max) break;
        }
        return results;
    }

    /// <summary>The text the model gets back as the tool result.</summary>
    public static string Format(string query, List<Result> results)
    {
        if (results.Count == 0) return $"No results for \"{query}\".";
        var sb = new StringBuilder($"Web results for \"{query}\":\n");
        for (int i = 0; i < results.Count; i++)
            sb.Append($"\n{i + 1}. {results[i].Title}\n   {results[i].Url}\n   {results[i].Snippet}\n");
        return sb.ToString();
    }

    private static string? RealUrl(string href)
    {
        if (href.StartsWith("//", StringComparison.Ordinal)) href = "https:" + href;
        if (!Uri.TryCreate(href, UriKind.Absolute, out var u)) return null;
        if (u.Host.EndsWith("duckduckgo.com", StringComparison.OrdinalIgnoreCase))
        {
            var m = UddgRe().Match(u.Query);  // redirect links: /l/?uddg=<target>
            return m.Success ? Uri.UnescapeDataString(m.Groups[1].Value) : null;
        }
        return u.Scheme is "http" or "https" ? u.ToString() : null;
    }

    private static string Clean(string html) => WebUtility.HtmlDecode(TagRe().Replace(html, "")).Trim();

    [GeneratedRegex("class=\"result__a\" href=\"([^\"]+)\"[^>]*>(.*?)</a>", RegexOptions.Singleline)]
    private static partial Regex TitleRe();

    [GeneratedRegex("class=\"result__snippet\"[^>]*>(.*?)</a>", RegexOptions.Singleline)]
    private static partial Regex SnippetRe();

    [GeneratedRegex("[?&]uddg=([^&]+)")]
    private static partial Regex UddgRe();

    [GeneratedRegex("<[^>]+>")]
    private static partial Regex TagRe();
}
