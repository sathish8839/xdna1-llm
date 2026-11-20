using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.UI;
using Microsoft.UI.Input;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Windows.Graphics;
using Windows.Storage.Pickers;
using Windows.System;
using Windows.UI.Core;

namespace Xdna1Chat;

public sealed partial class MainWindow : Window
{
    private const int MaxSearchRounds = 3;  // after this many tool rounds the model has to answer

    private readonly Settings settings = Settings.Load();
    private readonly EngineHost engine = new();
    private readonly List<JsonObject> history = [];
    private ChatClient? client;
    private CancellationTokenSource? starting;
    private CancellationTokenSource? turn;
    private string ready = "";

    public MainWindow()
    {
        InitializeComponent();
        AppWindow.Resize(new SizeInt32(1000, 900));
        ExtendsContentIntoTitleBar = true;
        SetTitleBar(StatusText);
        ThinkToggle.IsChecked = settings.Thinking;
        SearchToggle.IsChecked = settings.WebSearch;
        Closed += (_, _) =>
        {
            starting?.Cancel();
            turn?.Cancel();
            engine.Dispose();
        };
        _ = StartEngineAsync();
    }

    private async Task StartEngineAsync()
    {
        starting?.Cancel();
        var cts = starting = new CancellationTokenSource();
        client = null;
        UpdateButtons();

        var python = EngineHost.FindPython(settings.Python);
        if (python == null)
        {
            SetStatus(Status.Error, "Engine not found",
                "Set XDNA1_LLM_PYTHON to the engine's python.exe (engine\\.venv\\Scripts\\python.exe in a source checkout).");
            return;
        }
        if (string.IsNullOrEmpty(settings.Gguf) || !File.Exists(settings.Gguf))
        {
            SetStatus(Status.Idle, "Choose a model file to start: LFM3-1.7B Q4_K_M (faster) or Q6_K (closer to full quality)");
            return;
        }

        SetStatus(Status.Loading, $"Loading {Path.GetFileNameWithoutExtension(settings.Gguf)} onto the NPU… (about a minute)");
        var sw = Stopwatch.StartNew();
        try
        {
            await engine.StartAsync(python, settings.Gguf, cts.Token);
        }
        catch (OperationCanceledException)
        {
            return;
        }
        catch (Exception e)
        {
            if (!cts.IsCancellationRequested)
                SetStatus(Status.Error, "The engine did not start", $"{e.Message}\n\nLog: {EngineHost.LogPath}");
            return;
        }
        client = new ChatClient(engine.BaseUrl);
        ready = $"{engine.Model} on {engine.Device}";
        SetStatus(Status.Ready, $"{ready} · loaded in {sw.Elapsed.TotalSeconds:0} s");
        UpdateButtons();
        Input.Focus(FocusState.Programmatic);
    }

    private async void OnSend(object sender, RoutedEventArgs e)
    {
        if (turn != null)
        {
            turn.Cancel();  // the button reads "Stop" while a reply streams
            return;
        }
        var text = Input.Text.Trim();
        if (text.Length == 0 || client == null) return;
        Input.Text = "";
        EmptyHint.Visibility = Visibility.Collapsed;
        history.Add(new JsonObject { ["role"] = "user", ["content"] = text });
        Messages.Children.Add(MessageViews.User(text));
        ScrollToEnd(force: true);

        var cts = turn = new CancellationTokenSource();
        UpdateButtons();
        try
        {
            await RunTurnAsync(client, cts.Token);
        }
        catch (Exception ex) when (cts.IsCancellationRequested && ex is OperationCanceledException or IOException or HttpRequestException)
        {
        }
        catch (Exception ex)
        {
            SetStatus(Status.Error, "The request failed", ex.Message);
        }
        finally
        {
            turn = null;
            cts.Dispose();
            UpdateButtons();
        }
    }

    /// <summary>Streams replies until the model answers without a tool call; runs its web searches in between.</summary>
    private async Task RunTurnAsync(ChatClient client, CancellationToken ct)
    {
        bool think = ThinkToggle.IsChecked == true, search = SearchToggle.IsChecked == true;
        var view = new AssistantView();
        Messages.Children.Add(view.Root);
        var speed = new Speed();
        try
        {
            for (int round = 0; ; round++)
                if (!await RunRoundAsync(client, view, think, search && round < MaxSearchRounds, speed, ct))
                    return;
        }
        finally
        {
            ShowStats(view, speed, ct.IsCancellationRequested);
        }
    }

    /// <summary>Totals over the rounds of one reply: tokens, and decode time after each round's first token.</summary>
    private sealed class Speed
    {
        public int Tokens, Timed;
        public double Seconds;
        public double? FirstToken;
    }

    /// <summary>One model call; true if it asked for tools (which have run and been answered).</summary>
    private async Task<bool> RunRoundAsync(
        ChatClient client, AssistantView view, bool think, bool tools, Speed speed, CancellationToken ct)
    {
        view.BeginRound();
        var body = new JsonObject
        {
            ["model"] = engine.Model,
            ["messages"] = new JsonArray(history.Select(m => (JsonNode)m.DeepClone()).ToArray()),
            ["stream"] = true,
            ["stream_options"] = new JsonObject { ["include_usage"] = true },
            ["chat_template_kwargs"] = new JsonObject { ["enable_thinking"] = think },
        };
        if (tools) body["tools"] = WebSearch.Tools();

        var content = new StringBuilder();
        var reasoning = new StringBuilder();
        var calls = new SortedDictionary<int, ToolCall>();
        JsonNode? usage = null;
        double? first = null;
        var sw = Stopwatch.StartNew();
        try
        {
            await foreach (var chunk in client.StreamAsync(body, ct))
            {
                if (chunk["usage"] is JsonObject u) usage = u;
                if (chunk["choices"] is not JsonArray { Count: > 0 } choices || choices[0]?["delta"] is not JsonObject d)
                    continue;
                var c = (string?)d["content"];
                var r = (string?)d["reasoning_content"];
                if (!string.IsNullOrEmpty(c) || !string.IsNullOrEmpty(r)) first ??= sw.Elapsed.TotalSeconds;
                if (!string.IsNullOrEmpty(r)) view.SetReasoning(reasoning.Append(r).ToString(), done: false);
                if (!string.IsNullOrEmpty(c)) view.SetContent(content.Append(c).ToString(), final: false);
                if (d["tool_calls"] is JsonArray deltas) Accumulate(calls, deltas);
                ScrollToEnd();
            }
        }
        finally
        {
            view.SetReasoning(reasoning.ToString(), done: true);
            view.SetContent(content.ToString(), final: true);
            int n = (int?)usage?["completion_tokens"] ?? 0;
            speed.Tokens += n;
            speed.FirstToken ??= first;
            if (first is { } f && n > 1)
            {
                speed.Timed += n - 1;
                speed.Seconds += sw.Elapsed.TotalSeconds - f;
            }
            if (content.Length > 0 || calls.Count > 0)  // keep a stopped reply too, so the chat reads on
                history.Add(AssistantMessage(content.ToString(), calls.Values));
        }
        if (calls.Count == 0) return false;

        foreach (var call in calls.Values)
            history.Add(new JsonObject
            {
                ["role"] = "tool",
                ["tool_call_id"] = call.Id,
                ["content"] = await RunToolAsync(view, call, ct),
            });
        ScrollToEnd();
        return true;
    }

    private static async Task<string> RunToolAsync(AssistantView view, ToolCall call, CancellationToken ct)
    {
        if (call.Name != "web_search") return $"Unknown tool: {call.Name}";
        string query = "";
        try { query = (string?)JsonNode.Parse(call.Args.ToString())?["query"] ?? ""; }
        catch (JsonException) { }
        if (query.Length == 0) return "The web_search tool needs a query.";
        var note = view.AddSearch(query);
        try
        {
            var results = await WebSearch.SearchAsync(query, ct);
            note.SetResults(results);
            return WebSearch.Format(query, results);
        }
        catch (Exception e) when (!ct.IsCancellationRequested && e is HttpRequestException or TaskCanceledException)
        {
            note.SetFailed(e.Message);
            return $"The web search failed ({e.Message}). Answer from what you know, and say so.";
        }
    }

    private sealed record ToolCall(string Id, string Name, StringBuilder Args);

    private static void Accumulate(SortedDictionary<int, ToolCall> calls, JsonArray deltas)
    {
        foreach (var tc in deltas.OfType<JsonObject>())
        {
            int i = (int?)tc["index"] ?? calls.Count;
            if (!calls.TryGetValue(i, out var call))
                calls[i] = call = new ToolCall((string?)tc["id"] ?? $"call_{i}", "", new StringBuilder());
            if (tc["function"] is JsonObject fn)
            {
                if ((string?)fn["name"] is { Length: > 0 } name) calls[i] = call = call with { Name = call.Name + name };
                call.Args.Append((string?)fn["arguments"]);
            }
        }
    }

    private static JsonObject AssistantMessage(string content, IEnumerable<ToolCall> calls)
    {
        var msg = new JsonObject { ["role"] = "assistant", ["content"] = content };
        var list = calls.Select(c => (JsonNode)new JsonObject
        {
            ["id"] = c.Id,
            ["type"] = "function",
            ["function"] = new JsonObject { ["name"] = c.Name, ["arguments"] = c.Args.ToString() },
        }).ToArray();
        if (list.Length > 0) msg["tool_calls"] = new JsonArray(list);
        return msg;
    }

    private void ShowStats(AssistantView view, Speed speed, bool stopped)
    {
        var parts = new List<string>();
        if (speed.Tokens > 0) parts.Add($"{speed.Tokens} tokens");
        if (speed.Seconds > 0)
        {
            double rate = speed.Timed / speed.Seconds;
            parts.Add($"{rate:0.0} tok/s");
            parts.Add($"first token {speed.FirstToken:0.0} s");
            SetStatus(Status.Ready, $"{ready} · last reply {rate:0.0} tok/s");
        }
        if (stopped) parts.Add("stopped");
        if (parts.Count > 0) view.SetStats(string.Join(" · ", parts));
    }

    private async void OnChooseModel(object sender, RoutedEventArgs e)
    {
        var picker = new FileOpenPicker { SuggestedStartLocation = PickerLocationId.Downloads };
        picker.FileTypeFilter.Add(".gguf");
        WinRT.Interop.InitializeWithWindow.Initialize(picker, WinRT.Interop.WindowNative.GetWindowHandle(this));
        var file = await picker.PickSingleFileAsync();
        if (file == null) return;
        settings.Gguf = file.Path;
        settings.Save();
        turn?.Cancel();
        await StartEngineAsync();
    }

    private void OnNewChat(object sender, RoutedEventArgs e)
    {
        turn?.Cancel();
        history.Clear();
        Messages.Children.Clear();
        EmptyHint.Visibility = Visibility.Visible;
        Input.Focus(FocusState.Programmatic);
    }

    private void OnToggle(object sender, RoutedEventArgs e)
    {
        settings.Thinking = ThinkToggle.IsChecked == true;
        settings.WebSearch = SearchToggle.IsChecked == true;
        settings.Save();
    }

    private void OnInputKeyDown(object sender, KeyRoutedEventArgs e)
    {
        if (e.Key != VirtualKey.Enter) return;
        if (InputKeyboardSource.GetKeyStateForCurrentThread(VirtualKey.Shift).HasFlag(CoreVirtualKeyStates.Down)) return;
        e.Handled = true;
        if (turn == null) OnSend(sender, e);
    }

    private void UpdateButtons()
    {
        SendButton.Content = turn != null ? "Stop" : "Send";
        SendButton.IsEnabled = turn != null || client != null;
    }

    private void ScrollToEnd(bool force = false)
    {
        bool atEnd = Scroller.VerticalOffset >= Scroller.ScrollableHeight - 64;
        if (!force && !atEnd) return;
        Scroller.UpdateLayout();
        Scroller.ChangeView(null, Scroller.ScrollableHeight, null, disableAnimation: true);
    }

    private enum Status { Idle, Loading, Ready, Error }

    private void SetStatus(Status status, string text, string? detail = null)
    {
        StatusText.Text = text;
        StatusDot.Fill = status switch
        {
            Status.Ready => new SolidColorBrush(Colors.MediumSeaGreen),
            Status.Loading => new SolidColorBrush(Colors.Goldenrod),
            Status.Error => new SolidColorBrush(Colors.IndianRed),
            _ => (Brush)Application.Current.Resources["TextFillColorDisabledBrush"],
        };
        Banner.IsOpen = detail != null;
        if (detail != null)
        {
            Banner.Title = text;
            Banner.Message = detail;
        }
    }
}
