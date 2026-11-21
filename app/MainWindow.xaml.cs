using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.UI;
using Microsoft.UI.Input;
using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Hosting;
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
    private CancellationTokenSource? downloading;
    private string ready = "";

    private const float TileRest = 0.16f;
    private readonly IntPtr hwnd;
    private readonly List<UserView> users = [];
    private readonly List<AssistantView> replies = [];
    private readonly List<Border> tiles = [];
    private readonly Random random = new();
    private readonly Stopwatch live = new();
    private Hotkey? hotkey;
    private bool compact, applying;
    private int liveTokens;
    private long liveShown;

    public MainWindow()
    {
        InitializeComponent();
        hwnd = WinRT.Interop.WindowNative.GetWindowHandle(this);
        AppWindow.Resize(new SizeInt32(1000, 900));
        ExtendsContentIntoTitleBar = true;
        SetTitleBar(StatusText);
        ThinkToggle.IsChecked = settings.Thinking;
        SearchToggle.IsChecked = settings.WebSearch;
        BuildModelChoices();
        BuildThemeMenu();
        BuildSidebar();
        EmptyHint.Foreground = LicenseText.Foreground = StatusDot.Fill = Ui.Tertiary;
        SetupText.Foreground = DownloadText.Foreground = Ui.Secondary;
        PromptGlyph.Foreground = OrbCore.Fill = Ui.Accent;
        Root.ActualThemeChanged += (_, _) =>
        {
            if (!applying && Ui.Current.Colors == null)
                Ui.SetPalette(Themes.NativePalette(Root.ActualTheme == ElementTheme.Dark));
        };
        ApplyTheme(Themes.Find(settings.Theme), save: false);
        Closed += (_, _) =>
        {
            downloading?.Cancel();
            starting?.Cancel();
            turn?.Cancel();
            hotkey?.Dispose();
            engine.Dispose();
        };
        _ = StartEngineAsync();
    }

    private void BuildThemeMenu()
    {
        foreach (var t in Themes.All)
        {
            var item = new RadioMenuFlyoutItem { Text = t.Name, GroupName = "theme", Tag = t.Id };
            item.Click += (_, _) => ApplyTheme(t);
            ThemeMenu.Items.Add(item);
        }
    }

    private void BuildSidebar()
    {
        Sidebar.Background = Ui.Surface;
        Sidebar.BorderBrush = NpuCard.BorderBrush = Ui.Stroke;
        NpuCard.Background = Ui.Raised;
        Chip.BorderBrush = ChipCore.Background = Ui.Accent;
        NpuLabel.Foreground = SideModel.Foreground = NpuState.Foreground = Ui.Secondary;
        for (int i = 0; i < 16; i++)
        {
            var tile = new Border { Height = 36, CornerRadius = new CornerRadius(5), Background = Ui.Accent };
            ElementCompositionPreview.GetElementVisual(tile).Opacity = TileRest;
            Grid.SetRow(tile, i / 4);
            Grid.SetColumn(tile, i % 4);
            Tiles.Children.Add(tile);
            tiles.Add(tile);
        }
    }

    /// <summary>Switches the look: brushes, WinUI's control colours, fonts, layout and, for Spotlight, the window.</summary>
    private void ApplyTheme(Theme t, bool save = true)
    {
        Ui.Current = t;
        applying = true;
        Root.Resources.MergedDictionaries.Clear();
        if (t.Colors != null) Root.Resources.MergedDictionaries.Add(Ui.Overrides(t));
        // Flipping the requested theme makes every {ThemeResource} look up again, now in the new dictionary.
        bool dark = (t.Base == ElementTheme.Default ? Root.ActualTheme : t.Base) == ElementTheme.Dark;
        Root.RequestedTheme = dark ? ElementTheme.Light : ElementTheme.Dark;
        Root.RequestedTheme = t.Base;
        applying = false;
        Ui.SetPalette(t.Colors ?? Themes.NativePalette(Root.ActualTheme == ElementTheme.Dark));

        SystemBackdrop = t.Backdrop switch
        {
            Backdrop.Mica => new MicaBackdrop(),
            Backdrop.Acrylic => new DesktopAcrylicBackdrop(),
            _ => null,
        };
        Frame.Background = t.Colors != null ? Ui.Background : null;  // a UserControl draws no background itself
        if (t.Colors != null)
        {
            Root.Foreground = Ui.Text;
            Root.FontFamily = new FontFamily(t.Body);
        }
        else
        {
            Root.ClearValue(Control.ForegroundProperty);
            Root.ClearValue(Control.FontFamilyProperty);
        }
        SetCaptionColors(t.Colors);

        static Visibility Show(bool on) => on ? Visibility.Visible : Visibility.Collapsed;
        Sidebar.Visibility = Show(t.Sidebar);
        PromptGlyph.Visibility = Show(t.Prompt);
        Orb.Visibility = Show(t.Orb);
        OrbRing.Stroke = new LinearGradientBrush
        {
            StartPoint = new(0, 0),
            EndPoint = new(1, 1),
            GradientStops =
            {
                new GradientStop { Color = Ui.Accent.Color, Offset = 0 },
                new GradientStop { Color = Ui.AccentSoft.Color, Offset = 1 },
            },
        };
        Grid.SetRow(Composer, t.ComposerTop ? 2 : 4);
        Composer.Padding = t.ComposerTop ? new Thickness(24, 6, 24, 10) : new Thickness(24, 4, 24, 20);
        Messages.MaxWidth = Composer.MaxWidth = t.Column;
        bool rule = t.Sidebar || t.Prompt;  // Silicon and Console draw a line under the title bar
        TitleBar.BorderBrush = Ui.Stroke;
        TitleBar.BorderThickness = new Thickness(0, 0, 0, rule ? 1 : 0);
        TitleBar.Margin = new Thickness(0, 0, 0, rule ? 8 : 0);

        Brand.Visibility = Show(t.Id is "paper" or "console");
        Brand.Background = t.Prompt ? Ui.Accent : null;
        Brand.Padding = t.Prompt ? new Thickness(10, 2, 10, 2) : new Thickness(0, 0, 6, 0);
        BrandText.Foreground = t.Prompt ? Ui.OnAccent : Ui.Text;
        BrandText.FontFamily = new FontFamily(t.Prompt ? t.Body : "Sitka Heading, Georgia");
        BrandText.FontSize = t.Prompt ? 14 : 21;
        BrandText.FontWeight = t.Prompt ? Microsoft.UI.Text.FontWeights.Bold : Microsoft.UI.Text.FontWeights.SemiBold;

        if (t.Underline)
        {
            Input.BorderThickness = new Thickness(0, 0, 0, t.Prompt ? 1 : 1.5);
            Input.CornerRadius = new CornerRadius(0);
        }
        else
        {
            Input.ClearValue(Control.BorderThicknessProperty);
            if (t.Colors == null) Input.ClearValue(Control.CornerRadiusProperty);
            else Input.CornerRadius = new CornerRadius(t.Sidebar ? 14 : t.Radius);
        }
        if (t.Sidebar) Input.Padding = new Thickness(14, 10, 14, 10);
        else Input.ClearValue(Control.PaddingProperty);
        if (t.Compact) Input.FontSize = 18;
        else Input.ClearValue(Control.FontSizeProperty);

        foreach (var u in users) u.Apply(t);
        foreach (var r in replies) r.Apply(t);
        foreach (var item in ThemeMenu.Items.OfType<RadioMenuFlyoutItem>()) item.IsChecked = (string)item.Tag == t.Id;

        if (t.Compact != compact) SetCompact(t.Compact);
        if (save)
        {
            settings.Theme = t.Id;
            settings.Save();
        }
    }

    private void SetCaptionColors(Palette? c)
    {
        var bar = AppWindow.TitleBar;
        bar.ButtonBackgroundColor = bar.ButtonInactiveBackgroundColor = Colors.Transparent;
        bar.ButtonForegroundColor = bar.ButtonHoverForegroundColor = bar.ButtonPressedForegroundColor = c?.Text;
        bar.ButtonHoverBackgroundColor = c?.Hover;
        bar.ButtonPressedBackgroundColor = c?.Stroke;
        bar.ButtonInactiveForegroundColor = c?.Tertiary;
    }

    /// <summary>Spotlight: a small window near the top of the screen, always on top, summoned with Alt+Space.</summary>
    private void SetCompact(bool on)
    {
        compact = on;
        ((OverlappedPresenter)AppWindow.Presenter).IsAlwaysOnTop = on;
        var area = DisplayArea.GetFromWindowId(AppWindow.Id, DisplayAreaFallback.Primary).WorkArea;
        double scale = on ? Hotkey.Scale(hwnd) : 1;  // the normal size is in pixels, like the startup size
        int w = Math.Min((int)((on ? 760 : 1000) * scale), area.Width);
        int h = Math.Min((int)((on ? 620 : 900) * scale), area.Height);
        int y = on ? area.Y + area.Height / 8 : area.Y + (area.Height - h) / 2;
        AppWindow.MoveAndResize(new RectInt32(area.X + (area.Width - w) / 2, y, w, h));
        hotkey?.Dispose();
        hotkey = on ? new Hotkey(hwnd, Hotkey.Alt, Hotkey.Space, OnSummon) : null;
        if (hotkey is { Registered: false })
            SetStatus(Status.Idle, StatusText.Text, "Alt+Space is taken by another app (PowerToys Run uses it), so it can't summon this window.");
    }

    private void OnSummon()
    {
        var presenter = (OverlappedPresenter)AppWindow.Presenter;
        if (presenter.State != OverlappedPresenterState.Minimized && Hotkey.IsForeground(hwnd))
        {
            presenter.Minimize();
            return;
        }
        presenter.Restore();
        Activate();
        Hotkey.Focus(hwnd);
        Input.Focus(FocusState.Programmatic);
    }

    private void OnEscape(KeyboardAccelerator sender, KeyboardAcceleratorInvokedEventArgs e)
    {
        if (!compact) return;
        ((OverlappedPresenter)AppWindow.Presenter).Minimize();
        e.Handled = true;
    }

    /// <summary>Per streamed token: Silicon lights a tile and shows the live speed.</summary>
    private void OnToken()
    {
        if (liveTokens++ == 0) live.Restart();
        if (!Ui.Current.Sidebar) return;
        Motion.Pulse(tiles[random.Next(tiles.Count)], TileRest);
        if (liveTokens > 4 && live.ElapsedMilliseconds - liveShown > 300)
        {
            liveShown = live.ElapsedMilliseconds;
            SideRate.Text = $"{(liveTokens - 1) / live.Elapsed.TotalSeconds:0.0}";
        }
    }

    private void SetBusy(bool busy)
    {
        liveTokens = 0;
        liveShown = 0;
        NpuState.Text = busy ? "live" : "idle";
        NpuState.Foreground = busy ? Ui.AccentText : Ui.Secondary;
        Motion.Spin(OrbRing, busy);
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
            ShowSetup(true);
            if (downloading == null) SetStatus(Status.Idle, "Get a model to start");
            return;
        }
        ShowSetup(false);

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
        SideModel.Text = $"tok/s · {engine.Model}";
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
        var user = new UserView(text);
        users.Add(user);
        Messages.Children.Add(user.Root);
        Motion.Enter(user.Root);
        ScrollToEnd(force: true);

        var cts = turn = new CancellationTokenSource();
        UpdateButtons();
        SetBusy(true);
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
            SetBusy(false);
        }
    }

    /// <summary>Streams replies until the model answers without a tool call; runs its web searches in between.</summary>
    private async Task RunTurnAsync(ChatClient client, CancellationToken ct)
    {
        bool think = ThinkToggle.IsChecked == true, search = SearchToggle.IsChecked == true;
        var view = new AssistantView();
        replies.Add(view);
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
                if (!string.IsNullOrEmpty(c) || !string.IsNullOrEmpty(r))
                {
                    first ??= sw.Elapsed.TotalSeconds;
                    OnToken();
                }
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
            SideRate.Text = $"{rate:0.0}";
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

    private void BuildModelChoices()
    {
        foreach (var m in ModelDownload.Models)
        {
            var row = new Grid { ColumnSpacing = 12 };
            row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            row.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            var text = new StackPanel { Spacing = 2 };
            text.Children.Add(new TextBlock { Text = m.Name, FontWeight = Microsoft.UI.Text.FontWeights.SemiBold });
            text.Children.Add(new TextBlock
            {
                Text = m.Note,
                FontSize = 12,
                Foreground = Ui.Secondary,
            });
            var size = new TextBlock { Text = ModelDownload.Gb(m.Size), VerticalAlignment = VerticalAlignment.Center };
            Grid.SetColumn(size, 1);
            row.Children.Add(text);
            row.Children.Add(size);
            var button = new Button
            {
                Content = row,
                HorizontalAlignment = HorizontalAlignment.Stretch,
                HorizontalContentAlignment = HorizontalAlignment.Stretch,
                Padding = new Thickness(14, 10, 14, 10),
            };
            Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(button, $"Download {m.Name}");
            button.Click += (_, _) => _ = DownloadModelAsync(m);
            ModelChoices.Children.Add(button);
        }
    }

    private void OnModelMenuOpening(object sender, object e)
    {
        ModelMenu.Items.Clear();
        if (downloading != null)
        {
            var cancel = new MenuFlyoutItem { Text = "Cancel download" };
            cancel.Click += OnCancelDownload;
            ModelMenu.Items.Add(cancel);
            ModelMenu.Items.Add(new MenuFlyoutSeparator());
        }
        foreach (var m in ModelDownload.Models)
        {
            bool have = ModelDownload.IsDownloaded(m);
            bool current = string.Equals(settings.Gguf, ModelDownload.LocalPath(m), StringComparison.OrdinalIgnoreCase);
            var item = new MenuFlyoutItem
            {
                Text = current ? $"{m.Name} (in use)" : have ? $"Use {m.Name}" : $"Download {m.Name} ({ModelDownload.Gb(m.Size)})",
                IsEnabled = !current && downloading == null,
            };
            item.Click += (_, _) => _ = DownloadModelAsync(m);
            ModelMenu.Items.Add(item);
        }
        ModelMenu.Items.Add(new MenuFlyoutSeparator());
        var file = new MenuFlyoutItem { Text = "Choose a GGUF file…" };
        file.Click += OnChooseModel;
        ModelMenu.Items.Add(file);
    }

    /// <summary>Downloads the model (or finds it downloaded), then loads it.</summary>
    private async Task DownloadModelAsync(ModelInfo m)
    {
        if (downloading != null) return;
        var cts = downloading = new CancellationTokenSource();
        ModelChoices.Visibility = Visibility.Collapsed;
        DownloadBar.Visibility = CancelDownloadButton.Visibility = DownloadRow.Visibility = Visibility.Visible;
        DownloadBar.Value = 0;
        DownloadText.Text = $"Downloading {m.Name}…";
        var progress = new Progress<DownloadProgress>(p =>
        {
            DownloadBar.Value = (double)p.Done / p.Total;
            var rate = p.BytesPerSecond > 0 ? $" · {p.BytesPerSecond / 1e6:0} MB/s" : "";
            DownloadText.Text = $"{p.Done / 1e6:N0} of {p.Total / 1e6:N0} MB{rate}";
            if (SetupPanel.Visibility != Visibility.Visible)
                SetStatus(Status.Loading, $"Downloading {m.Name}: {p.Done * 100 / p.Total}%{rate}");
        });
        string? path = null;
        try
        {
            path = await ModelDownload.DownloadAsync(m, progress, cts.Token);
        }
        catch (OperationCanceledException)
        {
            DownloadText.Text = "Download cancelled. Starting again resumes it.";
            if (client != null) SetStatus(Status.Ready, ready);
        }
        catch (Exception e) when (e is HttpRequestException or IOException or InvalidDataException or UnauthorizedAccessException)
        {
            DownloadText.Text = "";
            SetStatus(Status.Error, "The download failed", e.Message);
        }
        finally
        {
            downloading = null;
            cts.Dispose();
            ModelChoices.Visibility = Visibility.Visible;
            DownloadBar.Visibility = CancelDownloadButton.Visibility = Visibility.Collapsed;
            if (path != null) DownloadText.Text = "";
            DownloadRow.Visibility = DownloadText.Text.Length > 0 ? Visibility.Visible : Visibility.Collapsed;
        }
        if (path == null) return;
        settings.Gguf = path;
        settings.Save();
        turn?.Cancel();
        await StartEngineAsync();
    }

    private void OnCancelDownload(object sender, RoutedEventArgs e) => downloading?.Cancel();

    private void ShowSetup(bool show)
    {
        SetupPanel.Visibility = show ? Visibility.Visible : Visibility.Collapsed;
        EmptyHint.Visibility = !show && Messages.Children.Count == 0 ? Visibility.Visible : Visibility.Collapsed;
    }

    private void OnNewChat(object sender, RoutedEventArgs e)
    {
        turn?.Cancel();
        history.Clear();
        users.Clear();
        replies.Clear();
        Messages.Children.Clear();
        EmptyHint.Visibility = SetupPanel.Visibility == Visibility.Visible ? Visibility.Collapsed : Visibility.Visible;
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
        if (e.Key == VirtualKey.Escape && compact)
        {
            ((OverlappedPresenter)AppWindow.Presenter).Minimize();
            e.Handled = true;
            return;
        }
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
            _ => Ui.Tertiary,
        };
        Banner.IsOpen = detail != null;
        if (detail != null)
        {
            Banner.Title = text;
            Banner.Message = detail;
        }
    }
}
