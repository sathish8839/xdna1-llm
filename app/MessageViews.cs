using System.Diagnostics;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Documents;
using Microsoft.UI.Xaml.Media;

namespace Xdna1Chat;

public static class MessageViews
{
    public static UIElement User(string text) => new Border
    {
        HorizontalAlignment = HorizontalAlignment.Right,
        MaxWidth = 620,
        Margin = new Thickness(80, 4, 0, 4),
        Padding = new Thickness(14, 9, 14, 9),
        CornerRadius = new CornerRadius(12),
        Background = Brush("AccentFillColorTertiaryBrush"),
        Child = new TextBlock
        {
            Text = text,
            TextWrapping = TextWrapping.Wrap,
            IsTextSelectionEnabled = true,
            Foreground = Brush("TextOnAccentFillColorPrimaryBrush"),
        },
    };

    internal static Brush Brush(string key) => (Brush)Application.Current.Resources[key];
}

/// <summary>One assistant reply. A reply with web searches takes several model rounds; each round adds an
/// optional collapsed "Thinking" section, its searches and its text. One line of speed stats closes the reply.</summary>
public sealed class AssistantView
{
    public StackPanel Root { get; } = new() { Spacing = 8, Margin = new Thickness(0, 4, 40, 4) };

    private readonly TextBlock stats = new()
    {
        Style = (Style)Application.Current.Resources["CaptionTextBlockStyle"],
        Foreground = MessageViews.Brush("TextFillColorTertiaryBrush"),
        Visibility = Visibility.Collapsed,
    };
    private readonly Stopwatch lastRender = Stopwatch.StartNew();
    private Stopwatch roundTime = new();
    private Expander? thinking;
    private TextBlock? thinkingHeader, thinkingText;
    private StackPanel? content;
    private string shown = "";

    public AssistantView() => Root.Children.Add(stats);

    public void BeginRound()
    {
        thinking = null;
        content = null;
        shown = "";
        roundTime = Stopwatch.StartNew();
    }

    public void SetReasoning(string text, bool done)
    {
        if (text.Length == 0) return;
        if (thinking == null)
        {
            thinkingHeader = new TextBlock();
            thinkingText = new TextBlock
            {
                TextWrapping = TextWrapping.Wrap,
                IsTextSelectionEnabled = true,
                Foreground = MessageViews.Brush("TextFillColorSecondaryBrush"),
            };
            thinking = new Expander
            {
                Header = thinkingHeader,
                Content = thinkingText,
                HorizontalAlignment = HorizontalAlignment.Stretch,
                HorizontalContentAlignment = HorizontalAlignment.Left,
            };
            Add(thinking);
        }
        thinkingText!.Text = text.Trim();
        thinkingHeader!.Text = done ? $"Thought for {roundTime.Elapsed.TotalSeconds:0} s" : "Thinking…";
    }

    /// <summary>Re-renders the markdown at most every 60 ms while streaming, and always when final.</summary>
    public void SetContent(string markdown, bool final)
    {
        if (markdown == shown || (!final && lastRender.ElapsedMilliseconds < 60)) return;
        if (content == null)
        {
            content = new StackPanel { Spacing = 10 };
            Add(content);
        }
        MarkdownView.Render(content, markdown);
        shown = markdown;
        lastRender.Restart();
    }

    public void SetStats(string text)
    {
        stats.Text = text;
        stats.Visibility = Visibility.Visible;
    }

    public SearchNote AddSearch(string query)
    {
        var note = new SearchNote(query);
        Add(note.Root);
        return note;
    }

    private void Add(UIElement e) => Root.Children.Insert(Root.Children.Count - 1, e);  // stats stay last
}

/// <summary>"Searching the web for …", then the result links.</summary>
public sealed class SearchNote
{
    public RichTextBlock Root { get; }
    private readonly Paragraph head = new();
    private readonly string query;

    public SearchNote(string query)
    {
        this.query = query;
        head.Inlines.Add(new Run { Text = $"Searching the web for “{query}”…" });
        Root = new RichTextBlock
        {
            IsTextSelectionEnabled = true,
            TextWrapping = TextWrapping.Wrap,
            FontSize = 13,
            Foreground = MessageViews.Brush("TextFillColorSecondaryBrush"),
        };
        Root.Blocks.Add(head);
    }

    public void SetResults(List<WebSearch.Result> results)
    {
        head.Inlines.Clear();
        head.Inlines.Add(new Run { Text = $"Searched the web for “{query}”" });
        foreach (var r in results)
        {
            var link = new Hyperlink { NavigateUri = new Uri(r.Url) };
            link.Inlines.Add(new Run { Text = r.Title.Length > 0 ? r.Title : r.Url });
            head.Inlines.Add(new LineBreak());
            head.Inlines.Add(new Run { Text = "   " });
            head.Inlines.Add(link);
        }
        if (results.Count == 0) head.Inlines.Add(new Run { Text = ": no results" });
    }

    public void SetFailed(string message)
    {
        head.Inlines.Clear();
        head.Inlines.Add(new Run { Text = $"Web search for “{query}” failed: {message}" });
    }
}
