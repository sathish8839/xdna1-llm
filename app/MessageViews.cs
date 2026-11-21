using System.Diagnostics;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Documents;
using Microsoft.UI.Xaml.Media;

namespace Xdna1Chat;

/// <summary>A user message: a bubble, a serif heading (Paper) or a "›" prompt line (Console).</summary>
public sealed class UserView
{
    public Border Root { get; } = new();
    private readonly TextBlock prefix = new() { Text = "›", Margin = new Thickness(0, 0, 10, 0), Foreground = Ui.Accent };
    private readonly TextBlock text;

    public UserView(string message)
    {
        text = new TextBlock { Text = message, TextWrapping = TextWrapping.Wrap, IsTextSelectionEnabled = true };
        var row = new Grid();
        row.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        Grid.SetColumn(text, 1);
        row.Children.Add(prefix);
        row.Children.Add(text);
        Root.Child = row;
        Apply(Ui.Current);
    }

    public void Apply(Theme t)
    {
        bool bubble = t.User == UserStyle.Bubble;
        Root.HorizontalAlignment = bubble ? HorizontalAlignment.Right : HorizontalAlignment.Stretch;
        Root.MaxWidth = bubble ? 620 : double.PositiveInfinity;
        Root.Margin = t.User switch
        {
            UserStyle.Heading => new Thickness(0, 22, 0, 2),
            UserStyle.Prompt => new Thickness(0, 12, 0, 0),
            _ => new Thickness(80, 4, 0, 4),
        };
        Root.Padding = bubble ? new Thickness(14, 9, 14, 9) : new Thickness(0);
        Root.CornerRadius = new CornerRadius(Math.Min(2 * t.Radius, 18));
        Root.Background = bubble ? Ui.UserBubble : null;
        text.Foreground = bubble ? Ui.UserText : Ui.Text;
        prefix.Visibility = t.User == UserStyle.Prompt ? Visibility.Visible : Visibility.Collapsed;
        if (t.User == UserStyle.Heading)
        {
            text.FontFamily = new FontFamily(t.Reading);
            text.FontSize = 24;
        }
        else
        {
            text.ClearValue(TextBlock.FontFamilyProperty);
            text.ClearValue(TextBlock.FontSizeProperty);
        }
    }
}

/// <summary>One assistant reply. A reply with web searches takes several model rounds; each round adds an
/// optional collapsed "Thinking" section, its searches and its text. One line of speed stats closes the reply.</summary>
public sealed class AssistantView
{
    /// <summary>A Control, so the theme's reading font and size inherit to the rendered markdown.</summary>
    public UserControl Root { get; } = new() { Margin = new Thickness(0, 4, 40, 4) };

    private readonly StackPanel panel = new() { Spacing = 8 };
    private readonly TextBlock stats = new()
    {
        Style = (Style)Application.Current.Resources["CaptionTextBlockStyle"],
        Foreground = Ui.Tertiary,
        Visibility = Visibility.Collapsed,
    };
    private readonly Stopwatch lastRender = Stopwatch.StartNew();
    private Stopwatch roundTime = new();
    private Expander? thinking;
    private TextBlock? thinkingHeader, thinkingText;
    private StackPanel? content;
    private string shown = "";
    private InlineUIContainer? caret;
    private Paragraph? caretHost;

    public AssistantView()
    {
        panel.Children.Add(stats);
        Root.Content = panel;
        Apply(Ui.Current);
    }

    public void Apply(Theme t)
    {
        Root.FontFamily = new FontFamily(t.Reading);
        Root.FontSize = t.ReadingSize;
        stats.FontFamily = new FontFamily(t.Body);
        MarkdownView.SetCodeCorners(panel, new CornerRadius(t.Radius));
    }

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
                Foreground = Ui.Secondary,
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

    /// <summary>Re-renders the markdown at most every 60 ms while streaming, and always when final. New blocks
    /// animate in; Silicon and Console show a caret while text arrives.</summary>
    public void SetContent(string markdown, bool final)
    {
        if (markdown == shown)
        {
            if (final) RemoveCaret();
            return;
        }
        if (!final && lastRender.ElapsedMilliseconds < 60) return;
        if (content == null)
        {
            content = new StackPanel { Spacing = 10 };
            Add(content);
        }
        RemoveCaret();
        MarkdownView.Render(content, markdown, Motion.Enter);
        if (!final) AddCaret();
        shown = markdown;
        lastRender.Restart();
    }

    private void AddCaret()
    {
        if (!Ui.Current.Caret || content?.Children.LastOrDefault() is not RichTextBlock rtb
            || rtb.Blocks.LastOrDefault() is not Paragraph p)
            return;
        var block = new Border { Width = 8, Height = 16, Margin = new Thickness(3, 0, 0, -3), Background = Ui.Accent };
        caret = new InlineUIContainer { Child = block };
        p.Inlines.Add(caret);
        caretHost = p;
        Motion.Blink(block);
    }

    private void RemoveCaret()
    {
        if (caret != null) caretHost?.Inlines.Remove(caret);
        caret = null;
        caretHost = null;
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

    private void Add(UIElement e)
    {
        panel.Children.Insert(panel.Children.Count - 1, e);  // stats stay last
        Motion.Enter(e);
    }
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
            Foreground = Ui.Secondary,
        };
        Root.Blocks.Add(head);
    }

    public void SetResults(List<WebSearch.Result> results)
    {
        head.Inlines.Clear();
        head.Inlines.Add(new Run { Text = $"Searched the web for “{query}”" });
        foreach (var r in results)
        {
            var link = new Hyperlink { NavigateUri = new Uri(r.Url), Foreground = Ui.AccentText };
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
