using Markdig;
using Markdig.Extensions.Tables;
using Markdig.Syntax;
using Markdig.Syntax.Inlines;
using Microsoft.UI.Text;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Documents;
using Microsoft.UI.Xaml.Media;
using Windows.UI.Text;
using MdBlock = Markdig.Syntax.Block;
using MdInline = Markdig.Syntax.Inlines.Inline;
using XamlInline = Microsoft.UI.Xaml.Documents.Inline;

namespace Xdna1Chat;

/// <summary>Markdown to native WinUI elements: paragraphs, headings, lists, quotes, code blocks, tables, links.
/// Re-rendered as text streams in, which is cheap at the engine's ~15 tokens per second.</summary>
public static class MarkdownView
{
    private static readonly MarkdownPipeline Pipeline =
        new MarkdownPipelineBuilder().UsePipeTables().UseEmphasisExtras().UseAutoLinks().Build();

    private static readonly FontFamily Mono = new("Cascadia Mono, Consolas");

    /// <summary>Marks code block borders, whose corners follow the theme.</summary>
    public const string CodeTag = "code";

    public static void SetCodeCorners(UIElement e, CornerRadius r)
    {
        if (e is Border b)
        {
            if (b.Tag is CodeTag) b.CornerRadius = r;
            else if (b.Child != null) SetCodeCorners(b.Child, r);
        }
        else if (e is Panel p)
            foreach (var c in p.Children) SetCodeCorners(c, r);
    }

    /// <summary>Renders into <paramref name="host"/>, keeping the blocks whose source is unchanged since the last
    /// call and updating a growing last paragraph in place; <paramref name="added"/> sees each new block.</summary>
    public static void Render(StackPanel host, string markdown, Action<UIElement>? added = null)
    {
        var blocks = Markdown.Parse(markdown, Pipeline).ToList();
        var keys = blocks.Select(b => b.GetType().Name + ":" + Source(b, markdown)).ToList();
        var old = host.Tag as List<string> ?? [];
        int same = 0;
        while (same < old.Count && same < keys.Count && same < host.Children.Count && old[same] == keys[same]) same++;
        bool inPlace = same < host.Children.Count && same < blocks.Count
            && host.Children[same] is RichTextBlock { Tag: "p" } && blocks[same] is ParagraphBlock;
        while (host.Children.Count > same + (inPlace ? 1 : 0)) host.Children.RemoveAt(host.Children.Count - 1);
        for (int i = same; i < blocks.Count; i++)
        {
            if (i == same && inPlace)
            {
                Fill((RichTextBlock)host.Children[i], ((ParagraphBlock)blocks[i]).Inline);
                continue;
            }
            var e = Block(blocks[i], markdown);
            host.Children.Add(e);
            if (i >= old.Count) added?.Invoke(e);
        }
        host.Tag = keys;
    }

    private static string Source(MdBlock b, string src) =>
        src.Substring(b.Span.Start, Math.Max(0, Math.Min(b.Span.Length, src.Length - b.Span.Start)));

    private static UIElement Block(MdBlock b, string src) => b switch
    {
        HeadingBlock h => Text(h.Inline, Ui.Current.ReadingSize + h.Level switch { 1 => 8, 2 => 5, 3 => 3, _ => 1 }, bold: true),
        ParagraphBlock p => Paragraph(p.Inline),
        CodeBlock c => Code(c.Lines.ToString(), (c as FencedCodeBlock)?.Info),
        ListBlock l => List(l, src),
        QuoteBlock q => Quote(q, src),
        ThematicBreakBlock => new Border { Height = 1, Margin = new Thickness(0, 6, 0, 6), Background = Ui.Stroke },
        Table t => TableView(t),
        LeafBlock leaf when leaf.Inline != null => Text(leaf.Inline),
        _ => Plain(Source(b, src)),
    };

    /// <summary>Body text at the inherited (theme's reading) size; a heading gets its own size.</summary>
    private static RichTextBlock Text(ContainerInline? inlines, double? size = null, bool bold = false)
    {
        var rtb = new RichTextBlock { IsTextSelectionEnabled = true, TextWrapping = TextWrapping.Wrap };
        if (size is { } s) rtb.FontSize = s;
        Fill(rtb, inlines, bold);
        return rtb;
    }

    /// <summary>A paragraph, tagged so a re-render can update it in place.</summary>
    private static RichTextBlock Paragraph(ContainerInline? inlines)
    {
        var rtb = Text(inlines);
        rtb.Tag = "p";
        return rtb;
    }

    private static void Fill(RichTextBlock rtb, ContainerInline? inlines, bool bold = false)
    {
        var p = new Paragraph();
        if (bold) p.FontWeight = FontWeights.SemiBold;
        if (inlines != null) AddInlines(p.Inlines, inlines);
        rtb.Blocks.Clear();
        rtb.Blocks.Add(p);
    }

    private static TextBlock Plain(string text) =>
        new() { Text = text, IsTextSelectionEnabled = true, TextWrapping = TextWrapping.Wrap };

    private static void AddInlines(InlineCollection target, ContainerInline container, bool inLink = false)
    {
        foreach (MdInline i in container)
        {
            switch (i)
            {
                case LiteralInline lit:
                    target.Add(new Run { Text = lit.Content.ToString() });
                    break;
                case CodeInline code:
                    target.Add(new Run { Text = code.Content, FontFamily = Mono });
                    break;
                case LineBreakInline br:
                    target.Add(br.IsHard ? new LineBreak() : new Run { Text = " " });
                    break;
                case EmphasisInline em:
                    Span span = em.DelimiterChar == '~'
                        ? new Span { TextDecorations = TextDecorations.Strikethrough }
                        : em.DelimiterCount >= 2 ? new Bold() : new Italic();
                    AddInlines(span.Inlines, em, inLink);
                    target.Add(span);
                    break;
                case LinkInline link when !link.IsImage && !inLink:
                    target.Add(Link(link.Url, l => AddInlines(l.Inlines, link, inLink: true)));
                    break;
                case AutolinkInline auto when !inLink:
                    target.Add(Link(auto.IsEmail ? "mailto:" + auto.Url : auto.Url, l => l.Inlines.Add(new Run { Text = auto.Url })));
                    break;
                case HtmlEntityInline entity:
                    target.Add(new Run { Text = entity.Transcoded.ToString() });
                    break;
                case HtmlInline html:
                    target.Add(new Run { Text = html.Tag });
                    break;
                case ContainerInline inner:
                    AddInlines(target, inner, inLink);
                    break;
                default:
                    target.Add(new Run { Text = i.ToString() ?? "" });
                    break;
            }
        }
    }

    private static XamlInline Link(string? url, Action<Hyperlink> fill)
    {
        var h = new Hyperlink { Foreground = Ui.AccentText };
        if (Uri.TryCreate(url, UriKind.Absolute, out var u) && u.Scheme is "http" or "https" or "mailto")
            h.NavigateUri = u;
        fill(h);
        if (h.Inlines.Count == 0) h.Inlines.Add(new Run { Text = url ?? "" });
        return h;
    }

    private static UIElement Code(string code, string? language)
    {
        code = code.TrimEnd('\n');
        var text = new TextBlock { Text = code, FontFamily = Mono, FontSize = 13, IsTextSelectionEnabled = true };
        var header = new Grid { Margin = new Thickness(0, 0, 0, 4) };
        header.Children.Add(new TextBlock
        {
            Text = language ?? "",
            FontSize = 12,
            Foreground = Ui.Secondary,
            VerticalAlignment = VerticalAlignment.Center,
        });
        var copy = new Button
        {
            Content = "Copy",
            FontSize = 12,
            Padding = new Thickness(8, 2, 8, 2),
            Background = new SolidColorBrush(Microsoft.UI.Colors.Transparent),
            BorderThickness = new Thickness(0),
            HorizontalAlignment = HorizontalAlignment.Right,
        };
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(copy, "Copy code");
        copy.Click += async (_, _) =>
        {
            var data = new Windows.ApplicationModel.DataTransfer.DataPackage();
            data.SetText(code);
            Windows.ApplicationModel.DataTransfer.Clipboard.SetContent(data);
            copy.Content = "Copied";
            await Task.Delay(1500);
            copy.Content = "Copy";
        };
        header.Children.Add(copy);
        var body = new StackPanel();
        body.Children.Add(header);
        body.Children.Add(new ScrollViewer
        {
            Content = text,
            HorizontalScrollBarVisibility = ScrollBarVisibility.Auto,
            HorizontalScrollMode = ScrollMode.Auto,
            VerticalScrollBarVisibility = ScrollBarVisibility.Disabled,
            VerticalScrollMode = ScrollMode.Disabled,
        });
        return new Border
        {
            Background = Ui.Code,
            BorderBrush = Ui.Stroke,
            BorderThickness = new Thickness(1),
            CornerRadius = new CornerRadius(Ui.Current.Radius),
            Padding = new Thickness(12, 4, 6, 8),
            Child = body,
            Tag = CodeTag,
        };
    }

    private static UIElement List(ListBlock list, string src)
    {
        var panel = new StackPanel { Spacing = 4 };
        int n = int.TryParse(list.OrderedStart, out var start) ? start : 1;
        foreach (var item in list.OfType<ListItemBlock>())
        {
            var row = new Grid();
            row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(26) });
            row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            var marker = new TextBlock { Text = list.IsOrdered ? $"{n++}." : "•", HorizontalAlignment = HorizontalAlignment.Right, Margin = new Thickness(0, 0, 8, 0) };
            var body = new StackPanel { Spacing = 4 };
            foreach (var b in item) body.Children.Add(Block(b, src));
            Grid.SetColumn(body, 1);
            row.Children.Add(marker);
            row.Children.Add(body);
            panel.Children.Add(row);
        }
        return panel;
    }

    private static UIElement Quote(QuoteBlock quote, string src)
    {
        var body = new StackPanel { Spacing = 6 };
        foreach (var b in quote) body.Children.Add(Block(b, src));
        return new Border
        {
            BorderBrush = Ui.Accent,
            BorderThickness = new Thickness(3, 0, 0, 0),
            Padding = new Thickness(12, 2, 0, 2),
            Child = body,
        };
    }

    private static UIElement TableView(Table table)
    {
        var grid = new Grid { ColumnSpacing = 16, RowSpacing = 4 };
        int cols = table.OfType<TableRow>().Select(r => r.Count).DefaultIfEmpty(0).Max();
        for (int c = 0; c < cols; c++) grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        int r = 0;
        foreach (var row in table.OfType<TableRow>())
        {
            grid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            int c = 0;
            foreach (var cell in row.OfType<TableCell>())
            {
                var para = cell.OfType<ParagraphBlock>().FirstOrDefault();
                var text = Text(para?.Inline, bold: row.IsHeader);
                Grid.SetRow(text, r);
                Grid.SetColumn(text, c++);
                grid.Children.Add(text);
            }
            r++;
        }
        return new ScrollViewer
        {
            Content = grid,
            HorizontalScrollBarVisibility = ScrollBarVisibility.Auto,
            HorizontalScrollMode = ScrollMode.Auto,
            VerticalScrollMode = ScrollMode.Disabled,
        };
    }
}
