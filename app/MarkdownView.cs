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

    public static void Render(StackPanel host, string markdown)
    {
        host.Children.Clear();
        foreach (var block in Markdown.Parse(markdown, Pipeline))
            host.Children.Add(Block(block, markdown));
    }

    private static UIElement Block(MdBlock b, string src) => b switch
    {
        HeadingBlock h => Text(h.Inline, h.Level switch { 1 => 22, 2 => 19, 3 => 17, _ => 15 }, bold: true),
        ParagraphBlock p => Text(p.Inline),
        CodeBlock c => Code(c.Lines.ToString(), (c as FencedCodeBlock)?.Info),
        ListBlock l => List(l, src),
        QuoteBlock q => Quote(q, src),
        ThematicBreakBlock => new Border { Height = 1, Margin = new Thickness(0, 6, 0, 6), Background = Brush("DividerStrokeColorDefaultBrush") },
        Table t => TableView(t),
        LeafBlock leaf when leaf.Inline != null => Text(leaf.Inline),
        _ => Plain(src.Substring(b.Span.Start, Math.Max(0, Math.Min(b.Span.Length, src.Length - b.Span.Start)))),
    };

    private static RichTextBlock Text(ContainerInline? inlines, double size = 14, bool bold = false)
    {
        var p = new Paragraph();
        if (bold) p.FontWeight = FontWeights.SemiBold;
        if (inlines != null) AddInlines(p.Inlines, inlines);
        var rtb = new RichTextBlock { IsTextSelectionEnabled = true, TextWrapping = TextWrapping.Wrap, FontSize = size };
        rtb.Blocks.Add(p);
        return rtb;
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
        var h = new Hyperlink();
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
            Foreground = Brush("TextFillColorSecondaryBrush"),
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
            Background = Brush("CardBackgroundFillColorSecondaryBrush"),
            BorderBrush = Brush("CardStrokeColorDefaultBrush"),
            BorderThickness = new Thickness(1),
            CornerRadius = new CornerRadius(6),
            Padding = new Thickness(12, 4, 6, 8),
            Child = body,
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
            BorderBrush = Brush("AccentFillColorDefaultBrush"),
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

    private static Brush Brush(string key) => (Brush)Application.Current.Resources[key];
}
