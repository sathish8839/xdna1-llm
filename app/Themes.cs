using System.Numerics;
using Microsoft.UI;
using Microsoft.UI.Composition;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Hosting;
using Microsoft.UI.Xaml.Media;
using Windows.UI;
using Windows.UI.ViewManagement;

namespace Xdna1Chat;

public enum Backdrop { Mica, Acrylic, Solid }

/// <summary>How a user message looks: a bubble, a serif heading (Paper) or a prompt line (Console).</summary>
public enum UserStyle { Bubble, Heading, Prompt }

public sealed record Palette(
    Color Background, Color Surface, Color Raised, Color Hover, Color Stroke,
    Color Text, Color Secondary, Color Tertiary,
    Color Accent, Color AccentHover, Color AccentSoft, Color AccentText, Color OnAccent,
    Color UserBubble, Color UserText, Color Code);

/// <summary>A look for the app. <see cref="Colors"/> is null for the native theme, which takes the system's.</summary>
public sealed record Theme(
    string Id, string Name, ElementTheme Base, Backdrop Backdrop, Palette? Colors,
    string Body, string Reading, double ReadingSize, UserStyle User, double Column, double Radius,
    bool Sidebar = false, bool ComposerTop = false, bool Compact = false, bool Prompt = false,
    bool Underline = false, bool Caret = false, bool Orb = false, bool Motion = true);

public static class Themes
{
    private const string Segoe = "Segoe UI Variable Text, Segoe UI";
    private const string Mono = "Cascadia Mono, Consolas";

    public static readonly Theme Native = new("native", "Windows", ElementTheme.Default, Backdrop.Mica, null,
        Segoe, Segoe, 14, UserStyle.Bubble, 860, 6, Motion: false);

    public static readonly Theme[] All =
    [
        Native,
        new("silicon", "Silicon", ElementTheme.Dark, Backdrop.Solid, new(
                Hex("0E1113"), Hex("12161A"), Hex("151A1E"), Hex("1C2328"), Hex("2A3238"),
                Hex("E6E8EA"), Hex("9AA3AB"), Hex("7C858D"),
                Hex("5EE6C8"), Hex("8FEFDA"), Hex("16302B"), Hex("7CEBD3"), Hex("0E1113"),
                Hex("1B2328"), Hex("E6E8EA"), Hex("0A0D0F")),
            Segoe, Segoe, 15, UserStyle.Bubble, 820, 10, Sidebar: true, Caret: true),
        new("paper", "Paper", ElementTheme.Light, Backdrop.Solid, new(
                Hex("FBFBFA"), Hex("F3F4F5"), Hex("FFFFFF"), Hex("EEF0F2"), Hex("D4D8DE"),
                Hex("1B1F24"), Hex("5B6470"), Hex("6B7380"),
                Hex("2F5BD3"), Hex("2449B0"), Hex("E6ECFB"), Hex("2449B0"), Hex("FFFFFF"),
                Hex("00FFFFFF"), Hex("1B1F24"), Hex("F0F1F3")),
            Segoe, "Sitka Text, Georgia", 17, UserStyle.Heading, 680, 6, Underline: true),
        new("console", "Console", ElementTheme.Dark, Backdrop.Solid, new(
                Hex("0B0B0C"), Hex("141310"), Hex("0B0B0C"), Hex("1E1C18"), Hex("3A352C"),
                Hex("E9E4D8"), Hex("A8A397"), Hex("8E8A80"),
                Hex("FFB547"), Hex("FFC870"), Hex("2A2418"), Hex("FFC870"), Hex("0B0B0C"),
                Hex("00000000"), Hex("E9E4D8"), Hex("0B0B0C")),
            Mono, Mono, 14, UserStyle.Prompt, 900, 0, Prompt: true, Underline: true, Caret: true),
        new("spotlight", "Spotlight (Alt+Space)", ElementTheme.Light, Backdrop.Acrylic, new(
                Hex("D9F7F8FB"), Hex("F0F2F7"), Hex("FFFFFF"), Hex("E6EAF2"), Hex("D5DAE3"),
                Hex("14171C"), Hex("5A6271"), Hex("6A7282"),
                Hex("3A5BD9"), Hex("26409F"), Hex("E8EDFC"), Hex("26409F"), Hex("FFFFFF"),
                Hex("E8EDFC"), Hex("14171C"), Hex("EEF1F7")),
            Segoe, Segoe, 15, UserStyle.Bubble, 720, 12, ComposerTop: true, Compact: true, Orb: true),
    ];

    public static Theme Find(string? id) => All.FirstOrDefault(t => t.Id == id) ?? Native;

    /// <summary>The native theme's colours: Fluent's tokens for light or dark, with the system accent.</summary>
    public static Palette NativePalette(bool dark)
    {
        var ui = new UISettings();
        Color A(UIColorType t) => ui.GetColorValue(t);
        Color Alpha(Color c, byte a) => Color.FromArgb(a, c.R, c.G, c.B);
        return dark
            ? new(Colors.Transparent, Hex("08FFFFFF"), Hex("0FFFFFFF"), Hex("15FFFFFF"), Hex("15FFFFFF"),
                Hex("FFFFFF"), Hex("C5FFFFFF"), Hex("87FFFFFF"),
                A(UIColorType.AccentLight2), A(UIColorType.AccentLight1), Alpha(A(UIColorType.AccentLight2), 0x33),
                A(UIColorType.AccentLight3), Hex("000000"),
                Alpha(A(UIColorType.AccentLight2), 0xCC), Hex("000000"), Hex("08FFFFFF"))
            : new(Colors.Transparent, Hex("80F6F6F6"), Hex("B3FFFFFF"), Hex("0A000000"), Hex("0F000000"),
                Hex("E4000000"), Hex("9E000000"), Hex("72000000"),
                A(UIColorType.AccentDark1), A(UIColorType.AccentDark2), Alpha(A(UIColorType.AccentDark1), 0x22),
                A(UIColorType.AccentDark2), Hex("FFFFFF"),
                Alpha(A(UIColorType.AccentDark1), 0xCC), Hex("FFFFFF"), Hex("80F6F6F6"));
    }

    private static Color Hex(string s)
    {
        uint v = Convert.ToUInt32(s, 16);
        return s.Length == 8
            ? Color.FromArgb((byte)(v >> 24), (byte)(v >> 16), (byte)(v >> 8), (byte)v)
            : Color.FromArgb(0xFF, (byte)(v >> 16), (byte)(v >> 8), (byte)v);
    }
}

/// <summary>The current theme's brushes. They are shared and recoloured in place, so a theme switch repaints
/// everything that uses them without rebuilding the chat.</summary>
public static class Ui
{
    public static readonly SolidColorBrush Background = new(), Surface = new(), Raised = new(), Hover = new(),
        Stroke = new(), Text = new(), Secondary = new(), Tertiary = new(), Accent = new(), AccentHover = new(),
        AccentSoft = new(), AccentText = new(), OnAccent = new(), UserBubble = new(), UserText = new(), Code = new();

    public static Theme Current { get; set; } = Themes.Native;

    public static void SetPalette(Palette p)
    {
        Background.Color = p.Background; Surface.Color = p.Surface; Raised.Color = p.Raised; Hover.Color = p.Hover;
        Stroke.Color = p.Stroke; Text.Color = p.Text; Secondary.Color = p.Secondary; Tertiary.Color = p.Tertiary;
        Accent.Color = p.Accent; AccentHover.Color = p.AccentHover; AccentSoft.Color = p.AccentSoft;
        AccentText.Color = p.AccentText; OnAccent.Color = p.OnAccent; UserBubble.Color = p.UserBubble;
        UserText.Color = p.UserText; Code.Color = p.Code;
    }

    /// <summary>WinUI's own control colours (lightweight styling keys) pointed at the theme's brushes.</summary>
    public static ResourceDictionary Overrides(Theme t)
    {
        var d = new ResourceDictionary();
        void Set(Brush b, params string[] keys)
        {
            foreach (var k in keys) d[k] = b;
        }
        Set(Raised, "TextControlBackground", "TextControlBackgroundPointerOver", "TextControlBackgroundFocused",
            "ButtonBackground", "ButtonBackgroundDisabled", "ToggleButtonBackground", "ToggleButtonBackgroundDisabled",
            "ExpanderHeaderBackground", "ExpanderHeaderBackgroundPointerOver", "ExpanderHeaderBackgroundPressed");
        Set(Surface, "ExpanderContentBackground");
        Set(Hover, "ButtonBackgroundPointerOver", "ToggleButtonBackgroundPointerOver");
        Set(Stroke, "TextControlBorderBrush", "TextControlBorderBrushPointerOver", "ButtonBackgroundPressed",
            "ToggleButtonBackgroundPressed", "ButtonBorderBrush", "ButtonBorderBrushPointerOver",
            "ButtonBorderBrushPressed", "ButtonBorderBrushDisabled", "ToggleButtonBorderBrush",
            "ToggleButtonBorderBrushPointerOver", "ToggleButtonBorderBrushPressed", "AccentButtonBackgroundDisabled",
            "ExpanderHeaderBorderBrush", "ExpanderContentBorderBrush");
        Set(Text, "TextControlForeground", "TextControlForegroundPointerOver", "TextControlForegroundFocused",
            "ButtonForeground", "ButtonForegroundPointerOver", "ButtonForegroundPressed", "ToggleButtonForeground",
            "ToggleButtonForegroundPointerOver", "ToggleButtonForegroundPressed", "ExpanderHeaderForeground",
            "ExpanderHeaderForegroundPointerOver", "ExpanderHeaderForegroundPressed", "TextFillColorPrimaryBrush");
        Set(Secondary, "TextFillColorSecondaryBrush");
        Set(Tertiary, "TextControlPlaceholderForeground", "TextControlPlaceholderForegroundPointerOver",
            "TextControlPlaceholderForegroundFocused", "ButtonForegroundDisabled", "ToggleButtonForegroundDisabled",
            "AccentButtonForegroundDisabled", "TextFillColorTertiaryBrush");
        Set(Accent, "TextControlBorderBrushFocused", "TextControlSelectionHighlightColor", "AccentButtonBackground",
            "AccentButtonBorderBrush", "ToggleButtonBorderBrushChecked", "ToggleButtonBorderBrushCheckedPointerOver",
            "ToggleButtonBorderBrushCheckedPressed", "AccentFillColorDefaultBrush");
        Set(AccentHover, "AccentButtonBackgroundPointerOver", "AccentButtonBackgroundPressed",
            "AccentButtonBorderBrushPointerOver", "AccentButtonBorderBrushPressed", "HyperlinkButtonForegroundPointerOver");
        Set(AccentSoft, "ToggleButtonBackgroundChecked", "ToggleButtonBackgroundCheckedPointerOver",
            "ToggleButtonBackgroundCheckedPressed");
        Set(AccentText, "ToggleButtonForegroundChecked", "ToggleButtonForegroundCheckedPointerOver",
            "ToggleButtonForegroundCheckedPressed", "AccentTextFillColorPrimaryBrush", "HyperlinkButtonForeground",
            "SystemControlHyperlinkTextBrush");
        Set(OnAccent, "AccentButtonForeground", "AccentButtonForegroundPointerOver", "AccentButtonForegroundPressed");
        d["ControlCornerRadius"] = new CornerRadius(t.Radius);
        d["OverlayCornerRadius"] = new CornerRadius(t.Radius);
        return d;
    }
}

/// <summary>Composition animations: they run on the compositor thread and cost the UI thread nothing. Off for the
/// native theme and when Windows' "Animation effects" setting is off.</summary>
public static class Motion
{
    private static readonly UISettings System = new();

    private static bool Allowed => System.AnimationsEnabled;
    public static bool On => Ui.Current.Motion && Allowed;

    /// <summary>A new element rises and fades in (Paper: a slower fade, like ink settling).</summary>
    public static void Enter(UIElement e)
    {
        if (!On) return;
        bool paper = Ui.Current.Id == "paper";
        var v = ElementCompositionPreview.GetElementVisual(e);
        var c = v.Compositor;
        var ease = c.CreateCubicBezierEasingFunction(new Vector2(0.2f, 0.8f), new Vector2(0.2f, 1f));
        var duration = TimeSpan.FromMilliseconds(paper ? 520 : 320);
        var fade = c.CreateScalarKeyFrameAnimation();
        fade.InsertKeyFrame(0, 0);
        fade.InsertKeyFrame(1, 1, ease);
        fade.Duration = duration;
        ElementCompositionPreview.SetIsTranslationEnabled(e, true);
        var rise = c.CreateVector3KeyFrameAnimation();
        rise.InsertKeyFrame(0, new Vector3(0, paper ? 6 : 10, 0));
        rise.InsertKeyFrame(1, Vector3.Zero, ease);
        rise.Duration = duration;
        v.StartAnimation("Opacity", fade);
        v.StartAnimation("Translation", rise);
    }

    /// <summary>One NPU tile lights up and fades back (Silicon, once per streamed token).</summary>
    public static void Pulse(UIElement tile, float rest)
    {
        var v = ElementCompositionPreview.GetElementVisual(tile);
        if (!Allowed)
        {
            v.Opacity = rest;
            return;
        }
        var c = v.Compositor;
        var a = c.CreateScalarKeyFrameAnimation();
        a.InsertKeyFrame(0, 1);
        a.InsertKeyFrame(1, rest, c.CreateCubicBezierEasingFunction(new Vector2(0.3f, 0f), new Vector2(0.2f, 1f)));
        a.Duration = TimeSpan.FromMilliseconds(700);
        v.StartAnimation("Opacity", a);
    }

    /// <summary>Spotlight's ring turns while the model works.</summary>
    public static void Spin(FrameworkElement e, bool on)
    {
        var v = ElementCompositionPreview.GetElementVisual(e);
        v.CenterPoint = new Vector3((float)e.Width / 2, (float)e.Height / 2, 0);
        if (!on || !Allowed)
        {
            v.StopAnimation("RotationAngleInDegrees");
            v.RotationAngleInDegrees = 0;
            return;
        }
        var c = v.Compositor;
        var a = c.CreateScalarKeyFrameAnimation();
        a.InsertKeyFrame(0, 0);
        a.InsertKeyFrame(1, 360, c.CreateLinearEasingFunction());
        a.Duration = TimeSpan.FromMilliseconds(1400);
        a.IterationBehavior = AnimationIterationBehavior.Forever;
        v.StartAnimation("RotationAngleInDegrees", a);
    }

    /// <summary>A terminal caret: solid while text arrives (each re-render restarts it), blinking when it pauses.</summary>
    public static void Blink(UIElement e)
    {
        if (!Allowed) return;
        var v = ElementCompositionPreview.GetElementVisual(e);
        var c = v.Compositor;
        var a = c.CreateScalarKeyFrameAnimation();
        a.InsertKeyFrame(0, 1);
        a.InsertKeyFrame(0.5f, 1);
        a.InsertKeyFrame(0.51f, 0);
        a.InsertKeyFrame(1, 0);
        a.Duration = TimeSpan.FromMilliseconds(1000);
        a.IterationBehavior = AnimationIterationBehavior.Forever;
        v.StartAnimation("Opacity", a);
    }
}
