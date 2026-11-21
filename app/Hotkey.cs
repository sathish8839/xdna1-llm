using System.Runtime.InteropServices;

namespace Xdna1Chat;

/// <summary>A system-wide hotkey delivered to one window (Spotlight's Alt+Space), through a window subclass.</summary>
internal sealed class Hotkey : IDisposable
{
    public const uint Alt = 0x1, Space = 0x20;
    private const uint NoRepeat = 0x4000, WmHotkey = 0x0312;
    private const int Id = 1;

    private readonly IntPtr hwnd;
    private readonly SubclassProc proc;  // kept alive: native code holds a pointer to it
    private readonly Action pressed;

    /// <summary>False if another app already holds the key (PowerToys Run uses Alt+Space too).</summary>
    public bool Registered { get; }

    public Hotkey(IntPtr hwnd, uint modifiers, uint key, Action pressed)
    {
        this.hwnd = hwnd;
        this.pressed = pressed;
        proc = WndProc;
        SetWindowSubclass(hwnd, proc, Id, 0);
        Registered = RegisterHotKey(hwnd, Id, modifiers | NoRepeat, key);
    }

    public static bool IsForeground(IntPtr hwnd) => GetForegroundWindow() == hwnd;
    public static void Focus(IntPtr hwnd) => SetForegroundWindow(hwnd);
    public static double Scale(IntPtr hwnd) => GetDpiForWindow(hwnd) / 96.0;

    private IntPtr WndProc(IntPtr h, uint msg, IntPtr w, IntPtr l, nuint id, nuint data)
    {
        if (msg == WmHotkey && w == Id)
        {
            pressed();
            return 0;
        }
        return DefSubclassProc(h, msg, w, l);
    }

    public void Dispose()
    {
        if (Registered) UnregisterHotKey(hwnd, Id);
        RemoveWindowSubclass(hwnd, proc, Id);
    }

    private delegate IntPtr SubclassProc(IntPtr hwnd, uint msg, IntPtr w, IntPtr l, nuint id, nuint data);

    [DllImport("comctl32.dll")]
    private static extern bool SetWindowSubclass(IntPtr hwnd, SubclassProc proc, nuint id, nuint data);

    [DllImport("comctl32.dll")]
    private static extern bool RemoveWindowSubclass(IntPtr hwnd, SubclassProc proc, nuint id);

    [DllImport("comctl32.dll")]
    private static extern IntPtr DefSubclassProc(IntPtr hwnd, uint msg, IntPtr w, IntPtr l);

    [DllImport("user32.dll")]
    private static extern bool RegisterHotKey(IntPtr hwnd, int id, uint modifiers, uint key);

    [DllImport("user32.dll")]
    private static extern bool UnregisterHotKey(IntPtr hwnd, int id);

    [DllImport("user32.dll")]
    private static extern IntPtr GetForegroundWindow();

    [DllImport("user32.dll")]
    private static extern bool SetForegroundWindow(IntPtr hwnd);

    [DllImport("user32.dll")]
    private static extern uint GetDpiForWindow(IntPtr hwnd);
}
