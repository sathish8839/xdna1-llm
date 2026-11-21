using System.Text.Json;

namespace Xdna1Chat;

/// <summary>%LOCALAPPDATA%\xdna1-llm\app.json: the model file and, optionally, the engine's Python.</summary>
public sealed class Settings
{
    public string? Gguf { get; set; }
    public string? Python { get; set; }
    public bool Thinking { get; set; }
    public bool WebSearch { get; set; }
    public string Theme { get; set; } = "native";

    private static string FilePath => Path.Combine(EngineHost.DataDir, "app.json");
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true };

    public static Settings Load()
    {
        Settings? s = null;
        try { s = JsonSerializer.Deserialize<Settings>(File.ReadAllText(FilePath)); }
        catch (IOException) { }
        catch (JsonException) { }
        s ??= new Settings();
        s.Gguf ??= Environment.GetEnvironmentVariable("XDNA1_LLM_GGUF");
        return s;
    }

    public void Save()
    {
        Directory.CreateDirectory(EngineHost.DataDir);
        File.WriteAllText(FilePath, JsonSerializer.Serialize(this, Json));
    }
}
