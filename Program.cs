using System.Text;
using LLama;
using LLama.Common;
using LLama.Native;
using LLama.Sampling;

// ────────────────────────────────────────────────────────────────
// Debate — LLamaSharp 0.27 + CUDA
// Simulates a debate between two speakers with opposing positions
// on a topic, using a single local model running on GPU.
// Required inputs: topic, speaker 1 position, speaker 2 position,
// debate tone (via command-line arguments or interactive prompts).
// ────────────────────────────────────────────────────────────────

Console.OutputEncoding = Encoding.UTF8;

// Silence llama.cpp native logs
NativeLibraryConfig.All.WithLogCallback((level, message) => { });

// ── 1. Model path ─────────────────────────────────────────────────
string modelPath;
string[] contentArgs;

if (args.Length > 0 && File.Exists(args[0]))
{
    modelPath   = args[0];
    contentArgs = args[1..];
}
else
{
    modelPath   = GetModelPathInteractive();
    contentArgs = args;
}

if (!File.Exists(modelPath))
{
    Console.ForegroundColor = ConsoleColor.Red;
    Console.WriteLine($"File not found: {modelPath}");
    Console.ResetColor();
    return 1;
}

// ── 2. Debate input ───────────────────────────────────────────────
string topic = contentArgs.Length > 0
    ? contentArgs[0]
    : AskInput("Debate topic", ConsoleColor.Cyan);

string position1 = contentArgs.Length > 1
    ? contentArgs[1]
    : AskInput("Position of Speaker 1", ConsoleColor.Green);

string position2 = contentArgs.Length > 2
    ? contentArgs[2]
    : AskInput("Position of Speaker 2", ConsoleColor.Magenta);

string tone = contentArgs.Length > 3
    ? contentArgs[3]
    : AskOptionalInput(
        "Debate tone (e.g. formal, aggressive, ironic, calm) [default: formal and respectful]",
        ConsoleColor.Yellow,
        "formal and respectful");

int exchangeCount = contentArgs.Length > 4 && int.TryParse(contentArgs[4], out int n) && n > 0
    ? n
    : AskExchangeCount();

// ── 3. Model loading ──────────────────────────────────────────────
var parameters = new ModelParams(modelPath)
{
    ContextSize   = 8192,
    GpuLayerCount = 99,
};

Console.WriteLine($"\n Loading: {Path.GetFileName(modelPath)}");
Console.WriteLine(" Backend: CUDA\n");

using var model    = await LLamaWeights.LoadFromFileAsync(parameters);
var       executor = new StatelessExecutor(model, parameters);

var inferenceParams = new InferenceParams
{
    MaxTokens = 400,

    AntiPrompts =
    [
        "<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|end|>",
        "<|endoftext|>", "<end_of_turn>", "</s>", "[/INST]",
    ],

    SamplingPipeline = new DefaultSamplingPipeline
    {
        Temperature   = 0.8f,
        TopP          = 0.9f,
        RepeatPenalty = 1.15f,
    },
};

// ── 4. Identity of the two speakers ────────────────────────────────
const string NameA = "Speaker 1";
const string NameB = "Speaker 2";

string DebaterSystemPrompt(string ownName, string ownPosition, string opponentName) =>
    $"""
    You are {ownName}, a skilled and persuasive debater.
    Debate topic: "{topic}"
    Your position: {ownPosition}
    Required tone: {tone}

    Rules:
    - Firmly support your position using logical arguments, concrete examples and, where useful, plausible data.
    - Directly rebut {opponentName}'s statements when present.
    - Adapt your vocabulary, style and register to the required tone ("{tone}"), while remaining consistent with your position.
    - Do not repeat yourself.
    - Reply ONLY in English, in the first person, with a 3-5 sentence statement.
    - Do not write your name before the reply: write only the text of the statement.
    """;

string ModeratorSystemPrompt =
    $"""
    You are an impartial moderator of public debates. Your task is to
    summarize, in a balanced way and without taking sides, the strengths
    brought by each of the two parties during the debate (which was
    conducted with a {tone} tone) and to close with a brief neutral
    concluding note. Your language remains professional and calm
    regardless of the debate's tone. Reply in English, in at most 6-8
    sentences.
    """;

// ── 5. Header ───────────────────────────────────────────────────────

string header    = $"  AI DEBATE — generated locally with the model {Path.GetFileName(modelPath)}  ";
string separator = new string('═', header.Length);

Console.ForegroundColor = ConsoleColor.Cyan;
Console.WriteLine(separator);
Console.WriteLine(header);
Console.WriteLine(separator);
Console.ResetColor();
Console.WriteLine($"\nTopic: {topic}");
Console.ForegroundColor = ConsoleColor.Green;
Console.WriteLine($"{NameA}: {position1}");
Console.ForegroundColor = ConsoleColor.Magenta;
Console.WriteLine($"{NameB}: {position2}");
Console.ForegroundColor = ConsoleColor.Yellow;
Console.WriteLine($"Tone: {tone}");
Console.ResetColor();
Console.WriteLine();

// ── 6. Debate loop ────────────────────────────────────────────────
var transcript = new StringBuilder();
var log        = new StringBuilder();

log.AppendLine($"DEBATE — {DateTime.Now:yyyy-MM-dd HH:mm} - Model: {Path.GetFileName(modelPath)}");
log.AppendLine($"Topic: {topic}");
log.AppendLine($"{NameA}: {position1}");
log.AppendLine($"{NameB}: {position2}");
log.AppendLine($"Tone: {tone}");
log.AppendLine(new string('-', 60));

for (int round = 1; round <= exchangeCount; round++)
{
    // Speaker 1's turn
    await RunTurn(
        name: NameA,
        color: ConsoleColor.Green,
        systemPrompt: DebaterSystemPrompt(NameA, position1, NameB));

    // Speaker 2's turn
    await RunTurn(
        name: NameB,
        color: ConsoleColor.Magenta,
        systemPrompt: DebaterSystemPrompt(NameB, position2, NameA));
}

// ── 7. Moderator's final summary ───────────────────────────────────
Console.ForegroundColor = ConsoleColor.Yellow;
Console.WriteLine("─── Moderator's summary ───\n");
Console.ResetColor();

string moderatorPrompt = BuildPrompt(
    ModeratorSystemPrompt,
    $"Here is the full debate:\n\n{transcript}\nProvide your impartial summary.");

var summary = new StringBuilder();
await foreach (string token in executor.InferAsync(moderatorPrompt, inferenceParams))
{
    if (IsStopToken(token)) break;
    Console.Write(token);
    summary.Append(token);
}
Console.WriteLine("\n");

log.AppendLine(new string('-', 60));
log.AppendLine("MODERATOR'S SUMMARY:");
log.AppendLine(summary.ToString().Trim());

// ── 8. Saving the transcript ────────────────────────────────────────
try
{
    string fileName = $"debate_{DateTime.Now:yyyyMMdd_HHmmss}.txt";
    string filePath = Path.Combine(AppContext.BaseDirectory, fileName);
    await File.WriteAllTextAsync(filePath, log.ToString(), Encoding.UTF8);

    Console.ForegroundColor = ConsoleColor.DarkGray;
    Console.WriteLine($"Transcript saved to: {filePath}");
    Console.ResetColor();
}
catch (Exception ex)
{
    Console.ForegroundColor = ConsoleColor.Red;
    Console.WriteLine($"Unable to save transcript: {ex.Message}");
    Console.ResetColor();
}

Console.ForegroundColor = ConsoleColor.DarkGray;
Console.WriteLine("Debate finished.");
Console.ResetColor();
return 0;

// ── Helpers ──────────────────────────────────────────────────────

/// <summary>
/// Runs a single speaking turn for a debater: builds the prompt from the
/// conversation so far, streams the model's response to the console token
/// by token, and appends the resulting statement to both the transcript
/// and the log.
/// </summary>
/// <param name="name">Display name of the speaker taking this turn.</param>
/// <param name="color">Console color used to highlight the speaker's name and output.</param>
/// <param name="systemPrompt">System prompt describing the speaker's role, position and tone.</param>
async Task RunTurn(string name, ConsoleColor color, string systemPrompt)
{
    string userMessage = transcript.Length == 0
        ? "The debate is about to begin. Open the discussion with a statement that clearly and persuasively presents your position."
        : $"Here is the debate so far:\n\n{transcript}\nIt's your turn: reply by rebutting or reinforcing your position.";

    string prompt = BuildPrompt(systemPrompt, userMessage);

    Console.ForegroundColor = color;
    Console.Write($"{name}: ");
    Console.ResetColor();

    var responseBuilder = new StringBuilder();
    await foreach (string token in executor.InferAsync(prompt, inferenceParams))
    {
        if (IsStopToken(token)) break;
        Console.Write(token);
        responseBuilder.Append(token);
    }
    Console.WriteLine("\n");

    string reply = responseBuilder.ToString().Trim();
    transcript.Append($"{name}: {reply}\n\n");
    log.AppendLine($"{name}: {reply}");
}

/// <summary>
/// Builds a chat-formatted prompt (ChatML-style) combining a system prompt
/// and a user message, ready to be passed to the inference executor.
/// </summary>
/// <param name="system">The system prompt that sets the role and rules for the model.</param>
/// <param name="userMessage">The user message the model should respond to.</param>
/// <returns>The fully formatted prompt string.</returns>
static string BuildPrompt(string system, string userMessage)
{
    var sb = new StringBuilder();
    sb.Append($"<|im_start|>system\n{system}<|im_end|>\n");
    sb.Append($"<|im_start|>user\n{userMessage}<|im_end|>\n");
    sb.Append("<|im_start|>assistant\n");
    return sb.ToString();
}

/// <summary>
/// Determines whether a generated token is one of the known stop/end-of-turn
/// tokens and should therefore not be emitted or appended to the output.
/// </summary>
/// <param name="token">The token produced by the inference executor.</param>
/// <returns><see langword="true"/> if the token marks the end of a turn; otherwise <see langword="false"/>.</returns>
static bool IsStopToken(string token) =>
    token is "<|im_end|>" or "<|eot_id|>" or "<|end|>"
          or "<|endoftext|>" or "<end_of_turn>" or "</s>";

/// <summary>
/// Prompts the user for a required text value on the console, re-asking
/// until a non-empty value is provided.
/// </summary>
/// <param name="label">The label displayed to the user before the input prompt.</param>
/// <param name="color">Console color used for the label.</param>
/// <returns>The trimmed, non-empty value entered by the user.</returns>
static string AskInput(string label, ConsoleColor color)
{
    Console.ForegroundColor = color;
    Console.Write($"{label}: ");
    Console.ResetColor();
    string? value = Console.ReadLine()?.Trim();
    while (string.IsNullOrWhiteSpace(value))
    {
        Console.ForegroundColor = ConsoleColor.Red;
        Console.Write($"Value required. {label}: ");
        Console.ResetColor();
        value = Console.ReadLine()?.Trim();
    }
    return value;
}

/// <summary>
/// Prompts the user for an optional text value on the console, returning a
/// default value when the user leaves the input empty.
/// </summary>
/// <param name="label">The label displayed to the user before the input prompt.</param>
/// <param name="color">Console color used for the label.</param>
/// <param name="defaultValue">The value returned when the user provides no input.</param>
/// <returns>The trimmed value entered by the user, or <paramref name="defaultValue"/> if none was provided.</returns>
static string AskOptionalInput(string label, ConsoleColor color, string defaultValue)
{
    Console.ForegroundColor = color;
    Console.Write($"{label}: ");
    Console.ResetColor();
    string? value = Console.ReadLine()?.Trim();
    return string.IsNullOrWhiteSpace(value) ? defaultValue : value;
}

/// <summary>
/// Prompts the user for the number of exchanges per speaker, falling back
/// to a default of 4 if the input is missing or invalid.
/// </summary>
/// <returns>The number of exchanges per speaker to run.</returns>
static int AskExchangeCount()
{
    Console.ForegroundColor = ConsoleColor.Cyan;
    Console.Write("Number of exchanges per speaker [default 4]: ");
    Console.ResetColor();
    string? input = Console.ReadLine()?.Trim();
    if (int.TryParse(input, out int value) && value > 0) return value;
    return 4;
}

/// <summary>
/// Interactively asks the user for the path to a local GGUF model file,
/// printing a welcome banner and example download links first.
/// </summary>
/// <returns>The trimmed, unquoted path entered by the user (may be empty).</returns>
static string GetModelPathInteractive()
{
    Console.ForegroundColor = ConsoleColor.Cyan;
    Console.WriteLine("╔══════════════════════════════════════════════╗");
    Console.WriteLine("║        AI Debate — LLamaSharp + CUDA          ║");
    Console.WriteLine("╚══════════════════════════════════════════════╝");
    Console.ResetColor();
    Console.WriteLine("\nDownload GGUF models from HuggingFace, for example:");
    Console.ForegroundColor = ConsoleColor.DarkGray;
    Console.WriteLine("  https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF");
    Console.WriteLine("  https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF");
    Console.ResetColor();
    Console.Write("\nPath to .gguf file: ");
    return Console.ReadLine()?.Trim().Trim('"') ?? string.Empty;
}
