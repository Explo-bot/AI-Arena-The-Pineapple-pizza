# AI Arena: The Pineapple Pizza

> **⚠️ Note:** This project was written in large part with the support of **Claude Cowork**. The author has only made a few marginal additions and edits.

A .NET console application that simulates a debate between two AI speakers holding opposing positions on a topic, using a single local LLM running fully offline on GPU via [LLamaSharp](https://github.com/SciSharp/LLamaSharp).

Give it a topic, two positions, and a tone; the app runs a multi-round debate between "Speaker 1" and "Speaker 2", then has an impartial moderator summarize the exchange. The full transcript is saved to a text file. Debate content is generated in English by default (configurable via the system prompts in `Program.cs`).

## Features

- Runs entirely locally on your own GPU — no external API calls, no data leaves your machine.
- Simulates a structured, multi-round debate between two opposing viewpoints.
- Configurable topic, positions, tone, and number of exchanges.
- Generates an impartial closing summary from a moderator persona.
- Saves the full transcript (including the summary) to a timestamped `.txt` file.
- Works both interactively (prompts) and non-interactively (command-line arguments).

## Requirements

- [.NET 10 SDK](https://dotnet.microsoft.com/download) or later.
- An NVIDIA GPU with CUDA 12 support (the project references `LLamaSharp.Backend.Cuda12`). If you have CUDA 11 instead, swap the package reference in `AI Arena.csproj` for `LLamaSharp.Backend.Cuda11`.
- A local model file in GGUF format. You can download one from HuggingFace, for example:
  - [Qwen2.5-14B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF)
  - [Meta-Llama-3.1-8B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF)

## Getting started

Clone the repository and restore/build with the .NET CLI:

```bash
git clone <repository-url>
cd "AI Arena"
dotnet build
```

### Run interactively

```bash
dotnet run
```

You'll be prompted for the path to your `.gguf` model file, followed by the debate topic, each speaker's position, the tone, and the number of exchanges.

### Run with command-line arguments

```bash
dotnet run -- <path-to-model.gguf> "<topic>" "<position 1>" "<position 2>" "<tone>" <exchanges>
```

Example:

```bash
dotnet run -- ./models/qwen2.5-14b-instruct-q4_k_m.gguf \
  "Should remote work be the default for office jobs?" \
  "Remote work increases productivity and quality of life" \
  "In-person work is essential for collaboration and culture" \
  "formal and respectful" \
  4
```

Any arguments you omit will be requested interactively. The `<exchanges>` value is the number of turns per speaker (default: 4 if omitted or invalid).

## How it works

1. The app loads the specified GGUF model into memory using `LLamaWeights` with GPU offloading (`GpuLayerCount = 99`) and an 8192-token context window.
2. Two independent system prompts are built — one per speaker — instructing each to argue persuasively for its assigned position and to rebut the opponent when relevant.
3. For each round, both speakers take a turn generating a response conditioned on the debate transcript so far.
4. After all rounds complete, a moderator persona summarizes the debate impartially.
5. The full transcript and summary are written to a file named `debate_yyyyMMdd_HHmmss.txt` in the application's base directory.

## Project structure

- `Program.cs` — application entry point and all logic (model loading, prompt construction, debate loop, transcript persistence).
- `AI Arena.csproj` — project file targeting .NET 10 with the LLamaSharp CUDA 12 backend.

## Testing

Tested on an NVIDIA RTX 4060 Ti with 16GB VRAM. Among the models tried, **Phi-4 Q4_K_M (15B parameters)** gave the best results in terms of debate quality and speed on this hardware.

## Example

The repository includes a sample transcript, [`The pineapple pizza .txt`](./The%20pineapple%20pizza%20.txt), generated with the Phi-4 Q4_K_M model on the topic "Is pineapple pizza a crime or an innovation?" — useful as a reference for the kind of output the app produces.

## Notes

- Model loading and inference speed depend on your GPU's VRAM and the size of the chosen GGUF model.
- If no GPU is available or you want to use a different backend, replace `LLamaSharp.Backend.Cuda12` in `AI Arena.csproj` with the appropriate LLamaSharp backend package (e.g. CPU or Cuda11) and adjust `GpuLayerCount` accordingly.

## License

This project is licensed under the [Apache License 2.0](./LICENSE.txt). See the `LICENSE` file for the full text.
