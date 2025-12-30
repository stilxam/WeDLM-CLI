# WeDLM CLI

An interactive terminal interface for **Tencent's WeDLM-8B-Instruct**, a high-speed diffusion language model. This CLI features live Markdown streaming, context-aware file loading, and real-time model parameter tuning.

## Quick Start

### 1. Build the Environment
This project uses a Nix Flake to manage all dependencies (CUDA, PyTorch, WeDLM). Ensure you have [Nix](https://nixos.org/download.html) installed with flakes enabled and the Nix CUDA cache enabled in order to avoid compiling all the CUDA libraries locally.

```bash
nix develop
```
*The first time you run this, it will automatically download the WeDLM-8B-Instruct model (approx. 15GB) to your local directory.*

### 2. Launch the CLI
Once inside the Nix shell, start the chat interface:

```bash
python cli.py
```

---

## Model Settings

You can adjust the model's behavior in real-time using commands directly in the chat.

| Command | Parameter | Default | Description |
| :--- | :--- | :--- | :--- |
| `/temp=v` | **Temperature** | `0.1` | Controls randomness. Lower is more focused; higher is more creative. |
| `/entropy=v` | **Entropy Threshold** | `0.4` | Controls parallel decoding confidence. Lower values are more conservative; higher values allow more aggressive parallel generation. |
| `/penalty=v` | **Position Penalty** | `0.02` | Adjusts how the model weights tokens based on position during diffusion. Helps maintain structural coherence. |

---

## Chat Commands

*   **`/help`**: Show all available commands.
*   **`/settings`**: Display your current Temperature, Entropy, and Penalty values.
*   **`/clear`**: Wipe the conversation history to start fresh.
*   **`/exit`**: Safely close the CLI.
*   **`@path/to/file.txt`**: Type `@` followed by a file path to inject that file's content into your prompt as context.

---

## Credits & Licensing

*   **Model**: [WeDLM-8B-Instruct](https://github.com/Tencent/WeDLM) by Tencent.
*   **License**: Licensed under the WeDLM License (Apache 2.0 with Territorial Limitations).
*   **Restriction**: Per the Tencent license, this model is not intended for use within the European Union.

*   **Base Engine**: Built with components from Qwen (Alibaba Cloud) and nano-vllm.

*This is an unofficial CLI.*
