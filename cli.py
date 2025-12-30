import os
import sys
import re
import argparse
import readline
import time
import random
import threading
from typing import List, Dict, Tuple

from transformers import AutoTokenizer
from wedlm import LLM, SamplingParams

from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

BLUE, YELLOW, RESET = "\033[94m", "\033[93m", "\033[0m"

console = Console()

LOGO = rf"""

▄▄      ▄▄           ▄▄▄▄▄     ▄▄        ▄▄▄  ▄▄▄               ▄▄▄▄   ▄▄         ▄▄▄▄▄▄
██      ██           ██▀▀▀██   ██        ███  ███             ██▀▀▀▀█  ██         ▀▀██▀▀
▀█▄ ██ ▄█▀  ▄████▄   ██    ██  ██        ████████            ██▀       ██           ██
 ██ ██ ██  ██▄▄▄▄██  ██    ██  ██        ██ ██ ██            ██        ██           ██
 ███▀▀███  ██▀▀▀▀▀▀  ██    ██  ██        ██ ▀▀ ██            ██▄       ██           ██
 ███  ███  ▀██▄▄▄▄█  ██▄▄▄██   ██▄▄▄▄▄▄  ██    ██     ██      ██▄▄▄▄█  ██▄▄▄▄▄▄   ▄▄██▄▄
 ▀▀▀  ▀▀▀    ▀▀▀▀▀   ▀▀▀▀▀     ▀▀▀▀▀▀▀▀  ▀▀    ▀▀     ▀▀        ▀▀▀▀   ▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀


       Causal Attention Reordered Diffusion CLI
"""

def animate_logo(stop_event: threading.Event, total_duration: float = 1.):
    """
    Animates the logo with a progressive denoising effect.
    Runs in a separate thread and stops when the stop_event is set.
    """
    mask_char = '▒'
    lines = LOGO.strip().split('\n')

    positions = []
    for r, line in enumerate(lines):
        for c, char in enumerate(line):
            if char != ' ':
                positions.append((r, c))

    random.shuffle(positions)

    masked_lines = [[char for char in line] for line in lines]
    for r, c in positions:
        masked_lines[r][c] = mask_char

    num_steps = 100  # More steps for a smoother animation
    reveal_per_step = (len(positions) // num_steps) + 1
    sleep_interval = total_duration / num_steps

    revealed_count = 0
    while not stop_event.is_set() and revealed_count < len(positions):
        # Clear the terminal screen
        os.system('cls' if os.name == 'nt' else 'clear')

        # Reveal a new chunk of characters
        for _ in range(reveal_per_step):
            if positions:
                r, c = positions.pop()
                masked_lines[r][c] = lines[r][c]
                revealed_count += 1

        # Print the current state of the logo
        for line_list in masked_lines:
            print("".join(line_list))

        time.sleep(sleep_interval)

    # Once done or stopped, ensure the final, clean logo is printed
    os.system('cls' if os.name == 'nt' else 'clear')
    print(LOGO)

# ==================== Configuration ====================
GPU_MEMORY_UTILIZATION = 0.95
# MAX_MODEL_LEN =8192
MAX_MODEL_LEN = 16384

MAX_NUM_SEQS = 128
WeDLM_WINDOW_SIZE = 16
MAX_FILE_SIZE_KB = 128

def parse_args():
    """Parse command-line arguments to set initial model settings."""
    parser = argparse.ArgumentParser(description="WeDLM CLI Chat with Model Controls")
    parser.add_argument("--model-path", type=str, default="./WeDLM-8B-Instruct", help="Path to the downloaded model directory")
    parser.add_argument("--temperature", type=float, default=0.1, help="Initial temperature (0.0-2.0)")
    parser.add_argument("--entropy", type=float, default=0.4, help="Initial WeDLM entropy threshold (0.1-1.0)")
    parser.add_argument("--penalty", type=float, default=0.02, help="Initial WeDLM position penalty factor (0.0-0.1)")
    return parser.parse_args()

def get_stop_tokens(tokenizer):
    """Identify token IDs that signal the end of generation."""
    stop_ids = []
    if tokenizer.eos_token_id is not None:
        stop_ids.append(tokenizer.eos_token_id)
    extra_stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    for token_str in extra_stop_tokens:
        if token_str in tokenizer.get_vocab():
            tid = tokenizer.convert_tokens_to_ids(token_str)
            if tid not in stop_ids:
                stop_ids.append(tid)
    return stop_ids

def process_file_references(user_input: str) -> Tuple[str, bool]:
    """Detects @<path>, reads files, and formats them into the prompt."""
    pattern = re.compile(r'@(\S+)')
    matches = pattern.findall(user_input)
    if not matches:
        return user_input, True

    file_contexts = []
    for path in matches:
        expanded_path = os.path.expanduser(path)
        if not os.path.exists(expanded_path):
            console.print(f"[red]Error: File not found at '{expanded_path}'[/red]")
            return "", False
        if not os.path.isfile(expanded_path):
            console.print(f"[red]Error: Path '{expanded_path}' is a directory, not a file.[/red]")
            return "", False
        file_size = os.path.getsize(expanded_path)
        if file_size > MAX_FILE_SIZE_KB * 1024:
            console.print(f"[red]Error: File '{expanded_path}' is too large ({file_size/1024:.1f} KB). Limit is {MAX_FILE_SIZE_KB} KB.[/red]")
            return "", False
        try:
            with open(expanded_path, 'r', encoding='utf-8') as f:
                content = f.read()
            context_block = (
                f"--- File Content: {os.path.basename(expanded_path)} ---\n"
                f"{content.strip()}\n"
                f"--- End of File Content ---\n"
            )
            file_contexts.append(context_block)
            console.print(f"[yellow]Loaded context from: {expanded_path}[/yellow]")
        except Exception as e:
            console.print(f"[red]Error reading file '{expanded_path}': {e}[/red]")
            return "", False

    cleaned_prompt = pattern.sub('', user_input).strip()
    final_prompt = "\n".join(file_contexts) + "\n" + cleaned_prompt
    return final_prompt, True

def handle_command(user_input: str, settings: Dict) -> bool:
    """Parses and executes in-chat commands. Returns True if input was a command."""
    if not user_input.startswith('/'):
        return False

    cmd, _, value = user_input.partition('=')
    cmd = cmd.lower().strip()

    try:
        if cmd in ('/temp', '/temperature'):
            new_val = float(value)
            if 0.0 <= new_val <= 2.0:
                settings['temperature'] = new_val
                console.print(f"[yellow]Temperature set to {new_val}[/yellow]")
            else:
                console.print(f"[red]Error: Temperature must be between 0.0 and 2.0.[/red]")
        elif cmd == '/entropy':
            new_val = float(value)
            if 0.1 <= new_val <= 1.0:
                settings['entropy'] = new_val
                console.print(f"[yellow]Entropy threshold set to {new_val}[/yellow]")
            else:
                console.print(f"[red]Error: Entropy must be between 0.1 and 1.0.[/red]")
        elif cmd == '/penalty':
            new_val = float(value)
            if 0.0 <= new_val <= 0.1:
                settings['penalty'] = new_val
                console.print(f"[yellow]Position penalty set to {new_val}[/yellow]")
            else:
                console.print(f"[red]Error: Penalty must be between 0.0 and 0.1.[/red]")
        elif cmd == '/settings':
            console.print(f"[yellow]Current settings:[/yellow]\n  - Temperature: {settings['temperature']}\n  - Entropy Threshold: {settings['entropy']}\n  - Position Penalty: {settings['penalty']}")
        elif cmd == '/help':
            print_help()
        else:
            console.print(f"[red]Unknown command: '{cmd}'. Type /help for a list of commands.[/red]")

    except (ValueError, IndexError):
        console.print(f"[red]Invalid command format. Use /command=value (e.g., /temp=0.5).[/red]")

    return True

def print_help():
    """Prints the help message with all available commands."""
    console.print(f"\n[bold]Available Commands:[/bold]")
    console.print(f"  {'[cyan]/help[/cyan]':<35} Show this help message.")
    console.print(f"  {'[cyan]/settings[/cyan]':<35} Display the current generation settings.")
    console.print(f"  {'[cyan]/clear[/cyan]':<35} Clear the conversation history.")
    console.print(f"  {'[cyan]/exit[/cyan]':<35} Quit the program.")
    console.print(f"  {'[cyan]@/path/to/file.txt[/cyan]':<35} Add file content to your prompt.")
    console.print(f"\n[bold]Generation Parameters:[/bold]")
    console.print(f"  {'[cyan]/temperature=<value>[/cyan]':<35} (or /temp) Set temperature (e.g., 0.5).")
    console.print(f"  {'[cyan]/entropy=<value>[/cyan]':<35} Set WeDLM entropy (e.g., 0.4).")
    console.print(f"  {'[cyan]/penalty=<value>[/cyan]':<35} Set WeDLM penalty (e.g., 0.02).\n")

def main():
    args = parse_args()

    current_settings = {
        "temperature": args.temperature,
        "entropy": args.entropy,
        "penalty": args.penalty,
    }

    # --- Concurrent Model Loading and Animation ---
    stop_animation_event = threading.Event()
    animation_thread = threading.Thread(target=animate_logo, args=(stop_animation_event,))
    animation_thread.start()

    loading_result = {}

    def load_model_worker():
        console.print(f"[yellow]Loading Tokenizer from {args.model_path}...[/yellow]", style="white on blue")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        loading_result['tokenizer'] = tokenizer

        console.print(f"[yellow]Loading WeDLM Engine...[/yellow]", style="white on blue")
        llm = LLM(
            args.model_path,
            tensor_parallel_size=1,
            wedlm_window_size=WeDLM_WINDOW_SIZE,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_num_seqs=MAX_NUM_SEQS
        )
        loading_result['llm'] = llm

    loader_thread = threading.Thread(target=load_model_worker)
    loader_thread.start()

    loader_thread.join()

    stop_animation_event.set()
    animation_thread.join()

    llm = loading_result['llm']
    tokenizer = loading_result['tokenizer']

    stop_token_ids = get_stop_tokens(tokenizer)
    messages: List[Dict[str, str]] = []
    console.print(f"\n[green bold]Model Ready! Type /help for commands.[/green bold]")

    while True:
        try:
            current_tokens = 0
            if messages:
                try:
                    encoded_history = tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=False
                    )
                    current_tokens = len(encoded_history)
                except Exception:
                    pass

            count_color = "cyan" if current_tokens < MAX_MODEL_LEN * 0.9 else "red"
            prompt_str = f"[blue bold]You ([{count_color}]{current_tokens}/{MAX_MODEL_LEN}[/]):[/] "
            user_input = console.input(prompt_str)

            if not user_input:
                continue
            if user_input.lower() in ['/exit', '/quit']:
                break
            if user_input.lower() == '/clear':
                messages = []
                console.print(f"[yellow]Conversation history cleared.[/yellow]")
                continue
            if handle_command(user_input, current_settings):
                continue

            processed_input, success = process_file_references(user_input)
            if not success:
                continue

            messages.append({"role": "user", "content": processed_input})
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            sampling_params = SamplingParams(
                temperature=current_settings['temperature'],
                max_tokens=2048,
                stop_token_ids=stop_token_ids,
                wedlm_entropy_threshold=current_settings['entropy'],
                wedlm_pos_penalty_factor=current_settings['penalty']
            )

            console.print(f"[green bold]AI:[/] ", end="")
            full_response = ""

            with Live(console=console, auto_refresh=False, vertical_overflow="visible") as live:
                for output in llm.generate_stream([formatted_prompt], sampling_params):
                    if 'new_text' in output:
                        full_response += output['new_text']
                        live.update(Markdown(full_response, code_theme="monokai"), refresh=True)

            messages.append({"role": "assistant", "content": full_response})

        except KeyboardInterrupt:
            console.print()
            continue
        except Exception as e:
            console.print(f"\n[red bold]An unexpected error occurred:[/red bold] {e}")

if __name__ == "__main__":
    main()
