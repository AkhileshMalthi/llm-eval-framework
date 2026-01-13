# LLM Evaluation Framework Demo
# This tape file creates an animated GIF showing CLI usage

# Output configuration
Output docs/assets/demo.gif

# Terminal settings
Set FontSize 16
Set FontFamily "JetBrains Mono"
Set Width 1200
Set Height 700
Set Theme "Catppuccin Mocha"
Set Padding 20
Set TypingSpeed 50ms

# Hidden setup - install and prepare
Hide
Type "pip install -e . -q"
Enter
Sleep 2s
Type "clear"
Enter
Show

Sleep 500ms

# Step 1: Show help
Type "# LLM Evaluation Framework - Quick Demo"
Enter
Sleep 1s
Type "llm-eval --help"
Enter
Sleep 3s

Type "clear"
Enter
Sleep 500ms

# Step 2: Show example config
Type "# Example configuration file"
Enter
Sleep 500ms
Type "cat examples/config.yaml"
Enter
Sleep 3s

Type "clear"
Enter
Sleep 500ms

# Step 3: Run evaluation with single item (fast for demo)
Type "# Running evaluation..."
Enter
Sleep 500ms
Type "llm-eval --config examples/config.yaml --benchmark benchmarks/single_item_benchmark.jsonl"
Enter

# Wait for completion - match "Evaluation Complete!" with extended timeout for model downloads
Wait+Screen@120s /Evaluation Complete/
Sleep 3s

Type "clear"
Enter
Sleep 500ms

# Step 4: Show results
Type "# Generated outputs:"
Enter
Sleep 500ms
Type "ls -la results/"
Enter
Sleep 2s

# Step 5: Preview report
Type "head -40 results/report.md"
Enter
Sleep 4s

# Finish
Type "clear"
Enter
Sleep 500ms
Type "echo '✅ Done! Visit https://github.com/AkhileshMalthi/llm-eval-framework'"
Enter
Sleep 2s
