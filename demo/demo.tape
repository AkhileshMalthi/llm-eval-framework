# LLM Evaluation Framework Demo
# Premium terminal recording with polished styling

# Output configuration
Output docs/assets/demo.mp4

# Window appearance
Set WindowBar Colorful
Set BorderRadius 8
Set Margin 20
Set MarginFill "#1a1b26"

# Typography
Set FontSize 20
Set FontFamily "Fira Code"
Set LineHeight 1.2
Set LetterSpacing 0

# Dimensions
Set Width 1200
Set Height 600
Set Padding 25

# Theme - Tokyo Night (modern, easy on eyes)
Set Theme "TokyoNight"

# Animation settings
Set TypingSpeed 40ms
Set Framerate 30
Set CursorBlink true

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
Type "llm-eval --help"
Enter
Sleep 4s

Type "clear"
Enter
Sleep 500ms

# Step 2: Show example config
Type "cat examples/config.yaml"
Enter
Sleep 4s

Type "clear"
Enter
Sleep 500ms

# Step 3: Run evaluation
Type "llm-eval --config examples/config.yaml --benchmark benchmarks/single_item_benchmark.jsonl"
Enter

# Sleep to allow evaluation to complete (model download + processing)
Sleep 90s
