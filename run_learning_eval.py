"""Alias entrypoint for evaluate_generalization (see that module for details).

    ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe run_learning_eval.py \\
        --games public_unseen_split --weights 0,0.5,1.0
"""

from evaluate_generalization import main

if __name__ == "__main__":
    main()
