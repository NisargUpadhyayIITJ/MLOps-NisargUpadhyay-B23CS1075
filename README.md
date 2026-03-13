# MLOps Assignments Portfolio

This repository contains coursework for the MLOps course by Nisarg Upadhyay (`B23CS1075`, IIT Jodhpur).

## Assignment 4

Assignment 4 refactors the original English-to-Hindi transformer notebook into a reusable training and tuning pipeline with Ray Tune, Optuna, and ASHA.

### Key files

- `en_to_hi.ipynb`: original baseline notebook provided for the assignment
- `b23cs1075_ass_4_tuned_en_to_hi.py`: Assignment 4 baseline + tuning + final-training workflow
- `assignment4_requirements.txt`: packages used for the Assignment 4 environment
- `b23cs1075_ass_4_report.md`: auto-generated report source
- `b23cs1075_ass_4_report.pdf`: exported PDF report artifact
- `docs/assignment4.html`: GitHub Pages summary for Assignment 4

### Notebook baseline captured from `en_to_hi.ipynb`

- Training time for 100 epochs: `52.32` minutes
- Final loss: `0.0974`
- BLEU score: `52.47`

These baseline values were extracted from the recorded notebook outputs already saved in `en_to_hi.ipynb`.

### Install with the requested `uv` environment

```bash
uv pip install --python ../ops_venv/bin/python -r assignment4_requirements.txt
```

### Run the workflow

```bash
../ops_venv/bin/python b23cs1075_ass_4_tuned_en_to_hi.py --action baseline
../ops_venv/bin/python b23cs1075_ass_4_tuned_en_to_hi.py --action tune --num-samples 20 --tune-epochs 20 --cpus-per-trial 2 --gpus-per-trial 1
../ops_venv/bin/python b23cs1075_ass_4_tuned_en_to_hi.py --action final --final-epochs 30 --target-bleu 0.5247
../ops_venv/bin/python b23cs1075_ass_4_tuned_en_to_hi.py --action report
```

### Outputs written by the script

- `transformer_translation_final.pth`: baseline model weights
- `b23cs1075_ass_4_best_model.pth`: best-model weights after the tuned final run
- `artifacts/assignment4/best_config.json`: best configuration discovered by Ray Tune + Optuna
- `artifacts/assignment4/summary.json`: combined baseline/tuning/final metrics
- `b23cs1075_ass_4_report.md`: markdown report source generated from the latest summary
- `b23cs1075_ass_4_report.pdf`: PDF export generated from the markdown report

### Current repository status

- Baseline training path: validated
- Ray Tune + Optuna + ASHA integration: validated with a 2-trial, 2-epoch smoke sweep
- Final best-model training path: validated with a short 3-epoch run
- Submission artifacts currently present: `.py`, `.pdf`, `.pth`, metrics summary, docs page

The current tuning metrics come from a short validation run to prove the end-to-end pipeline. Re-running the sweep with larger `--num-samples` and `--tune-epochs` values is the next step if you want to push BLEU closer to or beyond the notebook baseline.
The GitHub Pages homepage is served from the `docs/` directory and now includes an Assignment 4 entry.
