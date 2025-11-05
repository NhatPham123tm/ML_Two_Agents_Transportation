# Experiment 3 Summary

- Algo: QLEARNING
- Steps: 8000 | Warmup: 500 | γ=0.5
- Alphas: [0.15, 0.45]
- Seeds F: [101, 202] | Seeds M: [303, 404]

## Per-run metrics

| folder                                             | algo      |   alpha |   run |   episodes |   mean_return |   best_return |   mean_steps |   avg_manhattan |   nzF |   nzM |    maxF |    maxM |
|:---------------------------------------------------|:----------|--------:|------:|-----------:|--------------:|--------------:|-------------:|----------------:|------:|------:|--------:|--------:|
| artifacts/exp3-20251031-152518/qlearning_a015_run1 | qlearning |    0.15 |     1 |         16 |      -28.75   |           -15 |          500 |         4.59162 |  3593 |  3599 | 1.89079 | 2.74725 |
| artifacts/exp3-20251031-152518/qlearning_a015_run2 | qlearning |    0.15 |     2 |         16 |      -24.6875 |           -15 |          500 |         4.598   |  3602 |  3599 | 1.89079 | 3.82016 |
| artifacts/exp3-20251031-152518/qlearning_a045_run1 | qlearning |    0.45 |     1 |         16 |      -28.75   |           -15 |          500 |         4.59162 |  3593 |  3599 | 4.455   | 6.90525 |
| artifacts/exp3-20251031-152518/qlearning_a045_run2 | qlearning |    0.45 |     2 |         16 |      -24.6875 |           -15 |          500 |         4.598   |  3602 |  3599 | 4.455   | 8.25289 |
