# Experiment 3 Summary

- Algo: QLEARNING
- Steps: 8000 | Warmup: 500 | γ=0.5
- Alphas: [0.15, 0.45]
- Seeds F: [101, 202] | Seeds M: [303, 404]

## Per-run metrics

| folder                                             | algo      |   alpha |   run |   episodes |   mean_return |   best_return |   mean_steps |   avg_manhattan |   nzF |   nzM |    maxF |   maxM |
|:---------------------------------------------------|:----------|--------:|------:|-----------:|--------------:|--------------:|-------------:|----------------:|------:|------:|--------:|-------:|
| artifacts/exp3-20251104-213201/qlearning_a015_run1 | qlearning |    0.15 |     1 |         17 |       35.2359 |         43.57 |      458.529 |         4.676   |  3123 |  3073 | 1.9455  | 1.9455 |
| artifacts/exp3-20251104-213201/qlearning_a015_run2 | qlearning |    0.15 |     2 |         18 |       36.4678 |         44.89 |      430.444 |         4.96459 |  3218 |  3142 | 2.21167 | 1.9455 |
| artifacts/exp3-20251104-213201/qlearning_a045_run1 | qlearning |    0.45 |     1 |         17 |       35.2359 |         43.57 |      458.529 |         4.676   |  3114 |  3071 | 5.8365  | 5.8365 |
| artifacts/exp3-20251104-213201/qlearning_a045_run2 | qlearning |    0.45 |     2 |         18 |       36.4233 |         44.89 |      430.444 |         4.95959 |  3208 |  3145 | 5.8365  | 5.8365 |
