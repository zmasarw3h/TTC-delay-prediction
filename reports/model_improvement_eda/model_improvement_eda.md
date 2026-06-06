# Model Improvement EDA

Phase 11A diagnostic report for planning historical features. This run does not train models or modify model artifacts.

Generated: 2026-06-06T15:33:30.943437+00:00
Validation rows: 70063
Test rows: 73134

## Highest-Priority Phase 11B Candidates

1. `prior_route_incident_mean_delay` (prior_mean, both): Captures route-specific incident severity while using prior-only history. Support: 96.1% of eval rows have at least 20 prior observations. EDA coverage: 99.8%.
2. `prior_mode_incident_mean_delay` (prior_mean, both): Lower-cardinality fallback for incident severity by vehicle mode. Support: 100.0% of eval rows have at least 20 prior observations. EDA coverage: 100.0%.
3. `prior_route_direction_mean_delay` (prior_mean, regression): Adds directional route behavior beyond current route-level history. Support: 99.0% of eval rows have at least 20 prior observations. EDA coverage: 99.9%.
4. `prior_route_incident_count` (prior_count, both): Exposes support/confidence for route-incident historical means. Support: 96.1% of eval rows have at least 20 prior observations.
5. `prior_route_30d_mean_delay` (rolling_mean, regression): Current recent route behavior may capture service disruptions and seasonal drift. Support: 99.8% of eval rows have at least 20 prior observations. EDA coverage: 99.7%.
6. `prior_incident_30d_mean_delay` (rolling_mean, both): Recent incident severity can adapt to operational changes with broad support. Support: 100.0% of eval rows have at least 20 prior observations. EDA coverage: 100.0%.
7. `prior_route_30d_severe_rate_30` (rolling_rate, severe_30): Directly targets severe-delay propensity using recent prior route outcomes. Support: 99.8% of eval rows have at least 20 prior observations. EDA coverage: 99.7%.
8. `prior_incident_30d_severe_rate_30` (rolling_rate, severe_30): Direct incident-level severe-rate signal with lower cardinality than route incident. Support: 100.0% of eval rows have at least 20 prior observations. EDA coverage: 100.0%.

## Candidate Support Snapshot

               grouping           group_columns  row_count  pct_with_prior_1  pct_with_prior_5  pct_with_prior_20  pct_with_prior_50  median_prior_count  p25_prior_count  p75_prior_count  max_prior_count
               Incident                Incident     143197        100.000000        100.000000         100.000000         100.000000             43494.0          19726.0         260213.0           296012
                   mode                    mode     143197        100.000000        100.000000         100.000000         100.000000            628568.0         592769.0         664367.0           700166
        mode + Incident         mode + Incident     143197        100.000000        100.000000         100.000000         100.000000             38561.0          12070.0         213717.0           249516
                  Route                   Route     143197         99.980447         99.942736          99.829605          99.628484              6761.0           2452.0          12495.0            37597
      Route + Direction       Route + Direction     143197         99.942038         99.750693          98.968554          97.280669              1226.0            390.0           4278.0            16014
           Route + hour            Route + hour     143197         99.745805         99.016739          96.476183          90.332200               294.0            115.0            637.0             2523
       Route + Incident        Route + Incident     143197         99.834494         99.206687          96.053688          89.869201               499.0            161.0           1328.0            13232
               Location                Location     143197         92.469814         85.955013          76.162908          66.833104               167.0             22.0           1286.0            17021
Route + Incident + hour Route + Incident + hour     143197         95.939161         81.917917          55.242777          34.508405                25.0              7.0             78.0             1213
    Location + Incident     Location + Incident     143197         82.879530         66.166191          46.630167          34.645279                15.0              2.0            110.0             9757
        Location + hour         Location + hour     143197         78.199962         57.737941          35.920445          25.960739                 8.0              1.0             56.0             2063

## Output Files

- `error_by_group.csv`
- `error_contribution_by_group.csv`
- `severe_delay_by_group.csv`
- `candidate_group_support.csv`
- `candidate_prior_mean_scores.csv`
- `rolling_window_opportunity.csv`
- `feature_recommendations.csv`
- `model_improvement_eda_summary.json`
