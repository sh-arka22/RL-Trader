# W&B in 2026 for Reinforcement-Learning HPO: Pricing, Limits, and Alternatives

**Fetch date: 2026-09-14.** I checked the live W&B pricing page and current documentation pages, plus current Optuna, Ray Tune, CleanRL, RL Baselines3 Zoo, PRIME-RL, and paper sources. A key distinction is that W&B now presents both a cloud-hosted **Free** plan and a privately hosted **Personal** plan. They are not the same entitlement.

## Executive summary

- **Best role for W&B:** Use W&B primarily as the experiment-tracking, artifact, visualization, collaboration, and audit layer. Its Sweeps service can also propose trials, but Optuna and Ray Tune are stronger HPO engines for pruning, persistent studies, scheduling, and complex distributed execution.
- **Free cloud plan:** The current pricing page shows **$0/month**, model seats **up to 5**, and **5 GB/month** storage. The page does not identify that storage as an artifact-only quota, does not state tracked-hour or retention limits, and does not publish a numeric run limit for the plan. [2]
- **Personal plan:** The privately hosted Personal plan is **$0/month**, has **1 user seat**, includes experiment tracking and registry/lineage tracking, and is for personal projects only. Corporate use is explicitly prohibited. Storage, tracked hours, and retention are not stated on the page. [2]
- **Academic plan:** Academic research is advertised as free forever. The free academic license has Pro features, unlimited tracked hours, **200 GB** cloud storage, up to **25 GB/month** Weave ingestion, and up to **100 seats**. Students are told to start with the pricing-page trial and convert it through the academic application process. [2]
- **Paid cloud plans:** Pro starts at **$60/month**, billed monthly. Enterprise is custom-priced. The visible page does not publish a 2025/2026 change log, so no specific tier change should be asserted from the live page. [2]
- **Sweeps:** Grid, random, and Bayesian methods are supported; YAML or Python configuration is supported; Hyperband early termination is available. The official documentation does not identify the Bayesian surrogate or acquisition function, so calling it Gaussian-process BO, TPE, or a particular implementation would be unsupported. [1]
- **RL logging:** Do not call `wandb.log()` for every environment transition in a high-throughput vectorized rollout. W&B recommends fewer than **1,000 log calls/minute**, fewer than **100,000 values/minute**, and fewer than **40 MB/minute** of video; these are performance guidelines, not hard product limits. [10]
- **RL HPO:** PPO should prioritize implementation correctness, observation normalization, rollout/batch structure, learning rate, clipping and value-related choices before adding many exotic knobs. SAC should tune the learning-rate/critic-temperature/exploration and replay-related choices jointly, evaluate across seeds, and use a held-out test seed set. Implementation and seed effects can dominate apparent algorithmic improvements. [17][7][11]

## A. W&B 2026 pricing and limits

| Plan | Current visible price | Seats | Tracking/storage shown | Tracked hours | Retention | Commercial restriction / notes |
|---|---:|---:|---|---|---|---|
| Free cloud | $0/mo | Up to 5 model seats | 5 GB/month storage; page does not say whether this is specifically artifacts, runs, or both | Not stated | Not stated | Described for personal development; the explicit corporate prohibition belongs to Personal, not this Free row [2] |
| Personal, privately hosted | $0/mo | 1 user seat | Experiment tracking and registry/lineage; storage not stated | Not stated | Not stated | Personal projects only; corporate use is not allowed [2] |
| Pro | Starts at $60/month, monthly billing | Up to 10 model seats in the comparison | 100 GB/month storage; 1.5 GB/month Weave ingestion | Not stated | Customizable data retention is listed | Intended for professionals and early-stage teams with fewer than 50 employees [2] |
| Enterprise | Custom | Customizable/not stated | Customizable/not stated | Not stated | Not stated | Security/compliance features include single tenant, private connectivity, CMEK, SSO, SCIM, custom roles, audit logs, and enterprise support [2] |
| Academic | Free forever for academic research | Up to 100 seats | 200 GB cloud storage; up to 25 GB/month Weave ingestion; extra cloud storage $0.03/GB/month | Unlimited | Not stated | Research must be academic and not connected to a for-profit entity; intended for students, professors, and postdoctoral researchers [2] |

### Academic eligibility and application

W&B describes the academic license as a free Pro license for academic institutions pursuing research not connected to a for-profit entity, intended for students, professors, and postdoctoral researchers. [2] The live pricing page links **Apply Now** to `https://wandb.ai/create-team`, while the student support page says to visit the pricing page, start with a 30-day trial, and convert it through the academic application page. [2]

The academic entitlement is materially better than the ordinary free cloud plan for RL sweeps: unlimited tracked hours, 200 GB cloud storage, 100 seats, and Pro features. The page does not state academic artifact-retention duration, so do not treat 200 GB as a guaranteed unlimited artifact archive. [2]

### 2025/2026 changes

The live pricing page does not provide a dated 2025/2026 change history. Therefore, the defensible 2026 statement is the current table above, not a claim that a particular quota changed in 2025 or 2026. The page also warns that locally hosted deployments do not use the cloud-storage limits in the same way. [2]

## B. W&B Sweeps: capabilities and operating model

### Search methods and configuration

The official guide supports `grid`, `random`, and `bayes` methods, and describes Bayesian Optimization and Hyperband/BOHB as supported sweep concepts. [1] A sweep configuration can be YAML or a Python dictionary and commonly contains `program`, `name`, `method`, `metric`, `parameters`, and optionally `command`. Parameter entries can use `value`, `values`, `distribution`, `min`, `max`, `q`, and nested `parameters`; the documented examples include `uniform` and `q_log_uniform_values`. [1]

A minimal PPO-style configuration is:

```yaml
program: train_ppo.py
method: bayes
metric:
  name: eval/mean_return
  goal: maximize
parameters:
  learning_rate:
    distribution: q_log_uniform_values
    min: 0.00001
    max: 0.003
  batch_size:
    values: [128, 256, 512]
  gamma:
    values:
  seed:
    values: [11, 23, 47]

early_terminate:
  type: hyperband
  min_iter: 5
  max_iter: 50
  s: 2
  eta: 3
```

The exact metric name must be emitted by the training process. In an RL sweep, use one scalar such as mean evaluation return at a fixed training budget, not a noisy per-worker rollout reward.

### Bayesian implementation

W&B's official documentation confirms the `bayes` method but does **not** specify the surrogate model, acquisition function, categorical encoding, conditional-search implementation, or a TPE implementation. [1] The practical consequence is that W&B is less transparent than Optuna when you need to reason about sampler behavior or reproduce a study at the algorithm level. Do not describe W&B's Bayesian method as TPE or Gaussian-process BO without a separate current W&B engineering source.

### Hyperband, agents, parallelism, and control

Hyperband uses `early_terminate` with `type: hyperband`, and the documentation shows `s`, `eta`, `max_iter`, and optionally `min_iter`. [1] In RL, `max_iter` should mean a comparable training unit, such as evaluation checkpoints or environment-step milestones, rather than wall-clock time that varies with GPU contention.

The documented Python pattern is `sweep_id = wandb.sweep(sweep=configuration, project=...)` followed by `wandb.agent(sweep_id, function=main, count=...)`. [1] The agent asks the sweep service for a configuration, starts one training run, and repeats until its count or the sweep's available work is exhausted. Each agent should normally create one W&B run per RL trial.

For multiple GPUs, the current documentation shows separate processes with GPU binding, for example:

```bash
CUDA_VISIBLE_DEVICES=0 wandb agent SWEEP-ID
CUDA_VISIBLE_DEVICES=1 wandb agent SWEEP-ID
```

It also documents multiple agents on multi-core or multi-GPU machines and the notebook equivalent `wandb.agent(...)`. [14] Across machines, launch agents with the same sweep ID and give each process distinct resources and a distinct seed. The service coordinates suggestions; the training scheduler, GPU allocation, container placement, and retry policy remain your responsibility.

The CLI supports pausing, resuming, stopping, and cancelling a sweep. Pausing preserves state while temporarily halting exploration; resume continues creating runs according to the search strategy; stop prevents new runs; cancel terminates the sweep lifecycle. The official pages reviewed do not publish a maximum number of runs, maximum agents, a fixed concurrency cap, or a sweep-duration quota. The absence of a documented limit is not evidence of unlimited service capacity.

## C. Logging performance, rate limits, and offline RL runs

| Item | Current documented position |
|---|---|
| Scope | Metric-logging rate limits apply at the project level and include request rate and total request size over a rolling window [10] |
| Numeric API quota | W&B does not publish the actual request-rate, request-size, or rolling-window quota on the current limits page; it says limits can change [10] |
| Performance guidance | Under 10,000 runs/project, under 500,000 steps/run, under 100,000 metric values/minute, under 1,000 log calls/minute, and under 40 MB/minute video are recommended operating targets, not hard limits [10] |
| Payload size | Keep one logged value below 1 MB and one `wandb.Run.log()` call below 25 MB; keep run config below 10 MB and files below 1,000 per run [10] |
| HTTP 429 | A 429 means the project quota was exceeded; the response includes `RateLimit-Limit`, `RateLimit-Remaining`, and `RateLimit-Reset` headers [10] |
| Recovery | The SDK retries with backoff; `run.finish()` can be delayed. Reduce frequency, batch metrics, update the SDK, or use offline logging and later sync. Wait for `RateLimit-Reset` when remaining quota is zero [10] |
| Offline | Use `WANDB_MODE=offline`; data is stored locally and uploaded with `wandb sync` or by returning to online mode [6][25] |

There is no current official numeric statement in the reviewed documentation for "runs per minute" or a universal `wandb.log()` calls-per-second quota. Older community discussions mention values such as 50 or 200 requests/minute, but they should not be presented as current contractual W&B limits because the current official page says the quotas are dynamic and does not publish those numbers.

Logging every environment step is usually a bad RL design. With `N` vectorized environments, one call per environment step can multiply the call rate by `N`, while most adjacent transitions are not useful for experiment-level diagnosis. Log aggregate rollout statistics, losses, entropy, explained variance, throughput, and evaluation return at rollout or evaluation intervals. Keep raw trajectories in files or artifacts only when needed. W&B's own guidance says overly frequent logging increases overhead and can make projects slower, and it explicitly recommends batching related metrics into fewer calls. [10]

Offline mode is useful on isolated clusters and for avoiding online throttling during high-throughput training. It also means dashboards, sweep-service decisions, and online visibility are delayed until synchronization. Treat offline sweeps carefully: let the sweep agent remain online if it needs service-side suggestions, or use an external search controller and upload each completed trial afterward.

## D. RL integrations and vectorized environments

### Stable-Baselines3

The W&B SB3 integration is `wandb.integration.sb3.WandbCallback`. Its source says that it logs SB3 experiments, adds model tracking and uploading, records complete hyperparameters, and supports gradient logging; `wandb.init(...)` must be called before the callback is used. [8] A current SB3 example also emphasizes tracking configs, metrics, and videos. [26]

For a sweep, initialize once inside each trial process:

```python
import wandb
from wandb.integration.sb3 import WandbCallback

run = wandb.init(project="ppo-sweep", config=trial_config)
model = PPO(..., env=vec_env, seed=run.config.seed)
model.learn(total_timesteps=..., callback=WandbCallback())
wandb.log({"eval/mean_return": mean_return, "eval/std_return": std_return})
run.finish()
```

The important RL rule is one W&B run per sweep trial, not one run per subprocess environment. `SubprocVecEnv` workers should not each call `wandb.init()`. The main learner process aggregates rollout data and reports one scalar objective at a defined evaluation checkpoint.

### CleanRL

CleanRL's official experiment-tracking page uses `--track`; adding `--capture_video` uploads recorded videos as well. The CLI example is `python cleanrl/ppo.py --track --capture_video`, and the page shows the resulting project and run URLs. [15] CleanRL's PPO documentation describes vectorized environments as a core performance feature. [27]

For sweeps, pass W&B's assigned configuration into the single-file script, preserve the script's global step and seed semantics, and do not confuse vectorized sample count with optimizer-update count. A trial objective should be computed from evaluation episodes, not the maximum reward observed in a training worker.

### PRIME-RL

PRIME-RL's current repository exposes a W&B monitor implementation and documents W&B integration for experiment tracking, training metrics, sample visualizations, and distribution statistics across orchestrator and evaluation processes. [28][29] This is a more centralized monitor pattern than a minimal SB3 callback: it is suited to asynchronous agentic RL, where training, rollout, and evaluation streams must be correlated. Because the raw GitHub file did not render through the live extractor, exact internal method names and logging intervals should be taken from the current repository source before treating them as API guarantees.

### Practical vectorized-sweep rules

1. One sweep trial owns one process-level W&B run.
2. Use one deterministic trial seed plus an explicit list of evaluation seeds.
3. Seed the environment workers deterministically, for example with `base_seed + worker_id`, while retaining the trial seed in `wandb.config`.
4. Aggregate vectorized rewards before logging; do not upload one metric per worker per transition.
5. Report a scalar objective such as mean return over fixed evaluation seeds at a fixed environment-step budget.
6. Use Hyperband only when partial learning curves are comparable across trials and early stopping is not biased against slower-learning configurations.

## E. W&B versus Optuna and Ray Tune

| Capability | W&B Sweeps | Optuna | Ray Tune |
|---|---|---|---|
| Search | Grid, random, Bayesian; exact Bayesian internals are not specified in the reviewed docs [1] | TPE and other samplers; define-by-run search spaces; `TPESampler` is shown in current examples [16] | Search-algorithm wrappers, including Optuna and HyperOpt; HyperOptSearch uses TPE [22][30] |
| Early stopping | Hyperband configuration with `s`, `eta`, `min_iter`, `max_iter` [1] | Pruners such as `MedianPruner`; objective reports intermediate values and raises `TrialPruned` [16] | ASHA early termination and PBT pause/clone/mutate/resume behavior [19] |
| Persistent study state | Managed by W&B service, but public docs reviewed do not specify an open local study database or exact limits | RDB storage and `GrpcStorageProxy` support multi-machine studies; MySQL and other RDB backends are documented [12] | Ray cluster execution and resource-aware trial scheduling |
| Dashboard | Strong experiment, run, metric, artifact, report, and collaboration UI | `optuna-dashboard` visualizes optimization history, importances, and relationships [16] | Ray dashboard plus Tune results and integrations |
| Distributed execution | Multiple agents can share a sweep ID; GPU binding is documented [14] | Threads/processes, RDB multi-node, and gRPC storage [12] | Designed for distributed resource scheduling and cluster-scale execution [9] |
| RL experiment layer | Strong run lineage, media, artifacts, model tracking, and collaborative analysis | Primarily an HPO engine; W&B callback integration exists [31] | Primarily a distributed execution and HPO scheduler |
| Best fit | Teams that value a hosted system of record and low-friction visualization | Fine-grained sampler/pruner control and a lightweight Python-native HPO core | Large clusters, heterogeneous resources, ASHA/PBT, and scheduling |
| Main weakness | Less transparent BO internals and less explicit HPO-state control; rate/storage economics matter | Requires adding a tracking/metadata layer for polished team experiment management | More infrastructure and operational complexity than a simple study or W&B sweep |

W&B adds a polished cross-trial experiment record: charts, reports, artifacts, model lineage, media, permissions, and a hosted UI that remains useful after the search algorithm has finished. It is especially valuable when comparing PPO/SAC implementations, videos, checkpoints, configs, and evaluation curves across many contributors.

W&B is worse when the central problem is HPO control rather than observability. Optuna exposes sampler and pruner objects, persistent RDB studies, intermediate-value pruning, and a dashboard. Ray Tune adds cluster-aware resource scheduling, ASHA, and population-based checkpoint transfer. W&B's official docs reviewed here do not document the internals of its Bayesian method or a fixed run/agent limit.

The common and sensible hybrid is **Optuna for search plus W&B for logging**. Optuna ships a `WeightsAndBiasesCallback`, and the integration documentation says it tracks suggested hyperparameters and optimized metrics in W&B. [31] An example creates an Optuna study, defines an objective, and passes W&B project arguments to the callback. [32] This pattern lets Optuna own trial selection and pruning while W&B owns experiment presentation and artifact lineage.

## F. Evidence-based PPO and SAC HPO guidance

### PPO: tune the implementation and rollout geometry first

Engstrom et al., **"Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO," arXiv:2005.12729**, studies PPO and TRPO and reports that code-level optimizations have a major effect, account for much of PPO's cumulative-reward advantage over TRPO, and fundamentally change agent behavior. [7] This means an HPO comparison is invalid if one trial silently changes normalization, advantage handling, initialization, clipping, batching, or evaluation protocol.

Andrychowicz et al., **"What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study," arXiv:2006.05990**, trained more than 250,000 agents and found that implementation-level choices can drive performance. [3][17] Their practical findings include strong support for observation normalization and checking value-function normalization, while gradient clipping is secondary; policy initialization can significantly influence results. [17]

A defensible PPO search order is therefore:

| Priority | PPO variables | Why |
|---:|---|---|
| 1 | Learning rate, rollout length, number of environments, minibatch size, update epochs | These define optimization noise, batch reuse, and the number of updates per sample budget |
| 2 | Clip range, entropy coefficient, value coefficient, max gradient norm, target KL or early-stop rule | These control policy update size, exploration, value loss balance, and update stability |
| 3 | Gamma, GAE lambda, advantage/value normalization, initialization, activation and network width | These affect credit assignment and conditioning; normalization and initialization have strong evidence [17] |
| 4 | Seed, evaluation frequency, total budget, environment preprocessing | These are essential experimental controls, not merely nuisance variables |

RL Baselines3 Zoo uses Optuna for tuning, applies a default budget of **500 trials**, and performs an intermediate evaluation every **100,000 time steps** for pruning or early stopping. Its current tuning guide also notes that PPO tuning sets `ortho_init = False`, unlike the SB3 default of `True`. [18] That is useful evidence that the search space should include implementation defaults, not just textbook PPO coefficients.

### SAC: prioritize temperature, critic learning, replay, and evaluation stability

For SAC, tune the learning rate and entropy-temperature behavior together with replay-buffer and update-ratio choices. The high-value variables are usually actor/critic learning rate, initial or automatic entropy coefficient, batch size, replay-buffer size, warm-up steps, train-frequency/update-frequency, discount factor, target smoothing coefficient, and network size. Keep action scaling, reward scaling, termination handling, and observation normalization fixed or explicitly searched because they can masquerade as algorithmic effects.

The 2024 sensitivity work **"A Method for Evaluating Hyperparameter Sensitivity in Reinforcement Learning," arXiv:2412.07165**, proposes metrics and graphical methods for quantifying sensitivity and demonstrates them on PPO normalization variants. It warns that apparent algorithmic improvements can partly reflect increased reliance on tuning. [4][33] Its direct evidence is stronger for PPO than SAC, so do not claim that it establishes a universal SAC ranking.

### Seeds, trial counts, and validation

There is no evidence-based universal number of Bayesian trials for every five-to-ten-dimensional RL problem. A useful engineering starting point is **50-100 trials** for a five- to ten-variable search when each trial has a meaningful early-stopping curve, followed by confirmation of the top few configurations across several independent seeds. This is a recommendation, not a published law. If trials are expensive and noisy, reduce the search space using the sensitivity evidence first; a broad 10-variable BO search over noisy one-seed returns can be less informative than a focused 5-variable search with replication.

Eimer, Lindauer, and Raileanu, **"Hyperparameters in Reinforcement Learning and How To Tune Them," arXiv:2306.01324**, explicitly recommends separating tuning and testing seeds and applying principled HPO over a broad search space. [23] The paper reports that the performance landscape can depend strongly on the tuning seed, that the same configuration can have large performance variance, and that a configuration selected on one seed can perform poorly on others. [11] In some cases, incumbent configurations were more than four times worse on test seeds, and the authors recommend separate tuning and testing seeds. [11]

For finance or any non-stationary environment, use nested or walk-forward validation:

1. Split chronologically into train/tuning, validation, and untouched test periods.
2. Run HPO only inside each training window.
3. Select the configuration using validation returns and risk-adjusted metrics.
4. Evaluate once on the next untouched period.
5. Roll the window forward and repeat; report the distribution of out-of-sample results.
6. Never choose a final configuration after looking repeatedly at the test set.

Walk-forward validation is specifically used to reduce look-ahead and backtest overfitting, while recent finance literature distinguishes ordinary cross-validation, walk-forward paths, and combinatorial purged cross-validation. [34][35] In RL, also freeze evaluation seeds within each comparison so that every candidate faces the same test randomness.

## Recommended architecture

For a serious PPO or SAC project, use one of these two designs:

- **Small to medium search:** Optuna `TPESampler` plus a pruner, one trial process per GPU, W&B callback or explicit logging, and one W&B run per trial. Store the Optuna study in RDB storage if trials span machines. [12][16][31]
- **Large cluster search:** Ray Tune with ASHA for comparable learning curves or PBT when checkpoint transfer and online hyperparameter mutation are appropriate; use W&B inside each trial for metrics, checkpoints, videos, and reports. [19][9]
- **W&B-only sweep:** Use `bayes`, `random`, or `grid`, Hyperband where partial curves are comparable, multiple `wandb agent` processes with explicit GPU binding, and a single evaluation scalar per trial. [1][14]

The practical decision is simple: choose W&B when the durable research record and collaborative analysis are the bottleneck; choose Optuna when sampler/pruner/study control is the bottleneck; choose Ray Tune when cluster scheduling, elastic resources, ASHA, or PBT is the bottleneck. In many RL teams the strongest answer is not either/or: **Optuna or Ray Tune selects trials, and W&B records them.**

## Source URLs

- W&B pricing: https://wandb.ai/site/pricing/
- W&B academic research: https://wandb.ai/site/research/
- Academic application: https://wandb.ai/create-team and https://wandb.ai/academic_application
- Student academic-plan instructions: https://docs.wandb.ai/support/models/articles/can-i-get-an-academic-plan-as-a-student
- W&B Sweeps overview: https://docs.wandb.ai/models/sweeps
- Sweep configuration: https://docs.wandb.ai/models/sweeps/define-sweep-configuration
- Parallel agents: https://docs.wandb.ai/models/sweeps/parallelize-agents
- Pause/resume/cancel: https://docs.wandb.ai/models/sweeps/pause-resume-and-cancel-sweeps
- Logging limits: https://docs.wandb.ai/models/track/limits
- Rate-limit recovery: https://docs.wandb.ai/support/models/articles/rate-limit-exceeded-on-metric-logging
- Offline mode: https://docs.wandb.ai/models/ref/cli/wandb-offline
- SB3 callback source: https://github.com/wandb/wandb/blob/master/wandb/integration/sb3/sb3.py
- CleanRL tracking: https://docs.cleanrl.dev/get-started/experiment-tracking/
- CleanRL PPO: https://docs.cleanrl.dev/rl-algorithms/ppo/
- PRIME-RL W&B monitor: https://github.com/PrimeIntellect-ai/prime-rl/blob/main/src/prime_rl/utils/monitor/wandb.py
- Optuna distributed optimization: https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html
- Optuna efficient algorithms: https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html
- Optuna visualization/dashboard: https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/005_visualization.html
- Optuna W&B callback: https://optuna-integration.readthedocs.io/en/stable/reference/generated/optuna_integration.WeightsAndBiasesCallback.html
- Ray Tune: https://docs.ray.io/en/latest/tune/index.html
- Ray search algorithms: https://docs.ray.io/en/latest/tune/api/suggestion.html
- Ray schedulers: https://docs.ray.io/en/latest/tune/api/schedulers.html
- RL Baselines3 Zoo tuning: https://rl-baselines3-zoo.readthedocs.io/en/master/guide/tuning.html
- PPO config: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
- SAC config: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/sac.yml
- Engstrom et al., arXiv:2005.12729: https://arxiv.org/abs/2005.12729
- Andrychowicz et al., arXiv:2006.05990: https://arxiv.org/abs/2006.05990
- Eimer et al., arXiv:2306.01324: https://arxiv.org/abs/2306.01324
- RL sensitivity study, arXiv:2412.07165: https://arxiv.org/abs/2412.07165
- Finance walk-forward overview: https://alpha-suite.org/blog/walk-forward-optimization

Note on evidence gaps: the live W&B pages reviewed do not publish numeric request quotas, tracked-hour limits for Free/Personal/Pro, retention duration for those plans, a 2025/2026 pricing changelog, a documented maximum sweep-run or agent count, or the internal Bayesian surrogate. Those fields are reported as not stated rather than filled with older or third-party numbers.

## References

1. *Overview*. https://docs.wandb.ai/models/sweeps/define-sweep-configuration
2. *Explore Weights & Biases pricing plans*. https://wandb.ai/site/pricing/
3. *What Matters for On-Policy Deep Actor-Critic Methods? A Large ...*. https://openreview.net/forum?id=nIAxjsniDzg
4. *A Method for Evaluating Hyperparameter Sensitivity in ...*. https://arxiv.org/html/2412.07165v1
5. *Callbacks — Stable Baselines3 2.9.2a0 documentation*. https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
6. *wandb offline - Weights & Biases Documentation*. https://docs.wandb.ai/models/ref/cli/wandb-offline
7. *Implementation Matters in Deep Policy Gradients ...*. https://arxiv.org/abs/2005.12729
8. *wandb/wandb/integration/sb3/sb3.py at main - GitHub*. https://github.com/wandb/wandb/blob/master/wandb/integration/sb3/sb3.py
9. *Ray Tune: Hyperparameter Tuning — Ray 2.58.0*. https://docs.ray.io/en/latest/tune/index.html
10. *Logging at scale and performance - Weights & Biases*. https://docs.wandb.ai/models/track/limits
11. *Hyperparameters in Reinforcement Learning and How To Tune Them*. https://proceedings.mlr.press/v202/eimer23a/eimer23a.pdf
12. *Easy Parallelization*. http://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html
13. *Sign In with Auth0*. https://wandb.ai/academic_application
14. *Parallelize agents*. https://docs.wandb.ai/models/sweeps/parallelize-agents
15. *Experiment tracking - CleanRL*. https://docs.cleanrl.dev/get-started/experiment-tracking/
16. *Quick Visualization for Hyperparameter Optimization Analysis*. https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/005_visualization.html
17. *What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study*. https://ar5iv.labs.arxiv.org/html/2006.05990
18. *Hyperparameter Tuning*. https://rl-baselines3-zoo.readthedocs.io/en/master/guide/tuning.html
19. *Tune Trial Schedulers (tune.schedulers)*. https://docs.ray.io/en/latest/tune/api/schedulers.html
20. *prime-rl/src/prime_rl/utils/monitor/wandb.py at main ... - GitHub*. https://github.com/PrimeIntellect-ai/prime-rl/blob/main/src/prime_rl/utils/monitor/wandb.py
21. *Search Algorithms | wandb/sweeps | DeepWiki*. https://deepwiki.com/wandb/sweeps/5-search-algorithms
22. *OptunaSearch — Ray 2.58.0*. https://docs.ray.io/en/latest/tune/api/doc/ray.tune.search.optuna.OptunaSearch.html
23. *Hyperparameters in Reinforcement Learning and ...*. https://arxiv.org/abs/2306.01324
24. *Manage sweeps - Weights & Biases Documentation*. https://docs.wandb.ai/models/sweeps/pause-resume-and-cancel-sweeps
25. *How do I run W&B offline? - Weights & Biases Documentation*. https://docs.wandb.ai/support/models/articles/how-do-i-run-wandb-offline
26. *W&B and SB3.ipynb - Colab - Google Colab*. https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/stable_baselines_wandb.ipynb
27. *CleanRL: Advanced PPO - PettingZoo Documentation*. https://pettingzoo.farama.org/tutorials/cleanrl/advanced_PPO/
28. *Weights & Biases Integration | PrimeIntellect-ai/prime-rl ...*. https://deepwiki.com/PrimeIntellect-ai/prime-rl/10.2-weights-and-biases-integration
29. *Monitoring and Logging | PrimeIntellect-ai/prime-rl | DeepWiki*. https://deepwiki.com/PrimeIntellect-ai/prime-rl/10-monitoring-and-logging
30. *HyperOptSearch — Ray 2.58.0*. https://docs.ray.io/en/latest/tune/api/doc/ray.tune.search.hyperopt.HyperOptSearch.html
31. *optuna_integration.WeightsAndBiasesCallback*. https://optuna-integration.readthedocs.io/en/stable/reference/generated/optuna_integration.WeightsAndBiasesCallback.html
32. *optuna.integration.wandb — Optuna 3.4.1 documentation*. https://optuna.readthedocs.io/en/v3.4.1/_modules/optuna/integration/wandb.html
33. *A Method for Evaluating Hyperparameter ...*. https://arxiv.org/abs/2412.07165
34. *Walk-Forward Optimization: Avoiding Backtest Bias | Alpha Suite*. https://alpha-suite.org/blog/walk-forward-optimization
35. *Backtest overfitting in the machine learning era: A ...*. https://www.sciencedirect.com/science/article/pii/S0950705124011110
