# Prime Intellect Developer Platform: Live Docs and Pricing

**Fetch date: 2026-09-14.** I checked the live documentation index at `https://docs.primeintellect.ai/llms.txt`, the linked `.md` pages, the public compute dashboard, the Prime website, and the two GitHub repositories. Prices and stock are dynamic. Where the live pages did not expose a requested fact, I mark it **not publicly verified** rather than filling the gap with a third-party estimate.

## Executive summary

- **CLI coverage**: The official CLI documents install with `uv tool install prime` or `pip install prime`, support `prime login`, API-key configuration, SSH-key-path configuration, availability queries, pod creation, pod listing, and pod SSH. [17] [17]
- **Lifecycle gap**: The official CLI introduction did not document a status or terminate subcommand. The API documents do expose `GET /api/v1/pods/status` and `DELETE /api/v1/pods/{pod_id}`. [10] [13]
- **Live GPU floor**: The cheapest displayed GPU offer was an A2000 at **$0.15/hour**, with stock shown as 1. [22]
- **Dynamic marketplace**: The create-cluster page says prices update in realtime and exposes Community and Secure rates, but it does not define those labels, state a minimum billing increment, or identify every displayed rate as hourly. [22]
- **Spot**: The official FAQ describes Spot as interruptible unused capacity with discounts of up to 90%; the marketing page showed an H100 spot card at **$0.94/hour** versus **$2.43/hour** displayed on that page for the corresponding H100 card. [23] [20]
- **CPU-only**: I found no public CPU-only instance option or CPU-only rate. The cheapest verified public offer is therefore the **$0.15/hour A2000 GPU**, not a CPU instance. [22] [20]
- **Storage**: Persistent disks survive instance or cluster termination and can be reattached or shared, while ephemeral shared storage is deleted with the cluster. The live docs did not publish a persistent-disk price per GB-month. [11] [16] [16]
- **RL scope**: `prime-rl` is an asynchronous, agentic, language-model RL system designed around FSDP2 training and vLLM inference, not a general-purpose Gymnasium/PPO library. [2]
- **Algorithms**: Current `prime-rl` documentation explicitly lists GRPO, MaxRL, RAE, hierarchical GRPO, OPD, SFT, OPSD, and ECHO. [24]
- **Environments Hub**: The public `verifiers` description is explicitly for creating environments to train and evaluate LLMs. The Hub is moving to verifiers v1; the older v0 creation workflow is marked Legacy/deprecated. [3] [25]
- **Commercial products**: Hosted Training, Inference, and Sandboxes are publicly described, but I could not verify current public price sheets for Hosted Training or Inference. Sandbox resource rates are documented separately. [20] [26] [26] [26]

## 1. `prime` CLI: installation, authentication, compute, and SSH

### Installation and authentication

The official CLI introduction quotes these installation commands:

```bash
# uv
uv tool install prime

# pip
pip install prime
```

[17] [17]

The documented authentication paths are:

```bash
prime login
prime config set-api-key
export PRIME_API_KEY="your-api-key-here"
```

`prime login` and the CLI configuration commands are in the official CLI introduction. The repository README also shows the `PRIME_API_KEY` environment-variable pattern. [17] [1]

To create an API key, the live API-key page sends the user to **Settings -> API Keys** at `https://app.primeintellect.ai/dashboard/tokens`, then to **Generate New Key +**. The key is displayed only once, and the page recommends an expiration date. [27]

### Configuration and SSH setup

The documented configuration commands are:

```bash
prime config view
prime config set-api-key
prime config set-team-id
prime config remove-team-id
prime config set-share-resources-with-team true
prime config set-share-resources-with-team false
prime config set-base-url
prime config set-ssh-key-path
prime config reset
```

[18]

The practical SSH setup step is therefore to have a local private key and configure its path:

```bash
prime config set-ssh-key-path
```

The live docs do not publish a `ssh-keygen` example. They do document the API upload shape, which requires a key `name` and `publicKey`, and the API endpoint is `POST /api/v1/ssh_keys/`. [28] The SSH-key listing endpoint is `GET /api/v1/ssh_keys/`. [29]

### GPU availability and pod commands

The official CLI syntax is:

```bash
# All currently available GPU configurations
prime availability list

# Persistent-disk availability
prime availability disks

# Filter by GPU model
prime availability list --gpu-type H100_80GB

# Other documented filters include:
# --gpu-count 2
# --regions <region>

# Create and list pods
prime pods create
prime pods list

# SSH to a running pod
prime pods ssh <pod-id>
```

The availability page says `prime availability list` returns GPU type, count, location, hourly price, and stock status, and that `prime availability disks` returns persistent-storage configurations, pricing, capacity, and multinode support. [15] The CLI introduction supplies the pod command forms above. [17]

### Status and termination: do not assume undocumented aliases

I could not verify official CLI syntax for either `prime pods status` or `prime pods terminate` in the live CLI pages. The official API pages instead expose:

```text
GET    https://api.primeintellect.ai/api/v1/pods/status
DELETE https://api.primeintellect.ai/api/v1/pods/{pod_id}
```

[10] [13]

Therefore, `prime pods status` and `prime pods terminate` may exist in a newer CLI help build or another CLI package, but I am not presenting either as verified live documentation. The safest verified fallback is to inspect the installed CLI help and use the documented API endpoints if necessary:

```bash
prime pods --help
prime pods create --help
prime availability list --help
```

These help commands are operational suggestions, not quoted commands from the current docs.

## 2. Current GPU prices and availability

The table below transcribes the public create-cluster page fetched on **2026-09-14**. The page was showing one available unit for the listed entries. It says prices update in realtime. [22] [22]

| GPU | VRAM | Displayed availability | Community rate | Secure rate | Notes |
|---|---:|---:|---:|---:|---|
| RTX 4090 | 24 GB | 1 | **$0.37** | **$0.72** | Live create-cluster offer [22] |
| A10 | Not displayed | Not displayed | Not displayed | Not displayed | No A10 offer was visible on the fetched page |
| L4 | 24 GB | 1 | Not displayed | **$0.46** | [22] |
| L40S | 48 GB | 1 | Not displayed | **$1.00** | [22] |
| A100 40 GB | 40 GB | 1 | Not displayed | **$1.45** | [22] |
| A100 80 GB | 80 GB | 1 | **$1.22** | **$1.35** | [22] |
| H100 | 80 GB | 1 | **$2.72** | **$1.90** | [22] |
| H200 | 141 GB | 1 | Not displayed | **$3.65** | [22] |
| B200 | Not displayed on create-cluster table | Not displayed | Not displayed | Not displayed | Prime's public marketing page showed **$3.49/hour**, without Community/Secure breakdown [20] |

The displayed Community and Secure columns are not the same thing as the documented Spot product. The create-cluster page labels the two security/marketplace categories but does not, in the fetched text, define their SLA, interruption behavior, or whether either is Spot. It also does not state a minimum billing increment. [22]

The official FAQ separately defines Spot as unused-capacity virtual machines that can be interrupted when capacity is needed, with discounts of up to 90%. [23] The public marketing page displayed this H100 comparison:

| Marketing-page card | Rate |
|---|---:|
| H100 displayed rate | $2.43/hour |
| H100 Spot displayed rate | $0.94/hour |

[20]

**Billing increment:** I could not verify an official minimum billing increment from the live docs or public dashboard. The dashboard is explicitly realtime and marketplace-like, so the table should be treated as a point-in-time offer snapshot, not a fixed tariff. [22]

## 3. CPU-only instances and cheapest offer

No CPU-only instance or CPU-only price was visible in the public compute pages I fetched. The public compute section advertises access to 1-256 GPUs and displays GPU offers, not CPU-only machines. [20] [20]

The cheapest verified displayed compute offer was:

```text
NVIDIA A2000, 6 GB, stock 1, Community $0.15/hour
```

[22]

This is the cheapest **visible GPU offer**, not proof that it is the cheapest possible marketplace offer at every location or time. A10 was not visible in the fetched create-cluster inventory.

## 4. Storage

### Persistent disks

Prime's storage docs describe persistent storage as data volumes that survive the lifecycle of individual instances and can be attached to different instances. [11] The same disk can be attached to multiple instances simultaneously. [11]

For clusters, persistent storage:

- persists after cluster termination;
- is shareable across multiple clusters;
- requires a pre-created disk with `Active` status;
- is selected through **Add Shared Filesystem**; and
- must match the provider and location of the compute resource.

[16] [11] [16]

### Ephemeral shared storage

The cluster guide distinguishes ephemeral shared storage from persistent disks. Ephemeral storage is created for the cluster lifetime, is shared by the nodes in that cluster, and is deleted when the cluster terminates. [16] [16]

### Pricing and object storage

The current storage pages I fetched did **not** publish a persistent-disk price per GB-month, a minimum disk charge, or a retention fee. The CLI availability page says disk availability output includes pricing, so the most reliable current price source appears to be the authenticated or live availability response rather than the static tutorial. [15]

I also did not verify a separate S3-compatible object-storage product or a general-purpose shared filesystem independent of the provider/location-specific persistent-disk feature. The documented shared-filesystem feature is the disk attachment mechanism described above, not evidence of an object-storage service.

## 5. `prime-rl`: scope, algorithms, hardware, and small-policy RL

`prime-rl` describes itself as **"Fully asynchronous RL for high-throughput agentic training at scale."** Its stated scale target is training 1T+ mixture-of-experts models on 1,000+ GPUs, with **FSDP2 for training** and **vLLM for inference**, plus FP8 inference, disaggregated prefilling/decode, expert parallelism, and context parallelism. [2]

The current algorithms documentation says the algorithm is configured under `[orchestrator.algo]` and that `type` names the algorithm. [24] It explicitly lists:

| Algorithm type | Role in the current docs |
|---|---|
| `grpo` | Standard group-relative RL [24] |
| `max_rl` | GRPO-like centered reward normalized by the group mean [24] |
| `rae` | Self-play-oriented reward baseline [24] |
| `hierarchical_grpo` | GRPO for proposer-solver environments [24] |
| `opd` | On-policy distillation using a reference model [24] |
| `sft` | Frozen-model hard distillation [24] |
| `opsd` | Self-distillation/SDFT [24] |
| `echo` | GRPO plus weighted cross-entropy on selected environment-provided tokens [24] |

Thus, **GRPO is explicitly supported**. The current official algorithm page does not present a conventional standalone continuous-control PPO trainer. Its PPO-like policy optimization vocabulary is embedded in language-model/token-loss and rollout machinery, while the documented model references are a trainable policy or externally hosted OpenAI-compatible language models. [24]

### Does it require multiple GPUs, vLLM, and FSDP?

The architecture is built for distributed LLM RL and explicitly centers FSDP2 and vLLM at large scale. That does not mean every small experiment mathematically requires multiple GPUs, but it does mean the project is not documented as a lightweight single-process MLP RL library. [2]

### Can it train an MLP PPO trading policy?

Not as a documented drop-in path. A small MLP policy learning from a Gymnasium-style trading environment would normally use a general RL implementation such as PPO with vectorized observations and continuous/discrete action handling. The current `prime-rl` documentation instead assumes language-model policies, token-level losses, model references, rollouts, and verifier-style environments. [24] [24]

A trading project could theoretically build a substantial adapter around Prime's orchestration and reward concepts, but that would be custom engineering. I found no live documentation showing that `prime-rl` directly accepts a Gymnasium environment, trains an arbitrary MLP actor/critic, or provides a standard PPO API. Treating it as a ready-made MLP PPO trainer would therefore be unsupported.

## 6. Environments Hub and `verifiers`

The official repository description is explicit: **"verifiers is our library for creating environments to train and evaluate LLMs."** It is integrated with the Environments Hub, `prime-rl`, and Hosted Training. [3]

The live Hub documentation says the Hub is moving to **verifiers v1**. Environments built on the v0 API receive a Legacy tag, and the older create page explicitly says its workflow is v0 and deprecated. [25] [30]

The live v0 page exposes the older workflow names `load_environment` and `vf-eval`; it does not provide a verified current v1 `prime env push` command in the fetched text. [30]

### Is `prime env push` verified?

The requested command:

```bash
prime env push
```

was **not verifiable in the current official page text I fetched**. The official CLI introduction documents compute, availability, pods, and SSH commands, while the environment page returned the deprecated v0 workflow. [17] [30]

Therefore, use the current CLI help before relying on that syntax:

```bash
prime env --help
prime env push --help
```

Those are prudent discovery commands, not a claim that the current docs prove the subcommand exists.

### Are these Gymnasium environments?

The public framework is fundamentally an **LLM/text and tool-use environment framework**, not a Gymnasium continuous-control framework. The repository description itself limits the stated purpose to training and evaluating LLMs. [3]

A trading task could fit only if reformulated as an LLM agent task, for example an agent receiving market observations as serialized text, calling tools, and receiving a verifier/rubric score. A normal Gymnasium environment with numeric state vectors, continuous actions, episode resets, and an MLP policy is outside the documented model. I found no official adapter demonstrating direct Gymnasium interoperability.

This distinction matters: an LLM environment may use datasets, parsers, tool calls, multi-turn trajectories, and rubric/verifier scoring, while a continuous-control Gym environment exposes a numeric observation/action API and usually computes rewards at every simulator step. The live Prime material verifies the former, not the latter.

## 7. Prime Inference, Hosted Training, and Sandboxes

### Hosted Training

Prime describes Hosted Training as managed reinforcement-learning training on its infrastructure, with no GPUs for the user to manage. [8] The public website describes large-scale RL training, 2,500+ RL environments, managed workflows, visibility and control, and applied-research support. [20]

I could not verify a current public dollar price for Hosted Training. The documentation has a `Models & Pricing` page, but the live material fetched here did not expose a stable compute-rate table. Treat pricing as account-, model-, or quote-dependent unless the dashboard supplies a current authenticated rate.

### Prime Inference

Prime Inference provides access to state-of-the-art language models through a models endpoint. [31] The website describes dedicated or serverless inference, native LoRA support, one-click deployment for fine-tuned models, and serving LoRA adapters alongside base models. [20]

I could not verify current token rates from the public live pages used here. Do not substitute third-party price aggregators for the official current rate card.

### Sandboxes

Prime Sandboxes are disposable, isolated environments for AI-assisted coding, benchmarking, and experiments. The documentation describes them as VM sandboxes, billed while running. [26] [26]

The documented resource rates are:

| Sandbox resource | Rate |
|---|---:|
| CPU | **$0.05 per core-hour** [26] |
| Memory | **$0.01 per GB-hour** [26] |
| Disk | **$0.001 per GB-hour** [26] |

The docs separately state that GPU-enabled sandboxes are coming soon and require an explicit grant, so these rates should not be read as GPU-sandbox pricing. [26]

## 8. Credits, research programs, and academic access

I did not find a current official public page promising universal free credits, a standing academic discount, or automatic student access. The live site does promote open research, environments, and applied-research support, but that is not the same as a published free-credit entitlement. [20]

There is an external report of a **Fast Compute Grants** program offering compute credits to open-source or decentralized-AI researchers, but I could not verify current eligibility, availability, or terms on a live official Prime page during this check. Treat it as a lead to contact Prime about, not as guaranteed academic access. The relevant official-origin announcement link found during research was:

`https://x.com/PrimeIntellect/status/1786386588726960167`

The public open-environments program also describes grants and partnerships, but current application terms and amounts were not verified in the live pages used here. Contact Prime through its current website or Discord for an up-to-date grant decision.

## Synthesis and practical recommendation

Prime Intellect is strongest when the workload is one of three things: dynamic GPU procurement, managed LLM post-training, or verifier-based LLM agent RL. The CLI and dashboard are useful for discovering live offers, but the public documentation is uneven: compute availability and pod creation are documented, while the exact CLI status/termination syntax, persistent-disk tariff, and billing increment are not.

For a language-model RL project, `prime-rl` and `verifiers` are a coherent stack: verifiers defines LLM-oriented tasks and scoring, while prime-rl supplies asynchronous distributed training and algorithms such as GRPO. [3] [24] For a conventional MLP PPO trading agent, use a Gymnasium-compatible RL stack instead, or plan a custom integration rather than assuming Prime's Environments Hub is a Gym registry.

For cost planning, use `prime availability list` immediately before provisioning, because the dashboard explicitly updates prices in realtime. [15] [22] The **$0.15/hour A2000**, **$0.37/hour Community RTX 4090**, and **$1.90/hour Secure H100** figures are point-in-time displayed offers, not guaranteed prices. [22] [22] Keep checkpoints on persistent disks, not ephemeral shared storage, because the latter is deleted at cluster termination. [16]

## References

1. *GitHub - PrimeIntellect-ai/prime: Official CLI and Python SDK for Prime Intellect - access GPU compute, remote sandboxes, RL environments, and distributed training infrastructure for AI development at scale. · GitHub*. http://github.com/PrimeIntellect-ai/prime
2. *GitHub - PrimeIntellect-ai/prime-rl: Agentic RL Training at ...*. https://github.com/PrimeIntellect-ai/prime-rl
3. *GitHub - PrimeIntellect-ai/verifiers: Our library for RL ...*. https://github.com/primeintellect-ai/verifiers
4. *docs.primeintellect.ai*. https://docs.primeintellect.ai/llms.txt
5. *On-Demand GPUs | Prime Intellect*. https://app.primeintellect.ai/dashboard/on-demand-gpus
6. *Introduction - Prime Intellect Docs*. https://docs.primeintellect.ai
7. *Create Pod*. https://docs.primeintellect.ai/api-reference/pods/create-pod.md
8. *Getting Started*. https://docs.primeintellect.ai/hosted-training/getting-started.md
9. *Inference Overview*. https://docs.primeintellect.ai/inference/overview.md
10. *Get Pods Status*. https://docs.primeintellect.ai/api-reference/pods/get-pods-status.md
11. *Use persistent storage with instances*. https://docs.primeintellect.ai/tutorials-storage/use-persistent-storage-with-instances.md
12. *Create persistent storage*. https://docs.primeintellect.ai/tutorials-storage/create-persistent-storage.md
13. *Delete Pod*. https://docs.primeintellect.ai/api-reference/pods/delete-pod.md
14. *Advanced Usage*. https://docs.primeintellect.ai/inference/usage.md
15. *Get Availability Information*. https://docs.primeintellect.ai/cli-reference/check-gpu-availability.md
16. *Cluster storage*. https://docs.primeintellect.ai/tutorials-storage/cluster-storage.md
17. *Overview*. https://docs.primeintellect.ai/cli-reference/introduction.md
18. *Configuration*. https://docs.primeintellect.ai/cli-reference/config-cli.md
19. *Sandboxes Overview*. https://docs.primeintellect.ai/sandboxes/overview.md
20. *Prime Intellect - The Open Superintelligence Stack*. https://primeintellect.ai/
21. *Models & Pricing*. https://docs.primeintellect.ai/hosted-training/models-and-pricing.md
22. *PrimeIntellect Compute*. https://app.primeintellect.ai/dashboard/create-cluster
23. *FAQ - Prime Intellect Docs*. https://docs.primeintellect.ai/faq
24. *Algorithms*. https://docs.primeintellect.ai/prime-rl/algorithms.md
25. *Overview*. https://docs.primeintellect.ai/tutorials-environments/environments.md
26. *https://docs.primeintellect.ai/sandboxes/overview*. https://docs.primeintellect.ai/sandboxes/overview
27. *API keys*. https://docs.primeintellect.ai/api-reference/api-keys.md
28. *Upload Ssh Key*. https://docs.primeintellect.ai/api-reference/ssh-keys/upload-ssh-key.md
29. *Get Ssh Keys*. https://docs.primeintellect.ai/api-reference/ssh-keys/get-ssh-keys.md
30. *Create & Upload Environment*. https://docs.primeintellect.ai/tutorials-environments/create.md
31. *Inference Overview - Prime Intellect Docs*. https://docs.primeintellect.ai/inference/overview
