# KeqingRL Sampling Diversity Probe

source_type: `checkpoint`
training: `false`
candidate_summary: `reports/keqingrl_discard_research_20260425_candidates_checkpoint_rerun/summary.csv`
episodes: `16`
temperatures: `1.0,1.5,2.0,3.0`
learner_seats: `0`
seed_registry_id: `base=202604260000:stride=1:count=16`
seed_hash: `1a118258d3dc9335`

## Results

- source=93 rerun=0 opponent=rulebase temp=1 batch=954 selected_top1=0.919287 non_top1=77 non_top1_pos_adv=41 pos_top1=436 pos_non_top1=41 prior_prob_mean=0.612935 rank_hist={"1": 877, "10": 7, "11": 4, "12": 2, "13": 1, "14": 1, "2": 10, "3": 12, "4": 12, "5": 3, "6": 5, "7": 8, "8": 9, "9": 3}
- source=93 rerun=0 opponent=rulebase temp=1.5 batch=967 selected_top1=0.853154 non_top1=142 non_top1_pos_adv=72 pos_top1=413 pos_non_top1=72 prior_prob_mean=0.571299 rank_hist={"1": 825, "10": 13, "11": 13, "12": 6, "13": 1, "14": 2, "2": 9, "3": 13, "4": 19, "5": 14, "6": 10, "7": 18, "8": 11, "9": 13}
- source=93 rerun=0 opponent=rulebase temp=2 batch=936 selected_top1=0.821581 non_top1=167 non_top1_pos_adv=124 pos_top1=345 pos_non_top1=124 prior_prob_mean=0.551315 rank_hist={"1": 769, "10": 12, "11": 11, "12": 8, "13": 2, "2": 19, "3": 20, "4": 15, "5": 13, "6": 17, "7": 14, "8": 19, "9": 17}
- source=93 rerun=0 opponent=rulebase temp=3 batch=901 selected_top1=0.791343 non_top1=188 non_top1_pos_adv=105 pos_top1=347 pos_non_top1=105 prior_prob_mean=0.531246 rank_hist={"1": 713, "10": 11, "11": 19, "12": 10, "13": 6, "14": 1, "2": 11, "3": 25, "4": 14, "5": 17, "6": 19, "7": 14, "8": 22, "9": 19}
- source=57 rerun=1 opponent=rule_prior_greedy temp=1 batch=920 selected_top1=0.907609 non_top1=85 non_top1_pos_adv=39 pos_top1=421 pos_non_top1=39 prior_prob_mean=0.606214 rank_hist={"1": 835, "10": 8, "11": 8, "12": 4, "13": 2, "2": 7, "3": 9, "4": 8, "5": 8, "6": 6, "7": 12, "8": 5, "9": 8}
- source=57 rerun=1 opponent=rule_prior_greedy temp=1.5 batch=879 selected_top1=0.847554 non_top1=134 non_top1_pos_adv=70 pos_top1=369 pos_non_top1=70 prior_prob_mean=0.566151 rank_hist={"1": 745, "10": 14, "11": 9, "12": 6, "13": 3, "14": 1, "2": 11, "3": 11, "4": 18, "5": 9, "6": 15, "7": 8, "8": 15, "9": 14}
- source=57 rerun=1 opponent=rule_prior_greedy temp=2 batch=995 selected_top1=0.827136 non_top1=172 non_top1_pos_adv=115 pos_top1=385 pos_non_top1=115 prior_prob_mean=0.555113 rank_hist={"1": 823, "10": 22, "11": 12, "12": 13, "13": 3, "14": 1, "2": 22, "3": 11, "4": 16, "5": 15, "6": 17, "7": 20, "8": 13, "9": 7}
- source=57 rerun=1 opponent=rule_prior_greedy temp=3 batch=892 selected_top1=0.793722 non_top1=184 non_top1_pos_adv=83 pos_top1=361 pos_non_top1=83 prior_prob_mean=0.534338 rank_hist={"1": 708, "10": 12, "11": 18, "12": 9, "13": 4, "14": 1, "2": 19, "3": 17, "4": 18, "5": 18, "6": 10, "7": 17, "8": 20, "9": 21}
- source=8 rerun=3 opponent=rule_prior_greedy temp=1 batch=972 selected_top1=0.914609 non_top1=83 non_top1_pos_adv=50 pos_top1=437 pos_non_top1=50 prior_prob_mean=0.612309 rank_hist={"1": 889, "10": 10, "11": 7, "12": 5, "13": 3, "2": 7, "3": 11, "4": 11, "5": 7, "6": 4, "7": 7, "8": 6, "9": 5}
- source=8 rerun=3 opponent=rule_prior_greedy temp=1.5 batch=973 selected_top1=0.833505 non_top1=162 non_top1_pos_adv=111 pos_top1=380 pos_non_top1=111 prior_prob_mean=0.558649 rank_hist={"1": 811, "10": 14, "11": 10, "12": 9, "13": 2, "14": 2, "2": 18, "3": 8, "4": 17, "5": 17, "6": 19, "7": 12, "8": 18, "9": 16}
- source=8 rerun=3 opponent=rule_prior_greedy temp=2 batch=958 selected_top1=0.816284 non_top1=176 non_top1_pos_adv=86 pos_top1=392 pos_non_top1=86 prior_prob_mean=0.547883 rank_hist={"1": 782, "10": 14, "11": 13, "12": 10, "13": 7, "2": 14, "3": 19, "4": 17, "5": 21, "6": 15, "7": 15, "8": 11, "9": 20}
- source=8 rerun=3 opponent=rule_prior_greedy temp=3 batch=916 selected_top1=0.789301 non_top1=193 non_top1_pos_adv=87 pos_top1=369 pos_non_top1=87 prior_prob_mean=0.530615 rank_hist={"1": 723, "10": 13, "11": 17, "12": 14, "13": 4, "2": 19, "3": 11, "4": 16, "5": 17, "6": 14, "7": 18, "8": 26, "9": 24}

## Artifacts

- `sampling_diversity.json`
- `summary.csv`
- `batch_steps.csv`
