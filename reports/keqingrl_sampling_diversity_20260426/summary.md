# KeqingRL Sampling Diversity Probe

source_type: `checkpoint`
training: `false`
candidate_summary: `reports/keqingrl_discard_research_20260425_candidates_checkpoint_rerun/summary.csv`
episodes: `32`
temperatures: `1.0,1.25,1.5,2.0,3.0`
learner_seats: `0`
seed_registry_id: `base=202604260000:stride=1:count=32`
seed_hash: `cd7db370d379c462`

## Results

- source=93 rerun=0 opponent=rulebase temp=1 batch=493 selected_top1=0.675456 non_top1=160 non_top1_pos_adv=92 pos_top1=193 pos_non_top1=92 prior_prob_mean=0.456694 rank_hist={"1": 333, "10": 11, "11": 12, "12": 5, "13": 1, "14": 2, "2": 15, "3": 16, "4": 18, "5": 12, "6": 16, "7": 25, "8": 17, "9": 10}
- source=93 rerun=0 opponent=rulebase temp=1.25 batch=487 selected_top1=0.5154 non_top1=236 non_top1_pos_adv=120 pos_top1=135 pos_non_top1=120 prior_prob_mean=0.356087 rank_hist={"1": 251, "10": 19, "11": 16, "12": 13, "13": 4, "14": 2, "2": 18, "3": 23, "4": 28, "5": 24, "6": 18, "7": 22, "8": 26, "9": 23}
- source=93 rerun=0 opponent=rulebase temp=1.5 batch=486 selected_top1=0.401235 non_top1=291 non_top1_pos_adv=167 pos_top1=123 pos_non_top1=167 prior_prob_mean=0.283467 rank_hist={"1": 195, "10": 18, "11": 22, "12": 12, "13": 4, "14": 1, "2": 35, "3": 26, "4": 29, "5": 35, "6": 26, "7": 24, "8": 30, "9": 29}
- source=93 rerun=0 opponent=rulebase temp=2 batch=489 selected_top1=0.312883 non_top1=336 non_top1_pos_adv=176 pos_top1=87 pos_non_top1=176 prior_prob_mean=0.228229 rank_hist={"1": 153, "10": 21, "11": 34, "12": 21, "13": 4, "14": 1, "2": 29, "3": 37, "4": 23, "5": 31, "6": 37, "7": 30, "8": 36, "9": 32}
- source=93 rerun=0 opponent=rulebase temp=3 batch=498 selected_top1=0.200803 non_top1=398 non_top1_pos_adv=196 pos_top1=56 pos_non_top1=196 prior_prob_mean=0.15806 rank_hist={"1": 100, "10": 40, "11": 28, "12": 24, "13": 9, "14": 1, "2": 41, "3": 36, "4": 38, "5": 43, "6": 39, "7": 32, "8": 24, "9": 43}
- source=57 rerun=1 opponent=rule_prior_greedy temp=1 batch=490 selected_top1=0.663265 non_top1=165 non_top1_pos_adv=72 pos_top1=163 pos_non_top1=72 prior_prob_mean=0.453659 rank_hist={"1": 325, "10": 14, "11": 15, "12": 8, "13": 2, "2": 17, "3": 14, "4": 20, "5": 15, "6": 13, "7": 22, "8": 13, "9": 12}
- source=57 rerun=1 opponent=rule_prior_greedy temp=1.25 batch=478 selected_top1=0.516736 non_top1=231 non_top1_pos_adv=146 pos_top1=143 pos_non_top1=146 prior_prob_mean=0.356314 rank_hist={"1": 247, "10": 23, "11": 17, "12": 17, "13": 2, "2": 19, "3": 24, "4": 22, "5": 25, "6": 21, "7": 27, "8": 18, "9": 16}
- source=57 rerun=1 opponent=rule_prior_greedy temp=1.5 batch=491 selected_top1=0.415479 non_top1=287 non_top1_pos_adv=134 pos_top1=109 pos_non_top1=134 prior_prob_mean=0.294342 rank_hist={"1": 204, "10": 35, "11": 30, "12": 8, "13": 14, "2": 37, "3": 20, "4": 30, "5": 23, "6": 26, "7": 21, "8": 22, "9": 21}
- source=57 rerun=1 opponent=rule_prior_greedy temp=2 batch=480 selected_top1=0.322917 non_top1=325 non_top1_pos_adv=173 pos_top1=75 pos_non_top1=173 prior_prob_mean=0.233213 rank_hist={"1": 155, "10": 23, "11": 28, "12": 17, "13": 7, "14": 1, "2": 34, "3": 30, "4": 33, "5": 36, "6": 20, "7": 25, "8": 36, "9": 35}
- source=57 rerun=1 opponent=rule_prior_greedy temp=3 batch=501 selected_top1=0.187625 non_top1=407 non_top1_pos_adv=233 pos_top1=51 pos_non_top1=233 prior_prob_mean=0.148486 rank_hist={"1": 94, "10": 39, "11": 37, "12": 28, "13": 10, "14": 3, "2": 38, "3": 39, "4": 42, "5": 31, "6": 30, "7": 38, "8": 32, "9": 40}
- source=8 rerun=3 opponent=rule_prior_greedy temp=1 batch=500 selected_top1=0.64 non_top1=180 non_top1_pos_adv=109 pos_top1=212 pos_non_top1=109 prior_prob_mean=0.438484 rank_hist={"1": 320, "10": 15, "11": 17, "12": 16, "13": 4, "2": 16, "3": 21, "4": 19, "5": 15, "6": 17, "7": 14, "8": 13, "9": 13}
- source=8 rerun=3 opponent=rule_prior_greedy temp=1.25 batch=491 selected_top1=0.505092 non_top1=243 non_top1_pos_adv=140 pos_top1=149 pos_non_top1=140 prior_prob_mean=0.349787 rank_hist={"1": 248, "10": 26, "11": 16, "12": 13, "13": 3, "14": 2, "2": 23, "3": 25, "4": 26, "5": 23, "6": 19, "7": 20, "8": 19, "9": 28}
- source=8 rerun=3 opponent=rule_prior_greedy temp=1.5 batch=477 selected_top1=0.421384 non_top1=276 non_top1_pos_adv=173 pos_top1=106 pos_non_top1=173 prior_prob_mean=0.296915 rank_hist={"1": 201, "10": 32, "11": 25, "12": 12, "13": 9, "2": 16, "3": 34, "4": 24, "5": 24, "6": 26, "7": 23, "8": 21, "9": 30}
- source=8 rerun=3 opponent=rule_prior_greedy temp=2 batch=499 selected_top1=0.306613 non_top1=346 non_top1_pos_adv=185 pos_top1=84 pos_non_top1=185 prior_prob_mean=0.224479 rank_hist={"1": 153, "10": 32, "11": 30, "12": 17, "13": 5, "14": 2, "2": 36, "3": 24, "4": 35, "5": 24, "6": 28, "7": 34, "8": 32, "9": 47}
- source=8 rerun=3 opponent=rule_prior_greedy temp=3 batch=496 selected_top1=0.211694 non_top1=391 non_top1_pos_adv=190 pos_top1=53 pos_non_top1=190 prior_prob_mean=0.164955 rank_hist={"1": 105, "10": 33, "11": 24, "12": 24, "13": 3, "2": 22, "3": 40, "4": 30, "5": 48, "6": 45, "7": 46, "8": 47, "9": 29}

## Artifacts

- `sampling_diversity.json`
- `summary.csv`
- `batch_steps.csv`
