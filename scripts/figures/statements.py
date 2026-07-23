import pandas as pd
from belief_update_sim.normalization import NEGATIVE_FORMULATIONS


def normalize_data(df):
    # print(df[["persona_id", "statement_formulation", "llm_new_belief", "llm_general_public_stance"]].head(20))
    negative_mask = df["statement_formulation"].isin(NEGATIVE_FORMULATIONS)
    cols_to_flip = ["llm_new_belief", "llm_general_public_stance"]
    df.loc[negative_mask, cols_to_flip] *= -1
    df["positive"] = ~negative_mask
    # print(df[["persona_id", "statement_formulation", "llm_new_belief", "llm_general_public_stance"]].head(20))
    return df


def load_from_path_normalized(path):
    df = pd.read_excel(path)
    df = normalize_data(df)
    df = df.rename(columns={"llm_new_belief": f"post_stance"})
    return df[["persona_id", "topic", "post_stance", "positive"]]


def add_positive_column(df):
    negative_mask = df["statement_formulation"].isin(NEGATIVE_FORMULATIONS)
    # True if statement is NOT a negative formulation
    df["positive"] = ~negative_mask
    return df

def load_data(sample):
    if sample == "human":
        df = pd.read_csv('results/merged_llm_participants_data_with_normalized_beliefs.csv')
        df = add_positive_column(df)
        df = df.rename(columns={"final_belief_normalized": "post_stance"})
        df = df[["persona_id", "topic", "post_stance", "positive"]]
    else:
        df = load_from_path_normalized(paths[sample])
    df["sample"] = sample
    return df


paths = {
    "qwen": "results/Qwen3_32B_all_personas_results.xlsx",
    "qwen_all-positive": "results/ablations/Qwen3-32B_all-positive.xlsx",
    "qwen_all-negative": "results/ablations/Qwen3-32B_all-negative.xlsx",
    "llama": "results/Llama-3.3-70B-Instruct_all_personas_results.xlsx",
    "llama_all-positive": "results/ablations/Llama-3.3-70B-Instruct_all-positive.xlsx",
    "llama_all-negative": "results/ablations/Llama-3.3-70B-Instruct_all-negative.xlsx",
}

dfs = [load_data(sample) for sample in paths.keys()] + [load_data("human")]
merged_df = pd.concat(dfs, ignore_index=True)


print("############ Impact on human pre-stance #############")
df = pd.read_csv('results/merged_llm_participants_data_with_normalized_beliefs.csv')
df = add_positive_column(df)
print("---- initial_belief ----")
print(df.groupby("positive")["initial_belief"].mean().to_string())
print("-----------------------------------")
print("---- initial_belief_normalized ----")
print(df.groupby("positive")["initial_belief_normalized"].mean().to_string())
print("-----------------------------------")
print("---- final_belief ----")
print(df.groupby("positive")["final_belief"].mean().to_string())
print("-----------------------------------")
print("---- final_belief_normalized ----")
print(df.groupby("positive")["final_belief_normalized"].mean().to_string())
print("-----------------------------------")
print("########################################################")
exit()

print("############ Average post_stance by sample #############")
avg_df = merged_df.groupby("sample")["post_stance"].mean()
print(avg_df.to_string())
print("########################################################")



print("############ per persona impact of statement formulation #############")
for model in ["qwen", "llama"]:
    base = load_data(model)
    all_positive = load_data(f"{model}_all-positive")
    all_negative = load_data(f"{model}_all-negative")
print("######################################################################")