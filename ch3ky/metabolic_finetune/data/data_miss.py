import pandas as pd

# 1. 读取数据
# 如果文件不在同一目录，改成你的实际路径
file_path = "merged_HMDB_drugbank_reactions.csv"


df = pd.read_csv(file_path, encoding="latin1")


# 2. 统一处理 Metabolite_Role（去空格、小写）
df["Metabolite_Role"] = df["Metabolite_Role"].astype(str).str.strip().str.lower()

# 3. 按 Reaction_ID 汇总每个反应的所有角色
role_set_by_rxn = (
    df.groupby("Reaction_ID")["Metabolite_Role"]
      .agg(lambda x: set(x.dropna()))
      .reset_index(name="role_set")
)

# 4. 只包含 product 的 Reaction_ID（role_set 恰好等于 {"product"}）
only_product = role_set_by_rxn[
    role_set_by_rxn["role_set"].apply(lambda s: s == {"product"})
]["Reaction_ID"]

# 5. 只包含 substrate 的 Reaction_ID（role_set 恰好等于 {"substrate"}）
only_substrate = role_set_by_rxn[
    role_set_by_rxn["role_set"].apply(lambda s: s == {"substrate"})
]["Reaction_ID"]

# 6. 输出结果
print("只包含 product 的 Reaction_ID：")
print(only_product.to_list())

print("\n只包含 substrate 的 Reaction_ID：")
print(only_substrate.to_list())

# 7. 如需保存成文件，也可以这样：
only_product.to_csv("reaction_ids_only_product.txt", index=False, header=False)
only_substrate.to_csv("reaction_ids_only_substrate.txt", index=False, header=False)
