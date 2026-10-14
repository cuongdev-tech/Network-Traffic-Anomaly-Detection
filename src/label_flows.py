"""Buoc 7.1 (Option A): Gan nhan thu cong cho flows.csv -> labeled_flows.csv.
Rule: 1-suspicious neu rst>0 va (duration<6s hoac n_pkts<16), nguoc lai 0-normal.
Ly do: flow ngan + reset nhieu = dau hieu scan/inject; flow dai/on dinh = normal.
"""
import pandas as pd
import os

IN = r"D:\PATTT\data\processed\flows.csv"
OUT = r"D:\PATTT\data\processed\labeled_flows.csv"

def rule(row):
    if (row["rst"] > 0) and (row["duration"] < 6 or row["n_pkts"] < 16):
        return 1, "rst>0 and (duration<6 or n_pkts<16)"
    return 0, "flow dai/on dinh"

def main():
    df = pd.read_csv(IN)
    labels, rules = [], []
    for _, r in df.iterrows():
        lb, rl = rule(r)
        labels.append(lb); rules.append(rl)
    df["label"] = labels
    df["rule"] = rules
    df.to_csv(OUT, index=False)
    print(f"Doc: {IN} ({len(df)} flows)")
    print(f"Luu: {OUT}")
    print("Phan bo label:")
    print(df["label"].value_counts().to_string())
    print("\nChi tiet:")
    print(df[["src","dst","sport","dport","n_pkts","duration","rst","label"]].to_string())

if __name__ == "__main__":
    main()
