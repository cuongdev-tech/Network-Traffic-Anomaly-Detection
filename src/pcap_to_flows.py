"""Buoc 5-6: Gom goi tin thanh Flow (5-tuple) + trich feature flow-level va host-level."""
from scapy.all import rdpcap, IP, TCP, UDP
from collections import defaultdict
import pandas as pd
import math
import os

def get_pkt_info(p):
    """Tra ve (src, dst, sport, dport, proto, flags_str, size, time) hoac None neu non-IP."""
    if IP not in p:
        return None
    src, dst = p[IP].src, p[IP].dst
    t = float(p.time)
    size = len(p)
    if TCP in p:
        return src, dst, p[TCP].sport, p[TCP].dport, 6, p[TCP].sprintf("%TCP.flags%"), size, t
    elif UDP in p:
        return src, dst, p[UDP].sport, p[UDP].dport, 17, "", size, t
    else:
        proto = p[IP].proto
        return src, dst, 0, 0, proto, "", size, t

def canon_key(src, dst, sport, dport, proto):
    """Khoa 2 chieu: goi di va goi ve cung 1 flow."""
    if (src, sport) <= (dst, dport):
        return (src, dst, sport, dport, proto)
    else:
        return (dst, src, dport, sport, proto)

def pcap_to_flows(path):
    pkts = rdpcap(path)
    flows = defaultdict(list)  # key -> list of (time, src, size, flags)
    for p in pkts:
        info = get_pkt_info(p)
        if info is None:
            continue
        src, dst, sport, dport, proto, flags, size, t = info
        key = canon_key(src, dst, sport, dport, proto)
        flows[key].append({"time": t, "src": src, "size": size, "flags": flags})
    rows = []
    for (ip1, ip2, sp, dp, proto) in flows:
        lst = sorted(flows[(ip1, ip2, sp, dp, proto)], key=lambda x: x["time"])
        times = [x["time"] for x in lst]
        sizes = [x["size"] for x in lst]
        start, end = times[0], times[-1]
        dur = end - start
        n = len(lst)
        # goi xuoi = goi tu initiator (goi dau tien)
        init_src = lst[0]["src"]
        fwd = [x for x in lst if x["src"] == init_src]
        bwd = [x for x in lst if x["src"] != init_src]
        fwd_bytes = sum(x["size"] for x in fwd)
        bwd_bytes = sum(x["size"] for x in bwd)
        # inter-arrival
        iats = [times[i+1]-times[i] for i in range(len(times)-1)]
        mean_iat = sum(iats)/len(iats) if iats else 0
        # co TCP
        flags_all = "".join(x["flags"] for x in lst)
        syn = flags_all.count("S")
        ack = flags_all.count("A")
        fin = flags_all.count("F")
        rst = flags_all.count("R")
        psh = flags_all.count("P")
        syn_no_ack = 1 if (syn > 0 and ack == 0) else 0
        rows.append({
            "src": lst[0]["src"], "dst": ip2 if lst[0]["src"]==ip1 else ip1,
            "ip1": ip1, "ip2": ip2, "sport": sp, "dport": dp, "proto": proto,
            "n_pkts": n, "n_fwd": len(fwd), "n_bwd": len(bwd),
            "bytes_total": sum(sizes), "bytes_fwd": fwd_bytes, "bytes_bwd": bwd_bytes,
            "byte_ratio_fwd_bwd": fwd_bytes/(bwd_bytes+1),
            "duration": dur, "mean_iat": mean_iat,
            "mean_size": sum(sizes)/len(sizes),
            "syn": syn, "ack": ack, "fin": fin, "rst": rst, "psh": psh,
            "syn_no_ack": syn_no_ack,
            "start": start,
        })
    return pd.DataFrame(rows)

def entropy(vals):
    from collections import Counter
    c = Counter(vals)
    n = len(vals)
    return -sum((v/n)*math.log2(v/n) for v in c.values())

def flows_to_hosts(flow_df, window=60):
    """Gom theo IP nguon trong cua so 60s."""
    if flow_df.empty:
        return pd.DataFrame()
    df = flow_df.copy()
    df["win"] = (df["start"] // window).astype(int)
    rows = []
    for (src, win), g in df.groupby(["src", "win"]):
        n_flows = len(g)
        span = g["start"].max() - g["start"].min()
        span = span if span > 0 else window
        rows.append({
            "src": src, "window": win,
            "n_flows": n_flows,
            "n_dst_ip": g["dst"].nunique(),
            "n_dst_port": g["dport"].nunique(),
            "conn_per_sec": n_flows/span,
            "avg_duration": g["duration"].mean(),
            "total_bytes": g["bytes_total"].sum(),
            "syn_no_ack_ratio": g["syn_no_ack"].mean(),
            "rst_ratio": (g["rst"]>0).mean(),
            "port_entropy": entropy(list(g["dport"])) if len(g) > 1 else 0,
            # do lech chuan start -> nho = deu dan (beaconing)
            "start_std": g["start"].std() if len(g) > 1 else 0,
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    os.makedirs(r"D:\PATTT\data\processed", exist_ok=True)
    all_f, all_h = [], []
    for f in ["attack-injection-hao123.pcap", "attack-injection-id1cn.pcap"]:
        path = os.path.join(r"D:\PATTT\data\raw", f)
        print(f"\n=== {f} ===")
        fdf = pcap_to_flows(path)
        hdf = flows_to_hosts(fdf)
        fdf["file"] = f
        hdf["file"] = f
        print(f"Flows: {len(fdf)}, Hosts(window 60s): {len(hdf)}")
        print(fdf[["src","dst","sport","dport","proto","n_pkts","duration","syn","ack","rst","syn_no_ack"]].head(10).to_string())
        print(hdf.to_string())
        all_f.append(fdf); all_h.append(hdf)
    pd.concat(all_f).to_csv(r"D:\PATTT\data\processed\flows.csv", index=False)
    pd.concat(all_h).to_csv(r"D:\PATTT\data\processed\hosts.csv", index=False)
    print("\nDa luu flows.csv va hosts.csv")
