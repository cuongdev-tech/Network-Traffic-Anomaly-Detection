"""Buoc 4: Doc file pcap bang Scapy - trich thong tin co ban moi goi tin."""
from pathlib import Path
from scapy.all import rdpcap, IP, TCP, UDP, ICMP

BASE_DIR = Path(__file__).resolve().parent.parent

def describe_pcap(path, n=10):
    pkts = rdpcap(path)
    print(f"File: {path}")
    print(f"Tong so goi: {len(pkts)}")
    print(f"Goi dau tien: time={float(pkts[0].time):.6f}")
    print(f"Goi cuoi:     time={float(pkts[-1].time):.6f}")
    print(f"Khoang thoi gian: {float(pkts[-1].time)-float(pkts[0].time):.3f}s")
    print("-" * 100)
    for i, p in enumerate(pkts[:n]):
        ts = float(p.time)
        size = len(p)
        if IP in p:
            src, dst = p[IP].src, p[IP].dst
            proto = p[IP].proto  # 6=TCP, 17=UDP, 1=ICMP
            if TCP in p:
                info = f"TCP {p[TCP].sport}->{p[TCP].dport} flags={p[TCP].sprintf('%TCP.flags%')}"
            elif UDP in p:
                info = f"UDP {p[UDP].sport}->{p[UDP].dport}"
            elif ICMP in p:
                info = f"ICMP type={p[ICMP].type}"
            else:
                info = f"proto={proto}"
            print(f"[{i}] t={ts:.3f} {src} -> {dst} | {info} | size={size}")
        else:
            print(f"[{i}] t={ts:.3f} non-IP | {p.summary()} | size={size}")

if __name__ == "__main__":
    describe_pcap(str(BASE_DIR / "data" / "raw" / "attack-injection-hao123.pcap"), n=10)
    print()
    describe_pcap(str(BASE_DIR / "data" / "raw" / "attack-injection-id1cn.pcap"), n=10)
