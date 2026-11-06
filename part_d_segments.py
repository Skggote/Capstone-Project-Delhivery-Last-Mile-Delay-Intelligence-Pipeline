"""Part D (segments): hub profile, K-Means, separation evidence, tier check."""
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")

profile = (df.groupby("hub_id").agg(shipments=("shipment_id", "size"),
                                    mean_delivery_hours=("delivery_hours", "mean"),
                                    delay_rate=("is_delayed", "mean"),
                                    mean_distance_km=("distance_km", "mean"),
                                    mean_n_stops=("n_stops", "mean")).reset_index().sort_values("hub_id"))
profile.round(4).to_csv("outputs/hub_profile.csv", index=False)

cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
Z = StandardScaler().fit_transform(profile[cols])


def fit(k):
    return KMeans(n_clusters=k, random_state=42, n_init=10).fit(Z).labels_


def ratio(labels):
    across, within = [], []
    for i, j in combinations(range(len(Z)), 2):
        (within if labels[i] == labels[j] else across).append(np.linalg.norm(Z[i] - Z[j]))
    return float(np.mean(across) / np.mean(within))


sep = []
for k in [2, 3, 4, 5]:
    lab = fit(k)
    sizes = sorted(np.bincount(lab), reverse=True)
    sep.append({"K": k, "across_over_within": ratio(lab), "cluster_sizes": "-".join(str(int(s)) for s in sizes)})
pd.DataFrame(sep).round(4).to_csv("outputs/part_d_separation.csv", index=False)

labels = fit(3)
profile["cluster"] = labels
order = profile.groupby("cluster")["delay_rate"].mean().sort_values().index.tolist()
names = dict(zip(order, ["Standard", "Watch", "Escalate"]))
profile["segment"] = profile["cluster"].map(names)

seg = profile[["hub_id", "segment"]].merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
seg.to_csv("outputs/hub_segments.csv", index=False)

sp = (profile.groupby("segment").agg(hubs=("hub_id", "size"), mean_delivery_hours=("mean_delivery_hours", "mean"),
                                     delay_rate=("delay_rate", "mean"), mean_distance_km=("mean_distance_km", "mean"),
                                     mean_n_stops=("mean_n_stops", "mean")).reset_index())
sp["segment"] = pd.Categorical(sp["segment"], ["Standard", "Watch", "Escalate"])
sp.sort_values("segment").round(4).to_csv("outputs/segment_profile.csv", index=False)

implied = {1: "Standard", 2: "Watch", 3: "Escalate"}
seg[seg["segment"] != seg["hub_tier"].map(implied)].to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
print("part D segments done")
