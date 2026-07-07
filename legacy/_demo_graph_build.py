# Demo v2: knowledge graph tu WordNet + pyvis (chu de climate)
# - Node chinh: 1 mau. Quan he phan biet bang MAU CANH.
# - Bo co-occur ngay tho; chi giu collocation that (verb+obj / noun compound).
from nltk.corpus import wordnet as wn
from pyvis.network import Network

CHOSEN = {
    "reduce":    "reduce.v.01",
    "increase":  "increase.v.01",
    "emission":  "emission.n.01",
    "pollution": "pollution.n.01",
    "carbon":    "carbon.n.01",
    "gas":       "gas.n.02",
    "heat":      "heat.n.01",
    "damage":    "damage.n.01",
}
# Chi nhung collocation THAT (cum co nghia), khong noi bua moi tu cung cau
COLLOCATION = [("reduce","emission"), ("carbon","emission"), ("reduce","pollution")]

# MAU: node 1 mau; moi quan he 1 mau canh
NODE   = "#4e79a7"   # tat ca node mau nay
E_SYN  = "#59a14f"   # synonym - xanh la
E_ANT  = "#e15759"   # antonym - do
E_ISA  = "#9aa0a6"   # is-a    - xam
E_COL  = "#f28e2b"   # collocation - cam

def get_syn(name, word):
    try: return wn.synset(name)
    except Exception:
        ss = wn.synsets(word); return ss[0] if ss else None

net = Network(height="720px", width="100%", bgcolor="#ffffff",
              font_color="#222", notebook=False, cdn_resources="in_line")
net.barnes_hut(gravity=-12000, central_gravity=0.4, spring_length=120)

added = set()
def node(nid, label, size, shape, tip):
    if nid not in added:
        net.add_node(nid, label=label, title=tip, color=NODE, size=size, shape=shape)
        added.add(nid)

for word, sname in CHOSEN.items():
    s = get_syn(sname, word)
    if not s: continue
    node(word, word, 28, "dot", f"{word} ({s.lexname()})<br>{s.definition()}")  # node chinh: dot to
    for h in s.hypernyms()[:1]:                                                   # is-a -> node vuong
        hn = h.name().split(".")[0]
        node(hn, hn, 18, "square", f"concept: {h.definition()}")
        net.add_edge(word, hn, color=E_ISA, title="is-a")
    for syn in sorted({l.name() for l in s.lemmas()} - {word})[:3]:               # synonym -> node nho
        node(f"syn::{syn}", syn, 12, "dot", "synonym")
        net.add_edge(word, f"syn::{syn}", color=E_SYN, title="synonym")
    for ant in sorted({a.name() for l in s.lemmas() for a in l.antonyms()})[:2]:  # antonym -> node nho
        node(f"ant::{ant}", ant, 12, "dot", "antonym")
        net.add_edge(word, f"ant::{ant}", color=E_ANT, title="antonym")

for a, b in COLLOCATION:                                                          # collocation that
    if a in added and b in added:
        net.add_edge(a, b, color=E_COL, title="collocation", width=4)

out = r"C:\Users\THTran\Downloads\KxG\demo_graph.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(net.generate_html())
print("OK ->", out, "| nodes:", len(added))
