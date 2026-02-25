"""
LangGraph adalah sebuah framework untuk membangun graph yang menghubungkan berbagai node, 
di mana setiap node dapat berupa fungsi yang memanggil LLM (Language Model) atau melakukan operasi lainnya. 
Dengan LangGraph, Anda dapat merancang, menghubungkan berbagai langkah pemrosesan data, dan 
mengelola interaksi antara pengguna dan model.

"""

"""Section Dibawah Ini Adalah Contoh Sederhana Penggunaan LangGraph untuk Membuat Agent yang Menjawab Pertanyaan 
Dengan 
kita mendefinisikan state (di sini MessagesState), membuat node (mock_llm), 
menghubungkan dengan edges dari START ke node dan ke END, lalu mengeksekusi dengan graph.invoke. 
Saat dijalankan, node akan menerima state awal {"messages":[{"role":"user","content":"hi!"}]} 
dan mengembalikan balasan, menghasilkan state akhir dengan pesan “hello world”.
"""

"""“Hello World” graph dengan LangGraph (tanpa memanggil LLM sesungguhnya):"""
from langgraph.graph import StateGraph, MessagesState, START, END

# Fungsi node sederhana yang selalu memberikan balasan statis
def mock_llm(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "hello world"}]}

# Buat graph dengan state MessagesState (berisi daftar pesan)
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("mock_llm", mock_llm)            # tambahkan node
graph_builder.add_edge(START, "mock_llm")               # START -> mock_llm
graph_builder.add_edge("mock_llm", END)                 # mock_llm -> END

graph = graph_builder.compile()                        
result = graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
print(result)

"Konsep Workflow di LangGraph"
"""Workflow adalah cara untuk mendefinisikan alur kerja yang kompleks dengan menghubungkan berbagai node 
(fungsi) dalam sebuah graph.
Dalam LangGraph, sebuah workflow diimplementasikan sebagai graf. Komponen utamanya adalah State, Node, dan Edge.
State: Struktur data bersama (misalnya TypedDict) yang menyimpan semua variabel yang dibutuhkan di seluruh alur.
State terdefinisi sekali dan sama untuk semua node/edge.
Node: Fungsi Python (sinkron/asinkron) yang menerima state (serta objek konfigurasi/runtime) dan mengembalikan 
pembaruan state. Node ini yang melakukan pekerjaan seperti memanggil LLM, memanggil tool, atau operasi logika biasa
Edge: Menentukan alur (flow) di antara node. Edge bisa statis (selalu menuju node tertentu) 
atau kondisional (menggunakan fungsi routing). Nodes mengirim “pesan” melalui edges ke node berikutnya.

"""

"""Section Dibawah Ini contoh graf ini memiliki tiga simpul penting: START, node_a, node_b, END.
 START dan END adalah simpul virtual khusus. START menandai permulaan alur (di mana input awal disuntikkan), 
 sedangkan END menandai terminal. Contoh ini menambahkan 10 pada x di node_a dan mengalikan y di node_b. 
 Keluaran akhir menggabungkan pembaruan tersebut.

LangGraph juga mendukung edges kondisional: misalnya, setelah sebuah node selesai,
 kita bisa memutuskan node mana selanjutnya berdasarkan isi state. 
 Ini dilakukan dengan add_conditional_edges. Sebagai contoh sederhana:"""

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    x: int
    y: int

def node_a(state: State):
    # Tambah 10 ke x
    return {"x": state.get("x", 0) + 10}

def node_b(state: State):
    # Kalikan y dengan 2
    return {"y": state.get("y", 1) * 2}

builder = StateGraph(State)
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)

builder.add_edge(START, "node_a")   # Mulai di node_a
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", END)     # Akhir setelah node_b

graph = builder.compile()
output = graph.invoke({"x": 1, "y": 5})
print(output)



"""
Manajemen State dan Pengurangan (Reducers)
"""
"""
State di LangGraph bersifat persisten selama siklus eksekusi graf. 
Ketika sebuah node mengembalikan pembaruan (hanya sebagian state yang berubah), 
pembaruan tersebut digabungkan ke state utama menggunakan reducers. Secara default,
 setiap kunci state menimpa nilainya dengan nilai baru. 
 Namun kita dapat menambahkan reducer khusus menggunakan typing.Annotated. 
 Contohnya, jika kita ingin menambahkan elemen ke list:
"""
from typing import Annotated
from typing_extensions import TypedDict
import operator

class LogState(TypedDict):
    count: int
    logs: Annotated[list[int], operator.add]

def increment(state: LogState):
    # Hitung +1
    return {"count": state.get("count", 0) + 1}

def add_log(state: LogState):
    # Tambahkan log baru berisi count
    return {"logs": [state["count"]]}

builder3 = StateGraph(LogState)
builder3.add_node("inc", increment)
builder3.add_node("log", add_log)
builder3.add_edge(START, "inc")
builder3.add_edge("inc", "log")
builder3.add_edge("log", END)

graph3 = builder3.compile()
print(graph3.invoke({"count":0, "logs":[]}))
# {'count': 1, 'logs': [0]}

"Router, Manager, dan Memori"
"""
Router: Node khusus yang memilih jalur atau agen lain berdasarkan input. Misalnya,
sebuah fungsi dapat mengklasifikasikan query pengguna dan mengarahkan ke agen yang relevan. 
Manager / Supervisor:  mengorkestrasi agen lain. Misalnya, satu node utama merencanakan tugas (plan) lalu men-delegasi ke node/agen lain sebagai pekerja (worker).
atau menggunakan Send untuk sinyal ke beberapa agen secara paralel. Contoh:

Caching/Persistent Memory): LangGraph mendukung mekanisme caching dan persistence untuk mengurangi
beban komputasi dan menyimpan state. Anda dapat mengaktifkan caching pada node agar
hasil komputasi disimpan (misalnya dengan CachePolicy dan InMemoryCache). Contohnya:
"""

from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, Send, CachePolicy
from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver
import operator



""" STATE
 Ini wadah data utama yang bakal dishare ke semua node dalam graph.
 Semua key di sini bisa diakses dan diupdate oleh node manapun.
 Yang perlu diperkatiin: field logs pakai reducer (operator.add)
 supaya kalau ada beberapa node jalan paralel dan nulis ke logs barengan,
 hasilnya digabung (append), bukan saling timpa."""
class AppState(TypedDict):
    query: str
    plan: str
    result: str
    logs: Annotated[List[str], operator.add]  # reducer agar sinyal bisa tulis paralel
"""
MANAGER / SUPERVISOR
Manager ini kayak "bos" yang ngatur alur kerja. Dia bikin rencana (plan)
terus langsung delegasi ke worker pakai Command(goto=...).
Command itu cara kita buat ngarahin eksekusi ke node lain
sekaligus update state dalam satu langkah.
"""
def manager(state: AppState):
    print("Manager: membuat rencana kerja")
    plan = "Task: Analyze data and generate report."
    
    return Command(
        goto="worker",
        update={"plan": plan}
    )

""" WORKER
 Worker ini yang beneran ngerjain task dari manager.
 Dia ambil plan dari state, proses, terus simpan hasilnya.
 Karena logs pakai reducer, kita cukup return list baru aja ["worker_done"],
 nanti otomatis di-append ke logs yang sudah ada, gak perlu manual concat"""
def worker(state: AppState):
    print("Worker: menjalankan task dari manager")

    result = f"Processed -> {state['plan']}"
    
    return {
        "result": result,
        "logs": ["worker_done"]
    }

#ROUTER SINGLE
"""
Router ini fungsinya kayak persimpangan jalan -- dia yang nentuin
query user mau diarahkan ke agent mana. Kalau query-nya mengandung "multi",
lempar ke multi_router_node (yang nanti sinyal ke banyak agent).
Kalau bukan, langsung ke agent_a aja.
"""
def classify(query):
    # contoh sederhana
    if "multi" in query:
        return "multi_router_node"
    return "agent_a"

def route_to_agent(state: AppState):
    print("Router: menentukan agen tujuan")

    active = classify(state["query"])

    return Command(goto=active)
"""
ROUTER MULTI
Nah ini yang agak tricky. Kalau mau kirim task ke beberapa agent sekaligus
(paralel/sinyal), kita pakai Send. Tapi ingat: Send HARUS dikembalikan
dari fungsi conditional edge (add_conditional_edges), bukan dari node biasa.
Kalau dikembalikan langsung dari node, bakal kena error InvalidUpdateError
karena LangGraph expect node return dict atau Command, bukan list of Send.
"""
def classify_multi(query):
    return ["agent_a", "agent_b"]

def multi_route(state: AppState):
    """Fungsi routing untuk conditional edge -- mengembalikan list of Send.
    Send harus dikembalikan dari conditional edge, bukan dari node biasa."""
    print("Router Multi: kirim ke banyak agen")

    results = classify_multi(state["query"])

    return [
        Send(agent, {"query": state["query"], "logs": []})
        for agent in results
    ]
"""
AGENTS
Ini node-node "pekerja" yang beneran handle task.
agent_a setelah selesai langsung lempar ke manager pakai Command,
jadi ada loop: agent_a -> manager -> worker -> END.
agent_b lebih simpel, dia cuma update state terus selesai (ke END).
"""
def agent_a(state: AppState):
    print("Agent A bekerja")

    return Command(
        goto="manager",
        update={"logs": ["agent_a_done"]}
    )

def agent_b(state: AppState):
    print("Agent B bekerja")

    return {
        "logs": ["agent_b_done"]
    }
"""
GRAPH BUILD
Di sini kita rangkai semua node dan edge jadi satu graph utuh.
Alurnya: START -> router -> (agent_a ATAU multi_router_node)
Kalau single route: router -> agent_a -> manager -> worker -> END
Kalau multi route: router -> multi_router_node -> (agent_a + agent_b paralel)
#
multi_router_node itu cuma pass-through node (gak ngapa-ngapain),
tujuannya biar kita bisa pasang conditional edge yang return Send di situ.
cache_policy di manager berguna supaya kalau input sama, hasilnya di-cache
selama 60 detik -- hemat komputasi kalau ada request berulang.
"""

builder = StateGraph(AppState)

builder.add_node("router", route_to_agent)

builder.add_node("agent_a", agent_a)
builder.add_node("agent_b", agent_b)

builder.add_node(
    "manager",
    manager,
    cache_policy=CachePolicy(ttl=60)  # cache hasil manager selama 60 detik untuk input yang sama
)

builder.add_node("worker", worker)

# Flow utama
builder.add_edge(START, "router")

# sinyal: multi_router_node cuma pass-through, yang penting conditional edge-nya
# nge-return list of Send buat dispatch ke agent_a dan agent_b secara paralel
builder.add_node("multi_router_node", lambda state: state)
builder.add_conditional_edges("multi_router_node", multi_route)

builder.add_edge("agent_b", END)
builder.add_edge("worker", END)

# compile dengan memory (InMemorySaver) + cache (InMemoryCache)
# checkpointer bikin state bisa dipersist antar invoke di thread yang sama
graph = builder.compile(
    cache=InMemoryCache(),
    checkpointer=InMemorySaver()
)
"""
TEST RUN
Jalanin dua skenario buat liat perbedaan single route vs multi route.
Pakai thread_id yang sama supaya state (termasuk logs) lanjut terakumulasi.
- Query pertama gak ada kata "multi" -> masuk ke agent_a -> manager -> worker
- Query kedua ada kata "multi" -> sinyal ke agent_a + agent_b sekaligus

"""
config = {"configurable": {"thread_id": "demo-thread"}}
print("\n=== RUN SINGLE ROUTE ===")
result1 = graph.invoke(
    {
        "query": "please analyze data",
        "logs": []
    },
    config
)

print(result1)

print("\n=== RUN MULTI ROUTE ===")
result2 = graph.invoke(
    {
        "query": "multi analysis please",
        "logs": []
    },
    config
)

print(result2)