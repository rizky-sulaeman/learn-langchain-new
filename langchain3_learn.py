"""
Di file ini saya mencoba pendekatan ReAct (Reasoning + Acting) seperti yang dijelaskan
di dokumentasi LangChain. Jadi alurnya bukan sekadar:
input → output

Tapi ada proses:
1. Thought  → model mikir dulu perlu apa
2. Action   → pilih tool yang relevan
3. Input    → kirim parameter ke tool
4. Observation → lihat hasil tool ulang sampai cukup informasi
5. Final Answer → baru jawab ke user

Use case yang dipakai:
simulasi customer service sederhana.

Kalau user tanya soal refund, agent tidak boleh nebak.
Dia harus ambil data order dulu, cek policy, baru kasih jawaban.
"""
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()


"""
 --- 1. DEFINISI TOOLS (Bagian "Acting" di ReAct) ---
Catatan:
Tools ini ibarat jembatan antara model
Docstring sangat penting karena agent membaca ini untuk menentukan
kapan tool dipakai dan bagaimana cara pakainya.
Anggap saja seperti dokumentasi API tapi untuk AI.
"""

@tool
def get_latest_order(customer_id: str) -> dict:
    """Gets the latest order details for a given customer ID."""
    print(f"\n[Tool Execution] -> Mengambil data order untuk {customer_id}...")
    return {
        "order_id": "ORD-999",
        "item": "Mechanical Keyboard",
        "purchase_date": "2024-02-10",
        "status": "Delivered"
    }

@tool
def calculate_refund_eligibility(purchase_date: str) -> str:
    """Checks if a purchase date is eligible for a refund (within 30 days)."""
    print(f"\n[Tool Execution] -> Mengecek aturan refund untuk tanggal: {purchase_date}...")
    return "Eligible. The purchase is within the 30-day return window."

# --- 2. THE REASONING ENGINE ---


def run_react_agent():
    llm = ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model="stepfun/step-3.5-flash:free", temperature=0)

    # Giving the agent access to our defined functions
    tools = [get_latest_order, calculate_refund_eligibility]

    """--- 3. PROMPT UTAMA (Struktur Reasoning) ---
    Catatan:
    Di versi yang baru (dengan), kita tidak perlu lagi mendefinisikan
    prompt template manual berisi "Thought/Action/Observation".
    Secara otomatis menggunakan kemampuan native "Tool Calling" model
    seperti gpt-4o, yang jauh lebih efisien, pintar, dan tidak mudah salah format!
    Kita cukup tambahkan `state_modifier` sebagai system prompt.
    """
    system_prompt = "You are a customer service agent that strictly uses provided tools to confirm facts."

    """--- 4. MENYUSUN AGENT ---
    create_agent sekarang menggabungkan LLM dan tools jadi satu Graph secara native.
    Agent Executor lama (di Langchain klasik) ditinggalkan karena Tool Calling graph
    jauh lebih stabil untuk proses reasoningnya.
    """
    agent_executor = create_agent(llm, tools, system_prompt=system_prompt)

    # --- 5. EXECUTION & OBSERVATION ---
    print("\n" + "="*60)
    print("USER QUERY: I am customer CUST-123. Can I get a refund on my last order?")
    print("="*60)
    
    # agent menerima message array untuk properti "messages"
    response = agent_executor.invoke({
        "messages": [("user", "I am customer CUST-123. Can I get a refund on my last order?")]
    })

    # --- 6. UNPACKING THE REASONING ---
    """Setelah agent selesai, jawaban akhirnya ada di pesan terakhir array `messages`.
    Tapi kita bisa lihat jejak reasoning di seluruh log pesan (Human -> AI -> Tool -> dst).
    Ini sangat powerful untuk debugging dan memahami bagaimana model memutuskan tindakan.
    """
    messages = response["messages"]

    print("\n" + "="*60)
    print("FINAL ANSWER TO USER:")
    print(messages[-1].content) # Jawaban final ada di message index terakhir
    print("="*60)

    print("\n--- BEHIND THE SCENES: THE REASONING TRACE ---")

    # Enum message (bisa HumanMessage, AIMessage, ToolMessage)
    for step_num, msg in enumerate(messages, 1):
        print(f"\nStep {step_num}:")
        
        if msg.type == "human":
            print(f"  USER: {msg.content}")
            
        elif msg.type == "ai":
            # Jika AI memanggil tool, dia adalah bagian "Thought / Action"
            if msg.tool_calls:
                print(f"  THOUGHT: Model realized it needs external data.")
                for call in msg.tool_calls:
                    print(f"  ACTION TAKEN: {call['name']} with input '{call['args']}'")
            else:
                # Jika tidak ada tool calls dan message-nya AI, berarti "Final Answer"
                print(f"  FINAL AI THOUGHT: {msg.content}")
                
        elif msg.type == "tool":
            # Ini ekuivalen dengan bagian "Observation" di lama
            print(f"  OBSERVATION (Tool Result - {msg.name}): {msg.content}")

run_react_agent()

"""
My Thought:

model diberi kemampuan (tools) dan dibiarkan mencari informasi sendiri
secara bertahap.

Agent yang menentukan:
- data apa yang kurang
- tool mana yang perlu dipakai
- kapan informasi sudah cukup untuk jawab user

Kalau nanti saya tambah tool baru (misalnya buat return shipment atau cek status kiriman),
saya tidak perlu ubah prompt utama.
Agent tinggal belajar menggunakan tool itu.

"""