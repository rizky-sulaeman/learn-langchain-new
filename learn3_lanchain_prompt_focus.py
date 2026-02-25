

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
load_dotenv()


llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="stepfun/step-3.5-flash:free",
    temperature=0.1 
)

print("\n" + "==== 1.Role-Based Basic Without Promptemplate ====" * 1)
"""
Section ini adalah contoh penggunaan role-based system message untuk memberikan konteks dan instruksi 
yang spesifik kepada LLM, mensimulasikan peran seorang Principal Security Architect 
yang melakukan review backend secara teknis.
"""

system_prompt_template = """
You are a Principal Security Architect at a fintech company, responsible for reviewing backend architecture and engineering plans before they go into production.

Review the implementation plan carefully and identify any security risks, scalability concerns, or deviations from sound engineering practices.

Begin with a short executive summary (no more than two sentences).

Then organize your findings into three sections:
CRITICAL — issues that could cause serious security or system impact
WARNING — important concerns that should be addressed
SUGGESTIONS — improvements or best-practice recommendations

If there are no critical issues, explicitly state: "No critical issues detected."

For every issue you mention, include a concrete remediation step that engineers can act on.

Focus strictly on backend, infrastructure, and data concerns. Ignore frontend topics completely.

Maintain a professional, constructive tone similar to an internal architecture review.
Avoid generic advice — stay specific to the scenario provided.
"""

messages = [
    SystemMessage(content=system_prompt_template),
    HumanMessage(content="""
Context: We are building a new internal portal using FastAPI and PostgreSQL. The initial implementation plan includes:
1. Using FastAPI for the backend API layer, with SQLAlchemy as the ORM.
2. Deploying the application on AWS using EC2 instances and RDS for the database.
3. Implementing authentication using JWT tokens.
""")
]

print("Mengirim request ke LLM menggunakan role-based system message secara langsung...\n")
response_1 = llm.invoke(messages)
print(response_1.content)
print("\n" + "="*80)


print("\n" + "==== 2. Few-Shot Prompting dengan ChatPromptTemplate ====" * 1)
"""
Section ini menunjukkan bagaimana menyusun prompt dengan format percakapan (chat) yang mensimulasikan 
interaksi antara manusia dan AI, memberikan contoh-contoh spesifik untuk membantu LLM memahami tugas klasifikasi email 
pelanggan dengan lebih baik, termasuk kategori, prioritas, dan alasan di balik klasifikasi tersebut.
"""
# Menyusun template yang mensimulasikan riwayat percakapan untuk memberi contoh ke AI
few_shot_template = ChatPromptTemplate.from_messages([
    ("system", 
     "You are assisting the Customer Success team at TechGlobal. "
     "Classify incoming customer emails into a category and assign a priority level (Level 1, Level 2, Level 3). "
     "Follow the response format used in the examples."
    ),

    ("human", 
     "My app dashboard screen suddenly went blank after last night's update. "
     "My team can't work at all."
    ),
    ("ai", 
     "Category: Technical Bug\n"
     "Priority: Level 1\n"
     "Reason: The customer reports a complete workflow blocker affecting multiple users."
    ),

    ("human", 
     "I need to update the billing information for my account, but I can't find where to do that in the settings."
    ),
    ("ai", 
     "Category: Account Management\n"
     "Priority: Level 2\n"
     "Reason: The customer is requesting an update to their account settings, which is a standard support request but requires attention."
    ),

    ("human", 
     "Please set the primary email address in my company profile to riskisuleman76@gmail.com"
    ),
    ("ai", 
     "Category: Account Management\n"
     "Priority: Level 3\n"
     "Reason: This is a routine administrative request without urgency."
    ),

    ("human", "{user_email}")
])

print("Test Few-Shot Prompting untuk data keluhan email baru...\n")

chain_2 = few_shot_template | llm

response_2 = chain_2.invoke({
    "user_email": "My Invoice from last month is incorrect. It shows a charge for a service I didn't use. Please fix this immediately."
})
print(response_2.content)
print("\n" + "="*80)


print("\n" + "==== 3. Structured Output (Pydantic) ====" * 1)

"""
Section ini berfokus pada penggunaan skema data yang didefinisikan dengan Pydantic untuk memastikan bahwa output dari 
LLM berdasarkan pada informasi yang disebutkan dalam teks kontrak, tanpa melakukan inferensi atau penafsiran tambahan,
sehingga menghasilkan data yang di hasilkan relevan dan akurat sesuai dengan apa yang secara eksplisit tertulis 
dalam dokumen kontrak.

"""

class ContractExtraction(BaseModel):
    client_name: str = Field(description="Full name of the client or partner company.")
    contract_effective_date: str = Field(description="The effective date of the contract in YYYY-MM-DD format.")
    total_value_usd: float = Field(description="Total contract value in USD. If not explicitly stated, return 0.0")
    key_deliverables: List[str] = Field(description="List of key deliverables from the contract.")
    is_auto_renewal: bool = Field(description="Whether the contract is automatically renewed at the end of the period.")

# model dengan skema Pydantic
structured_extractor = llm.with_structured_output(ContractExtraction)

extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are analyzing a legal or contractual document.

Extract only the information that is explicitly stated in the text and map it to the provided schema.

If a field is missing or unclear, use the schema's default value.
Do not infer, guess, or reinterpret legal meaning.

Read carefully — contracts often contain dense language.

"""),
    ("human", "DOCUMENT TEXT:\n{document_text}")
])

# Menyambungkan prompt dinamis dengan LLM yang sudah dibinding ke output berskema (Pipeline)
extraction_chain = extraction_prompt | structured_extractor

document_sample = """
This is a contract between Calista. and Noxans. The contract is effective from 2026-2-2025 and has a total value of $500,000 USD. Key deliverables include:
1. Development of a custom software solution tailored to Noxans Corp's needs.
2. Integration of the solution with existing systems.
The contract will automatically renew for successive one-year terms unless either party provides written notice of non-renewal at least 30 days prior to the expiration date.   
"""

print("Mengekstraksi kontrak ke dalam bentuk Data Object (Pydantic)...\n")
try:
    contract_data = extraction_chain.invoke({"document_text": document_sample})
    
    # Output dijamin bertipe class ContractExtraction dari Pydantic
    print(f"Client Name     : {contract_data.client_name}")
    print(f"Effective Date  : {contract_data.contract_effective_date}")
    print(f"Total Value     : ${contract_data.total_value_usd}")
    print(f"Auto Renewal?   : {contract_data.is_auto_renewal}")
    print(f"Deliverables    : \n - " + "\n - ".join(contract_data.key_deliverables))
except Exception as e:
    print(f"Failed to extract structured output: {e}")

print("\n" + "="*80)


print("\n" + "==== 4. Chain of Thought ====" * 1)
"""
Section ini adalah teknik Chain of Thought (CoT) untuk memecah masalah kompleks yaitu menganalisis
 insiden produksi yang melibatkan kegagalan job ETL, dengan LLM untuk 
 mengikuti langkah-langkah berpikir yang terstruktur, mulai dari mengidentifikasi masalah yang diamati, 
 menyusun hipotesis penyebab, menentukan cara validasi, 
 hingga memberikan rekomendasi mitigasi atau pencegahan untuk memastikan analisis yang baik.

"""

cot_prompt = PromptTemplate(
    input_variables=["problem"],
    template="""
You are a Senior Data Engineer working on production reliability.

Analyze the incident using the following troubleshooting flow.

Step 1 — Observed Issue 
Describe what is happening based on the report.

Step 2 — Likely Root Causes  
List 2 to 3 plausible technical causes.

Step 3 — How to Validate  
Mention a specific log, metric, query, or command that would confirm the hypothesis.

Step 4 — Mitigation / Prevention  
Recommend actions to prevent recurrence.

Incident:
{problem}

Provide the analysis using the Step 1 to Step 4 structure.
"""
)

problem_case = "Yesterday, our nightly ETL job that processes user activity data failed with a timeout error. Because of this, the data pipeline is broken and our analytics dashboard is not updating with the latest user metrics. The error logs show a timeout when connecting to the database, but there are no recent changes to the database or network configuration."

print("Analyzing system issues in production using Chain of Thought...\n")
cot_chain = cot_prompt | llm
cot_response = cot_chain.invoke({"problem": problem_case})

print(cot_response.content)

print("\n" + "="*80)

