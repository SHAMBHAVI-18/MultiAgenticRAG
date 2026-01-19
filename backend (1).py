import os
import re
import pandas as pd
from typing import List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

# --- 1. Data Structures & Enums ---
class IntentCategory(Enum):
    GENERAL_AWARENESS = "GENERAL_AWARENESS"
    HR_QUERY = "HR_QUERY"
    FINANCE_QUERY = "FINANCE_QUERY"
    SENSITIVE_INTERNAL = "SENSITIVE_INTERNAL"
    PII = "PII"

@dataclass
class ClassificationResult:
    intent: IntentCategory
    confidence: float
    reason: str

@dataclass
class VerificationResult:
    verified: bool
    employee_number: Optional[int]
    message: str

# --- 2. Governance Rules ---
class ColumnGovernance:
    BLOCKED_COLUMNS = {'EmployeeNumber', 'MonthlyIncome', 'HourlyRate'}
    PUBLIC_COLUMNS = {'JobRole', 'Department', 'JobSatisfaction', 'WorkLifeBalance', 'BusinessTravel'}
    
    @classmethod
    def get_allowed_columns_for_agent(cls, agent_type: str) -> set:
        return cls.PUBLIC_COLUMNS.copy()

# --- 3. Security & Classification ---
class SecurityGate:
    @classmethod
    def validate_input(cls, query: str) -> Tuple[bool, str]:
        if re.search(r'ignore\s+previous', query, re.IGNORECASE): return False, "Security Violation"
        return True, "Safe"

class IntentClassifier:
    FINANCE_KEYWORDS = {'salary', 'ctc', 'compensation', 'cost', 'budget', 'bonus', 'stock'}
    HR_KEYWORDS = {'hr', 'policy', 'leave', 'holiday', 'performance', 'promotion', 'training'}
    SENSITIVE_KEYWORDS = {'my', 'me', 'mine', 'personal', 'i am'}
    
    @classmethod
    def classify(cls, query: str) -> ClassificationResult:
        query_lower = query.lower()
        # "My" or "Me" triggers Sensitive check immediately
        if any(k in query_lower for k in cls.SENSITIVE_KEYWORDS):
             return ClassificationResult(IntentCategory.SENSITIVE_INTERNAL, 1.0, "Personal Inquiry")
        if any(k in query_lower for k in cls.FINANCE_KEYWORDS):
            return ClassificationResult(IntentCategory.FINANCE_QUERY, 0.9, "Finance")
        if any(k in query_lower for k in cls.HR_KEYWORDS):
            return ClassificationResult(IntentCategory.HR_QUERY, 0.9, "HR")
        return ClassificationResult(IntentCategory.GENERAL_AWARENESS, 0.5, "General")

# --- 4. Auth & Agents ---
class AuthenticationSystem:
    def __init__(self, credentials_df: pd.DataFrame):
        self.credentials = {str(row['dummy_email']): {'password': str(row['dummy_password']), 'employee_number': row['EmployeeNumber']} for _, row in credentials_df.iterrows()}
    
    def verify_credentials(self, email, password):
        # Strip whitespace just in case
        email = email.strip()
        password = password.strip()
        if email in self.credentials and str(self.credentials[email]['password']) == password:
            return VerificationResult(True, self.credentials[email]['employee_number'], "Success")
        return VerificationResult(False, None, "Invalid credentials")

class BaseAgent:
    def __init__(self, name): self.name = name

class GeneralAgent(BaseAgent):
    def __init__(self): super().__init__("GeneralAgent")

class HRAgent(BaseAgent):
    def __init__(self): super().__init__("HRAgent")
    def requires_verification(self, intent): 
        return intent == IntentCategory.SENSITIVE_INTERNAL

class FinanceAgent(BaseAgent):
    def __init__(self): super().__init__("FinanceAgent")
    def requires_verification(self, intent): 
        return intent == IntentCategory.SENSITIVE_INTERNAL

# --- 5. Main Orchestrator ---
class RAGOrchestrator:
    def __init__(self, data_df, creds_df, vector_store, llm):
        self.data_df, self.vector_store, self.llm = data_df, vector_store, llm
        self.auth_system = AuthenticationSystem(creds_df)
        self.general_agent, self.hr_agent, self.finance_agent = GeneralAgent(), HRAgent(), FinanceAgent()
        self.verified_sessions = {} # Stores {session_id: employee_number}

    def login(self, email, password, session_id):
        res = self.auth_system.verify_credentials(email, password)
        if res.verified: 
            self.verified_sessions[session_id] = res.employee_number
        return res

    def logout(self, session_id):
        if session_id in self.verified_sessions:
            del self.verified_sessions[session_id]

    def process_query(self, query, session_id="default"):
        # 1. Security Check
        safe, msg = SecurityGate.validate_input(query)
        if not safe: return f"⛔ {msg}"
        
        # 2. Intent Classification
        classification = IntentClassifier.classify(query)
        
        # 3. Agent Selection
        if classification.intent == IntentCategory.HR_QUERY: agent = self.hr_agent
        elif classification.intent == IntentCategory.FINANCE_QUERY: agent = self.finance_agent
        elif classification.intent == IntentCategory.SENSITIVE_INTERNAL: agent = self.hr_agent # Default sensitive to HR/Personal
        else: agent = self.general_agent
        
        # 4. Auth Verification Logic
        # If it's sensitive OR the agent requires it
        is_sensitive = (classification.intent == IntentCategory.SENSITIVE_INTERNAL)
        
        if is_sensitive:
            if session_id not in self.verified_sessions:
                return "🔒 **Access Denied:** You are asking for personal data. Please **Log In** using the sidebar on the left."
        
        # 5. Retrieval & Response
        docs = self.vector_store.as_retriever().get_relevant_documents(query)
        context = "\n".join([d.page_content for d in docs[:3]])
        
        prompt = f"""
        Role: You are a corporate assistant.
        Context: {context}
        User Question: {query}
        Instruction: Answer the question based on the context. If the context doesn't help, answer generally about corporate standards.
        Answer:
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Error connecting to AI: {e}"

# --- 6. Initialization ---
def initialize_system():
    try:
        data_df = pd.read_csv("RAGbot_finance_enriched.csv")
        creds_df = pd.read_csv("dummy_employee_credentials.csv")
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        texts = data_df.select_dtypes(include='object').agg(' '.join, axis=1).fillna('').tolist()
        vector_store = FAISS.from_texts(texts, embeddings)
        return data_df, creds_df, vector_store
    except Exception as e:
        print(f"File Error: {e}")
        raise e
