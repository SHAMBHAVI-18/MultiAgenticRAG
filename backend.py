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
    MALICIOUS_OR_BLOCKED = "MALICIOUS_OR_BLOCKED"

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
    BLOCKED_COLUMNS = {'EmployeeNumber', 'MonthlyIncome', 'HourlyRate', 'DailyRate', 'MonthlyRate', 'ChargeOutRate', 'DepartmentBudgetCode', 'Over18', 'StandardHours'}
    SENSITIVE_COLUMNS = {'HR': {'Age', 'Gender', 'MaritalStatus'}, 'Finance': {'TotalCTC', 'ProjectCostAllocation%', 'CostCenter'}}
    INTERNAL_COLUMNS = {'HR': {'JobLevel', 'Education', 'EducationField', 'PerformanceRating', 'TrainingTimesLastYear', 'Attrition', 'OverTime', 'JobInvolvement', 'TotalWorkingYears', 'NumCompaniesWorked', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'PercentSalaryHike', 'SalaryBand'}, 'Finance': {'StockOptionLevel', 'IncentiveType', 'BonusEligibility', 'BillableStatus'}}
    PUBLIC_COLUMNS = {'JobRole', 'Department', 'JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction', 'WorkLifeBalance', 'BusinessTravel'}

    @classmethod
    def get_allowed_columns_for_agent(cls, agent_type: str) -> set:
        if agent_type == "GeneralAgent": return cls.PUBLIC_COLUMNS.copy()
        elif agent_type == "HRAgent": 
            allowed = cls.PUBLIC_COLUMNS.copy()
            allowed.update(cls.SENSITIVE_COLUMNS['HR'])
            allowed.update(cls.INTERNAL_COLUMNS['HR'])
            return allowed
        elif agent_type == "FinanceAgent":
            allowed = cls.PUBLIC_COLUMNS.copy()
            allowed.update(cls.SENSITIVE_COLUMNS['Finance'])
            allowed.update(cls.INTERNAL_COLUMNS['Finance'])
            return allowed
        return set()

# --- 3. Security & Classification ---
class SecurityGate:
    MALICIOUS_PATTERNS = [r'ignore\s+(previous|all|above)\s+instructions', r'system\s+prompt', r'you\s+are\s+now', r'forget\s+(everything|all|previous)', r'show\s+all\s+rows']
    @classmethod
    def validate_input(cls, query: str) -> Tuple[bool, str]:
        for pattern in cls.MALICIOUS_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE): return False, "Potential security violation detected"
        return True, "Input validated"

class IntentClassifier:
    FINANCE_KEYWORDS = {'salary', 'ctc', 'compensation', 'cost', 'budget', 'billable', 'bonus', 'incentive', 'stock', 'financial'}
    HR_KEYWORDS = {'hr', 'employee', 'attrition', 'performance', 'promotion', 'training', 'satisfaction', 'work-life', 'overtime'}
    SENSITIVE_KEYWORDS = {'my', 'personal', 'individual', 'specific employee', 'confidential'}
    
    @classmethod
    def classify(cls, query: str) -> ClassificationResult:
        query_lower = query.lower()
        if any(k in query_lower for k in cls.SENSITIVE_KEYWORDS) or re.search(r'\bmy\b|\bme\b', query_lower):
            return ClassificationResult(IntentCategory.SENSITIVE_INTERNAL, 1.0, "Sensitive keyword")
        if any(k in query_lower for k in cls.FINANCE_KEYWORDS):
            return ClassificationResult(IntentCategory.FINANCE_QUERY, 0.9, "Finance keyword")
        if any(k in query_lower for k in cls.HR_KEYWORDS):
            return ClassificationResult(IntentCategory.HR_QUERY, 0.9, "HR keyword")
        return ClassificationResult(IntentCategory.GENERAL_AWARENESS, 0.5, "General")

# --- 4. Authentication & Agents ---
class AuthenticationSystem:
    def __init__(self, credentials_df: pd.DataFrame):
        self.credentials = {row['dummy_email']: {'password': row['dummy_password'], 'employee_number': row['EmployeeNumber']} for _, row in credentials_df.iterrows()}
    
    def verify_credentials(self, email, password):
        if email in self.credentials and self.credentials[email]['password'] == password:
            return VerificationResult(True, self.credentials[email]['employee_number'], "Success")
        return VerificationResult(False, None, "Invalid credentials")

class BaseAgent:
    def __init__(self, name, allowed_columns): self.name, self.allowed_columns = name, allowed_columns

class GeneralAgent(BaseAgent):
    def __init__(self): super().__init__("GeneralAgent", ColumnGovernance.get_allowed_columns_for_agent("GeneralAgent"))

class HRAgent(BaseAgent):
    def __init__(self): super().__init__("HRAgent", ColumnGovernance.get_allowed_columns_for_agent("HRAgent"))
    def requires_verification(self, query): return "my" in query.lower() or "me" in query.lower()

class FinanceAgent(BaseAgent):
    def __init__(self): super().__init__("FinanceAgent", ColumnGovernance.get_allowed_columns_for_agent("FinanceAgent"))
    def requires_verification(self, query): return "my" in query.lower() or "me" in query.lower()

# --- 5. Main Orchestrator ---
class RAGOrchestrator:
    def __init__(self, data_df, creds_df, vector_store, llm):
        self.data_df, self.vector_store, self.llm = data_df, vector_store, llm
        self.auth_system = AuthenticationSystem(creds_df)
        self.general_agent, self.hr_agent, self.finance_agent = GeneralAgent(), HRAgent(), FinanceAgent()
        self.verified_sessions = {}

    def login(self, email, password, session_id):
        res = self.auth_system.verify_credentials(email, password)
        if res.verified: self.verified_sessions[session_id] = res.employee_number
        return res

    def process_query(self, query, session_id="default"):
        # 1. Security Check
        safe, msg = SecurityGate.validate_input(query)
        if not safe: return f"⛔ {msg}"
        
        # 2. Intent Classification
        classification = IntentClassifier.classify(query)
        
        # 3. Agent Selection
        if classification.intent == IntentCategory.HR_QUERY:
            agent = self.hr_agent
        elif classification.intent == IntentCategory.FINANCE_QUERY:
            agent = self.finance_agent
        else:
            agent = self.general_agent
        
        # 4. Auth Verification
        if hasattr(agent, 'requires_verification') and agent.requires_verification(query):
            if session_id not in self.verified_sessions:
                return "🔒 Authentication Required. Please login using the sidebar."
        
        # 5. Simple Retrieval & Response
        docs = self.vector_store.as_retriever().get_relevant_documents(query)
        context = "\n".join([d.page_content for d in docs[:3]])
        
        prompt = f"""
        Role: You are an enterprise {agent.name}.
        Security: You can ONLY use information from the Context.
        Context: {context}
        User Question: {query}
        Answer:
        """
        
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)

# --- 6. Initialization Helper ---
def initialize_system():
    # Helper to load files and build the system
    try:
        data_df = pd.read_csv("RAGbot_finance_enriched.csv")
        creds_df = pd.read_csv("dummy_employee_credentials.csv")
        
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        texts = data_df.select_dtypes(include='object').agg(' '.join, axis=1).fillna('').tolist()
        vector_store = FAISS.from_texts(texts, embeddings)
        
        return data_df, creds_df, vector_store
    except Exception as e:
        print(f"Error loading files: {e}")
        return None, None, None
