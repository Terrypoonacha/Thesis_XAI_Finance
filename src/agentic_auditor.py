import os
import sys
import joblib
import pandas as pd
import shap
import xgboost as xgb
import re
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_xgb.pkl"
KNOWLEDGE_PATH = PROJECT_ROOT / "data" / "knowledge"

# --- 1. Knowledge Loading (Fallback) ---
print("Loading Regulatory Documents into Memory...")
global_docs = []
try:
    pdf_files = list(KNOWLEDGE_PATH.glob("*.pdf"))
    for pdf in pdf_files:
        loader = PyPDFLoader(str(pdf))
        global_docs.extend(loader.load())
    print(f"Loaded {len(global_docs)} pages of regulation.")
except Exception as e:
    print(f"Error loading PDFs: {e}")
    global_docs = []

# --- 2. Tools ---

def get_shap_details(transaction_id_str):
    """
    Fetches the top 3 contributing features for a given transaction ID using SHAP.
    Input: transaction_id (int) as a string.
    Output: String describing top 3 features and their impact.
    """
    try:
        transaction_id = int(transaction_id_str)
    except ValueError:
        return "Error: Transaction ID must be an integer."

    print(f"Fetching SHAP details for Transaction {transaction_id}...")
    print(f"DEBUG: Model path: {MODEL_PATH}, Exists: {MODEL_PATH.exists()}")
    print(f"DEBUG: Data path: {DATA_PATH}, Exists: {DATA_PATH.exists()}")

    try:
        model = joblib.load(MODEL_PATH)
        df = pd.read_csv(DATA_PATH)
        X = df.drop('Class', axis=1)
        
        if transaction_id >= len(X):
             return f"Error: Transaction ID {transaction_id} not found."
             
        row = X.iloc[[transaction_id]]
        
        # XGBoost 1.7.6 + SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(row)
        
        feature_names = X.columns
        s_vals = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
        
        feature_importance = []
        for i, col in enumerate(feature_names):
            feature_importance.append((col, s_vals[i], row.iloc[0][col]))
            
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        top_3 = feature_importance[:3]
        
        result = f"Top 3 Features driving the model decision for Transaction {transaction_id}:\n"
        for rank, (feat, impact, val) in enumerate(top_3, 1):
            direction = "INCREASING" if impact > 0 else "DECREASING"
            result += f"{rank}. {feat} (Value: {val:.4f}): Impact = {impact:.4f} ({direction} fraud risk)\n"
            
        return result

    except Exception as e:
        return f"Error calculating SHAP values: {e}"

def query_regulations(query):
    """
    Searches BaFin MaRisk and EU AI Act for relevant clauses using keyword search.
    Input: Query string (e.g., "Transparency", "V14").
    Output: Relevant regulatory text with citations.
    """
    print(f"Searching regulations for: {query}")
    
    hits = []
    query_lower = query.lower()
    
    # Map key concepts to text if needed, effectively "semantic expansion"
    if "v14" in query_lower:
        query_lower = "risk" # V14 is anon, but map to general risk for demo
    
    for doc in global_docs:
        content = doc.page_content
        if query_lower in content.lower():
            # Extract a snippet around the match
            idx = content.lower().find(query_lower)
            start = max(0, idx - 100)
            end = min(len(content), idx + 300)
            snippet = content[start:end].replace('\n', ' ')
            
            source = Path(doc.metadata.get('source', 'Unknown')).name
            page = doc.metadata.get('page', 'Unknown')
            hits.append(f"- [{source}, Page {page}]: ...{snippet}...")
            
            if len(hits) >= 3: # Limit results
                break
    
    if not hits:
        if "transparency" in query_lower or "explainability" in query_lower:
             return "Article 13 (Transparency) of EU AI Act requires that high-risk AI systems shall be designed ... to ensure sufficient transparency."
        return "No direct mentions found. Ensure compliance with general MaRisk AT 4.3.2 (Model Risk)."
        
    return "Relevant Regulatory Clauses:\n" + "\n".join(hits)

# --- 3. Agent (Manual Loop) ---

def generate_compliance_memo(transaction_id):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return "Error: GOOGLE_API_KEY not set."

    # Initialize LLM
    try:
        llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", google_api_key=api_key, temperature=0)
    except Exception as e:
        return f"Error initializing LLM: {e}"

    # Tool mapping
    tool_map = {
        "SHAP_Fetcher": get_shap_details,
        "Regulatory_Retriever": query_regulations
    }
    
    # Prompt construction
    prompt_template = """Answer the following questions as best you can. You have access to the following tools:

SHAP_Fetcher: Useful for finding out WHY a specific transaction was flagged. Input should be the transaction ID.
Regulatory_Retriever: Useful for finding legal justification. Input should be a single keyword like 'Transparency', 'Risk', 'Model', or 'Article 13'.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [SHAP_Fetcher, Regulatory_Retriever]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {question}
Thought:"""

    system_instruction = """
    You are a BaFin Compliance Officer. When a transaction is flagged, you must:
    (1) Use SHAP_Fetcher to find why the model flagged it.
    (2) Use Regulatory_Retriever to find the legal justification. Search for concepts like "Transparency", "Model Risk", or "Article 13".
    (3) Write a professional Compliance Memo.
    
    Constraint: Cite specific PDF page numbers provided by the retriever in your final memo.
    """
    
    question = f"{system_instruction}\n\nInvestigate Transaction {transaction_id} and produce the Compliance Memo."
    
    history = prompt_template.format(question=question)
    
    max_steps = 15
    final_answer = None
    
    for i in range(max_steps):
        # Call LLM
        try:
            response_msg = llm.invoke(history)
            response = response_msg.content
        except Exception as e:
            return f"LLM Error: {e}"
            
        print(f"\nStep {i+1} LLM Output:\n{response}")
        
        history += response + "\n"
        
        # Check for Final Answer
        if "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
            break
            
        # Parse Action
        action_match = re.search(r"Action:\s*(.*)", response)
        action_input_match = re.search(r"Action Input:\s*(.*)", response)
        
        if action_match and action_input_match:
            action = action_match.group(1).strip()
            action_input = action_input_match.group(1).strip()
            
            # Sanitize input
            action_input = action_input.strip('"').strip("'")

            print(f"Executing {action} with input: {action_input}")
            
            if action in tool_map:
                try:
                    observation = tool_map[action](action_input)
                except Exception as e:
                    observation = f"Error executing tool: {e}"
            else:
                observation = f"Error: Tool {action} not found."
                
            print(f"Observation: {observation[:150]}...") 
            history += f"Observation: {observation}\nThought:"
        else:
             if "Final Answer" not in response and "Action:" not in response:
                 history += "\nThought:"

    if final_answer:
        return final_answer
    else:
        return "Agent failed to reach final answer."

if __name__ == "__main__":
    memo = generate_compliance_memo(541)
    print("\n--- Final Compliance Memo ---\n")
    print(memo)
