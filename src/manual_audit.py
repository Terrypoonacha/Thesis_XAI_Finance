from agentic_auditor import get_shap_details, query_regulations

print("--- Manual Audit Step 1: SHAP ---")
shap_output = get_shap_details("541")
print(shap_output)

print("\n--- Manual Audit Step 2: Regulations ---")
# Extract top feature from SHAP output (mock extraction)
# "V14" is usually top for 541.
reg_output = query_regulations("V14") 
print(reg_output)

reg_output_2 = query_regulations("Transparency")
print(reg_output_2)
