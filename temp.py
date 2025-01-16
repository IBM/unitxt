import os

import requests

url = "https://flowpilot.res.ibm.com/api/sql"

payload = {
    "messages": [{"content": "How many singers do we have?", "role": "user"}],
    "dataSourceId": "concert_singer",
}

api_host = os.getenv("FLOWPILOT_API_HOST", "https://flowpilot.res.ibm.com/api")

api_key = os.getenv("FLOWPILOT_API_KEY", None)
base_headers = {
    "Content-Type": "application/json",
    "accept": "application/json",
    "Authorization": f"Bearer {api_key}",
}

sql = """SELECT MFGN AS "MFGN", ORNO AS "Plant Order", PLANT AS "Mfg Plant", FOP_FNP AS "FNP Firm Flag", FOP_HW AS "HW Firm Flag", NOMINATION AS "Nomination", CAT_GRP AS "Product", MFGN_TP AS "Order Type", IOT_CD AS "Geo", ISO_DESC AS "Country", CUST_MAST_SHIP_TO_NM AS "Ship to Customer", BRAND_NM AS "Brand", CUTOFF_VAL AS "Cut-off Value", REVENUE AS "Revenue ($M)", OED AS "OED", CRAD AS "CRAD", PSSD AS "PSSD", RSSD AS "RSSD", SHIP_DT AS "Ship Date" FROM SSCEP_PZ.ORNO_DTL_SNAPSHOT WHERE UPPER(ISO_CD) = 'MX' AND (SHIP_DT BETWEEN '2023-10-01' AND '2023-12-31') ORDER BY REVENUE DESC"""
database_id = "clv2h06cu00038jw9yhhpebfs"

sql_api = f"{api_host}/sql"
sql_payload = {
    "sql": sql,
    "dataSourceId": database_id,
}
sql_response = requests.post(
    sql_api, headers=base_headers, json=sql_payload, verify=False
)
