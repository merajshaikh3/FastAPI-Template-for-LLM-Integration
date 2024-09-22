from datetime import datetime
from typing import List, Dict, Optional, Union

def has_valid_keys(result_json: Dict) -> None:

    LLM_OUTPUT_KEYS = {'sentiment', 'message', 'reason'}

    for key in LLM_OUTPUT_KEYS:
        if key in result_json:
            continue
        else:
            raise KeyError(f"The key '{key}' is not present in the dictionary.")
        
