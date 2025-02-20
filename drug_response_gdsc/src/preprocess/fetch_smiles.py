import os
import json
import requests
from concurrent.futures import ThreadPoolExecutor

# global in-memory cache
cache_memory = {}

### 1. 기본 단일 요청 함수 ###
def fetch_smiles(drug_name, session=None):
    """
    Fetch SMILES string for a given drug_name from PubChem.
    """
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/CanonicalSMILES/JSON"
    url = base_url.format(drug_name)

    try:
        response = session.get(url, timeout=5) if session else requests.get(url, timeout=5)
        if response.status_code == 200:
            try:
                smiles = response.json()['PropertyTable']['Properties'][0]['CanonicalSMILES']
                return smiles
            except (KeyError, IndexError):
                return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed for {drug_name}: {e}")
    return None


### 2. 캐시 로드 및 저장 함수 ###

def load_cache(cache_file="experiments/smiles_cache.json"):
    """
    Load the cache into memory.
    """
    global cache_memory
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache_memory = json.load(f)
        print(f"Cache loaded from {cache_file}")
    else:
        cache_memory = {}

def save_cache(cache_file="experiments/smiles_cache.json"):
    """
    Save the cache from memory to file.
    """
    global cache_memory
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(cache_memory, f)
    print(f"Cache saved to {cache_file}")
    

### 3. 병렬 SMILES 요청 ###
def fetch_smiles_parallel(drug_names, cache_file="experiments/smiles_cache.json"):
    """Fetch SMILES strings for multiple drug_names using threading."""
    # Load cache
    load_cache(cache_file)

    # Filter out already cached drug names
    to_fetch = [name for name in drug_names if name not in cache_memory]

    # Fetch missing SMILES
    if to_fetch:
        with requests.Session() as session:
            with ThreadPoolExecutor(max_workers=16) as executor:  # Adjust workers if needed
                results = list(executor.map(fetch_smiles_with_cache, to_fetch, [session] * len(to_fetch)))

            # Update cache
            for drug_name, smiles in zip(to_fetch, results):
                cache_memory[drug_name] = smiles

    # Save cache
    save_cache(cache_file)

    # Return SMILES for all requested drug_names
    return [cache_memory.get(name) for name in drug_names]

### 캐시와 세션을 사용한 SMILES 요청 ###
def fetch_smiles_with_cache(drug_name, session):
    """Fetch SMILES string with caching."""
    if drug_name in cache_memory:
        return cache_memory[drug_name]
    return fetch_smiles(drug_name, session)


# Test cases
if __name__ == "__main__":
    test_drug_names = ["Aspirin", "Ibuprofen", "Paracetamol", "Penicillin", "Caffeine"]

    # Fetch SMILES strings
    smiles_results = fetch_smiles_parallel(test_drug_names)

    # Print results
    for drug_name, smiles in zip(test_drug_names, smiles_results):
        print(f"{drug_name}: {smiles}")