import os
import json
import requests
from multiprocessing import Pool

def fetch_smiles(drug_name):
    """
    Fetch SMILES string for a given drug_name from PubChem.
    """
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/CanonicalSMILES/JSON"
    url = base_url.format(drug_name)
    response = requests.get(url)
    if response.status_code == 200:
        try:
            smiles = response.json()['PropertyTable']['Properties'][0]['CanonicalSMILES']
            return smiles
        except (KeyError, IndexError):
            return None
    else:
        return None
    
## caching and parallel processing for faster computation

def fetch_smiles_with_cache(drug_name, cache_file="smiles_cache.json"):
    """
    Fetch SMILES string for a given drug_name with caching.
    """
    # Load cache if exists
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)
    else:
        cache = {}

    # Check if the result is already cached
    if drug_name in cache:
        return cache[drug_name]

    # Fetch from PubChem API
    smiles = fetch_smiles(drug_name)
    cache[drug_name] = smiles

    # Save updated cache
    with open(cache_file, "w") as f:
        json.dump(cache, f)

    return smiles

    
def fetch_smiles_parallel(drug_names):
    """
    Fetch SMILES for a list of drug names using multiprocessing.
    """
    with Pool(processes=8) as pool:  # Adjust the number of processes as needed
        smiles_list = pool.map(fetch_smiles_with_cache, drug_names)
    return smiles_list
