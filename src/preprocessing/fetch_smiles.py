import requests

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