from py2neo import Graph
import pandas as pd
import warnings
from tqdm.auto import tqdm

website = 'fenninggroupnas.ucsd.edu'
port = 7687

graph = Graph(f"bolt://{website}:{port}", auth=("neo4j", "magenta-traffic-powder-anatomy-basket-8461")) # magenta-etc is the passphrase

def grab_sample(batch_id, sample_id):
    """
    Retrieve the latest unique nodes representing a given sample of a given batch.

    Parameters:
        batch_id (str): The identifier of the batch.
        sample_id (str): The identifier of the sample within the batch.

    Returns:
        pandas.DataFrame: A DataFrame containing the latest unique nodes representing the sample.
    """
    
    g = graph.run(f"""MATCH (n:Chemical)
    WHERE (n.batch_id = '{batch_id}' and n.sample_id = '{sample_id}')
    WITH n.chemical_id AS chemical_id, collect(n) AS nodes
    RETURN nodes[-1] as unique_node
    UNION
    MATCH (n:Action)
    WHERE (n.batch_id = '{batch_id}' and n.sample_id = '{sample_id}')
    WITH n.step_id AS step_id, collect(n) AS nodes
    RETURN nodes[-1] as unique_node""")
    return g.to_data_frame()

def grab_batch(batch_id):
    """
    Fetches all samples of a given batch.

    Parameters:
        batch_id (str): The ID of the batch.

    Returns:
        numpy.ndarray: Array containing pairs of batch IDs and sample IDs of the given batch.
    """
    
    g = graph.run(f"""MATCH (n:Action)
    WHERE (n.step_id = '1' and n.batch_id = '{batch_id}')
    WITH n.sample_id AS sample_id, collect(n) AS nodes
    WITH nodes[-1] AS unique_node
    RETURN unique_node.batch_id, unique_node.sample_id
    ORDER BY unique_node.batch_id, unique_node.sample_id""")
    
    return g.to_ndarray()

def get_batch_names():
    """
    Retrieves all batch IDs.

    Returns:
        numpy.ndarray: Array containing unique batch IDs.
        
    """
    
    g = graph.run(f"""MATCH (n)
    RETURN DISTINCT n.batch_id""")
    return g.to_ndarray()

def node_cluster_size(batch_id, sample_id):
    """
    Calculates the number of the nodes for a given batch ID and sample ID.

    Parameters:
        batch_id (str): The ID of the batch.
        sample_id (str): The ID of the sample.

    Returns:
        int: The size of the node cluster.
    """
    
    return len(grab_sample(batch_id, sample_id))

def segment_samples(rules):
    """
    Retrieves all samples that meet the passed conditions in the rules.

    Parameters:
        rules (list): A list of rules specifying conditions for segmenting samples.

    Returns:
        set: batch_id and sample_id pairs that meet the passed conditions.

    Raises:
        ValueError: If the passed ruleset does not contain any rules or if a rule format is invalid,
                    or if the ruleset results in 0 samples.
        Warning: If samples in a segment have different numbers of nodes, tabularizing the segment is not advised.

    """
    
    nds = []
    if len(rules) == 0:
        raise ValueError('Passed ruleset does not contain any rules.')
        
    
    for r in rules:
        if r[0] == '(':
            if (r[:3] != '(n.' and r[:3] != '(c.'):
                raise ValueError(f"The first two characters in a rule must be 'n.' or 'c.': {r} ")
        else:
            if (r[:2] != 'n.' and r[:2] != 'c.'):
                raise ValueError(f"The first two characters in a rule must be 'n.' or 'c.': {r} ")

        if r[0 + (r[0] == '(')] == 'n':
            q = f"""MATCH (n:Action)
            WHERE ({r})
            WITH n.batch_id as batch_id, n.sample_id AS sample_id, collect(n) AS nodes
            WITH nodes[-1] AS unique_node
            RETURN unique_node.batch_id, unique_node.sample_id
            ORDER BY unique_node.batch_id, unique_node.sample_id"""
            nds.append(graph.run(q).to_ndarray())
            
        if r[0 + (r[0] == '(')] == 'c':
            q = f"""MATCH (c:Chemical)
            WHERE ({r})
            WITH c.batch_id as batch_id, c.sample_id AS sample_id, collect(c) AS nodes
            WITH nodes[-1] AS unique_node
            RETURN unique_node.batch_id, unique_node.sample_id
            ORDER BY unique_node.batch_id, unique_node.sample_id"""
            nds.append(graph.run(q).to_ndarray())
            
    if len(rules) == 1:
        return set(nds[0])
    else:  
        nds = [set([tuple(j) for j in i]) for i in nds]
        nds = set.intersection(*nds)
        
    if len(nds) == 0:
        raise ValueError('Passed ruleset results in 0 samples.')
    
    lens = []
    
    for i in nds:
        lens.append(node_cluster_size(*i))
    
    lens = set(lens)
    
    if len(lens) > 1:
        warnings.warn("Samples in this segment have different number of nodes. Tabularizing this segment is not advised.")
        
    return nds

def create_row(sample, include_fitted_metrics = True, fitted_metrics_only = False):
    """
    Creates a row of data from a dataframe representing the nodes of a given sample.

    Parameters:
        sample (pandas.DataFrame): DataFrame containing a sample's information.
        include_fitted_metrics (bool, optional): Flag indicating whether to include fitted metrics. Defaults to True.
        fitted_metrics_only (bool, optional): Flag indicating whether to include only fitted metrics. Defaults to False.

    Returns:
        dict: Row of data containing sample information.

    Note:
        This function creates a row of data from a unique node representing a sample in the graph database.
        It extracts information such as batch ID, sample ID, solute concentrations, solvent details,
        solution molarity and volume, fitted metrics, and color metrics, based on specified flags.
    """
    
    sample = sample['unique_node']
    row = {}
    solute_counter = 1
    solvent_counter = 1
    row['batch_id'] = sample[0]['batch_id']
    row['sample_id'] = sample[0]['sample_id']
    for i in sample:
        if 'chem_type' in i:
            if i['chem_type'] == 'solute' and 'concentration' in i and not fitted_metrics_only:
                if 'concentration' in i:
                    row[f"solute_{i['content']}"] = float(i['concentration'])
                else:
                    row[f"solute_{solute_counter}"] = i['content']
                    solute_counter += 1
                    
            if i['chem_type'] == 'solvent' and not fitted_metrics_only and False:
                elems = extract_dict(i['content'])
                if '' in elems and len(elems['']) == len(i['content'])-1:
                    row[f'solvent_{solvent_counter}'] = i['content']
                else:
                    row[f'solvent_{solvent_counter}_elem_dict'] = elems
                solvent_counter += 1

            if i['chem_type'] == 'solution' and not fitted_metrics_only and False:
                row[f"solution_{i['content']}_molarity"] = float(i['molarity'])
                row[f"solution_{i['content']}_volume"] = float(i['volume']) 

        if 'action' in i:
            if i['action'] == 'fitted_metrics' and include_fitted_metrics:
                for j in i:
                    if j not in ['action', 'batch_id', 'sample_id', 'step_id', 't_samplepresent_0']:
                        row[j] = i[j]
            if i['action'] == 'colormetrics':
                row['curve_L'] = i['curve_L']
                row['curve_x0'] = i['curve_x0']
                row['curve_k'] = i['curve_k']
    return row
    
def tabularize_samples(samples, include_fitted_metrics = True, fitted_metrics_only = False):
    """
    Converts given samples into a tabular format.

    Args:
        samples (list): List of tuples containing batch ID and sample ID pairs.
        include_fitted_metrics (bool, optional): Flag indicating whether to include fitted metrics. Defaults to True.
        fitted_metrics_only (bool, optional): Flag indicating whether to include only fitted metrics. Defaults to False.

    Returns:
        pandas.DataFrame: DataFrame containing tabularized sample data.

    Note:
        Missing solute concentrations are replaced with 0.
    """
    
    rows = []
    for i in tqdm(samples, desc = 'Tabularizing Samples'):
        rows.append(create_row(grab_sample(*i),include_fitted_metrics, fitted_metrics_only))

    df = pd.DataFrame(rows)
    for i in range(len(df.columns)):
        if 'solute_' in df.columns[i] and not str.isdigit(df.columns[i][-1]):
            df[df.columns[i]] = df[df.columns[i]].fillna(0)
    return df
