from py2neo import Graph
import pandas as pd
import warnings
from tqdm.auto import tqdm

website = 'fenninggroupnas.ucsd.edu'
port = 7687

graph = Graph(f"bolt://{website}:{port}", auth=("neo4j", "magenta-traffic-powder-anatomy-basket-8461")) # magenta-etc is the passphrase

def grab_sample(batch_id, sample_id):
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
    
def grab_chemicals(batch_id, sample_id):
    g = graph.run(f"""MATCH (n:Chemical)
    WHERE (n.batch_id = '{batch_id}' and n.sample_id = '{sample_id}')
    WITH n.chemical_id AS chemical_id, collect(n) AS nodes
    RETURN nodes[-1] as unique_node""")
    return g.to_data_frame()

def grab_batch(batch_id):
    g = graph.run(f"""MATCH (n:Action)
    WHERE (n.step_id = '1' and n.batch_id = '{batch_id}')
    WITH n.sample_id AS sample_id, collect(n) AS nodes
    WITH nodes[-1] AS unique_node
    RETURN unique_node.batch_id, unique_node.sample_id
    ORDER BY unique_node.batch_id, unique_node.sample_id""")
    
    return g.to_ndarray()

def get_batch_names():
    g = graph.run(f"""MATCH (n)
    RETURN DISTINCT n.batch_id""")
    return g.to_ndarray()

def node_cluster_size(batch_id, sample_id):
    return len(grab_sample(batch_id, sample_id))
    

def get_sample_steps(batch_id, sample_id):
    g = graph.run(f"""MATCH (n:Action)
    WHERE (n.batch_id = '{batch_id}' and n.sample_id = '{sample_id}')
    WITH n.step_id AS step_id, collect(n) AS nodes
    with nodes[-1]  as unique_node
    RETURN unique_node.step_id
    ORDER BY unique_node.step_id""")
    return g.to_data_frame()

def all_samples():
    g = graph.run("""MATCH (n:Action)
    WHERE (n.step_id = 1)
    WITH n.batch_id AS batch_id, n.sample_id AS sample_id, collect(n) AS nodes
    WITH nodes[-1] AS unique_node
    RETURN unique_node.batch_id, unique_node.sample_id
    ORDER BY unique_node.batch_id, unique_node.sample_id""")

    return g.to_ndarray()


def segment_samples(rules):
    nds = []
    
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
            
    if len(rules) == 0:
        nds = set([tuple(i) for i in all_samples()])
    elif len(rules) == 1:
        return nds[0]
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

def extract_dict(s):
    idx_to_split = []
    for i in range(len(s)):
        if s[i] == '_' and str.isdigit(s[i-1]):
            idx_to_split.append(i)
    split_elem = []

    idx_to_split.append(-1)
    split_elem = []

    for i in range(len(idx_to_split)):
        if i == 0:
            split_elem.append(s[0:idx_to_split[0]])
        elif idx_to_split[i] == -1:
            split_elem.append(s[idx_to_split[i-1]+1:])
        else:
            split_elem.append(s[idx_to_split[i-1]+1:idx_to_split[i]])
                    
    elem_dict = {}

    for i in split_elem:
        num_idx = 0
        for j in range(len(i)):
            if str.isdigit(i[j]):
                num_idx= j
                break
        elem_dict[i[:num_idx]] = i[num_idx:]

    return elem_dict

def create_row(sample, include_fitted_metrics = True, fitted_metrics_only = False):
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
def expand_df(df):
    for i in range(len(df.columns)):
        if 'elem_dict' in df.columns[i]:
            elem_df = df[df.columns[i]].apply(pd.Series).fillna(0)
            elem_df.columns = [f"{df.columns[i][:-10]}_{j}" for j in elem_df.columns]
            num_elems = len(elem_df.columns)
            df = pd.concat([df,elem_df], axis = 1)
            col_index = df.columns.to_list().index(df.columns[i])
            df = df.drop(df.columns[i], axis = 1)
            col_list = df.columns.to_list()
            new_col_order = col_list[:col_index] + col_list[-num_elems:] + col_list[col_index:-num_elems]
            df = df.loc[:,new_col_order]
    return df
    
def tabularize_samples(samples, include_fitted_metrics = True, fitted_metrics_only = False):
    rows = []
    for i in tqdm(samples, desc = 'Tabularizing Samples'):
        rows.append(create_row(grab_sample(*i),include_fitted_metrics, fitted_metrics_only))

    df = pd.DataFrame(rows)
    for i in range(len(df.columns)):
        df = expand_df(df)
    for i in range(len(df.columns)):
        if 'solute_' in df.columns[i] and not str.isdigit(df.columns[i][-1]):
            df[df.columns[i]] = df[df.columns[i]].fillna(0)
    return df
