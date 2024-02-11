from py2neo import Graph
import pandas as pd
import warnings

website = 'fenninggroupnas.ucsd.edu'
port = 7687

graph = Graph(f"bolt://{website}:{port}", auth=("neo4j", "magenta-traffic-powder-anatomy-basket-8461")) # magenta-etc is the passphrase

def grab_sample(batch_id, sample_id):
    g = graph.run(f"""MATCH (n:Chemical)
    WHERE (n.batch_id = '{batch_id}' and n.sample_id = '{sample_id}')
    WITH n.chemical_id AS chemical_id, collect(n) AS nodes
    RETURN nodes[0] as unique_node
    UNION
    MATCH (n:Action)
    WHERE (n.batch_id = '{batch_id}' and n.sample_id = '{sample_id}')
    WITH n.step_id AS step_id, collect(n) AS nodes
    RETURN nodes[0] as unique_node""")
    if to_df == False:
        return g
    else:
        return g.to_data_frame()


def grab_batch(batch_id):
    g = graph.run(f"""MATCH (n:Action)
    WHERE (n.step_id = 1 and n.batch_id = '{batch_id}')
    WITH n.sample_id AS sample_id, collect(n) AS nodes
    WITH nodes[0] AS unique_node
    RETURN unique_node.batch_id, unique_node.sample_id
    ORDER BY unique_node.batch_id, unique_node.sample_id""")
    
    return g.to_ndarray()

def node_cluster_size(batch_id, sample_id):
    return len(grab_sample(batch_id, sample_id))
    

def get_sample_steps(batch_id, sample_id):
    g = graph.run(f"""MATCH (n:Action)
    WHERE (n.batch_id = '{batch_id}' and n.sample_id = '{sample_id}')
    WITH n.step_id AS step_id, collect(n) AS nodes
    with nodes[0]  as unique_node
    RETURN unique_node.step_id
    ORDER BY unique_node.step_id""")
    return g.to_data_frame()

def all_samples():
    g = graph.run("""MATCH (n:Action)
    WHERE (n.step_id = 1)
    WITH n.batch_id AS batch_id, n.sample_id AS sample_id, collect(n) AS nodes
    WITH nodes[0] AS unique_node
    RETURN unique_node.batch_id, unique_node.sample_id
    ORDER BY unique_node.batch_id, unique_node.sample_id""")

    return g.to_ndarray()


def segment_samples(rules):
    nds = []
    
    for r in rules:
        if (r[:2] != 'n.' and r[:2] != 'c.'):
            raise ValueError(f"The first two characters in a rule must be 'n.' or 'c.': {r} ")
            
        if r[0] == 'n':
            q = f"""MATCH (n:Action)
            WHERE ({r})
            WITH n.sample_id AS sample_id, collect(n) AS nodes
            WITH nodes[0] AS unique_node
            RETURN unique_node.batch_id, unique_node.sample_id
            ORDER BY unique_node.batch_id, unique_node.sample_id"""
            nds.append(graph.run(q).to_ndarray())
            
        if r[0] == 'c':
            q = f"""MATCH (c:Chemical)
            WHERE ({r})
            WITH c.sample_id AS sample_id, collect(c) AS nodes
            WITH nodes[0] AS unique_node
            RETURN unique_node.batch_id, unique_node.sample_id
            ORDER BY unique_node.batch_id, unique_node.sample_id"""
            nds.append(graph.run(q).to_ndarray())
            
    if len(rules) == 0:
        nds = set([tuple(i) for i in all_samples()])
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

def extract_elem_dict(s):
    elem_dict = {}
    
    split_elem = s.split('_')
    for i in split_elem:
        num_idx = 0
        for j in range(len(i)):
            if str.isdigit(i[j]):
                num_idx= j
                break
        if i[:num_idx] != '':
            elem_dict[i[:num_idx]] = float(i[num_idx:])
        else:
            elem_dict['Unknown'] = float(i[num_idx:])

    return elem_dict

def create_row(sample):
    new_row = {}
    sample = sample['unique_node']
    new_row['batch_id'] = sample[0]['batch_id']
    new_row['sample_id'] = sample[0]['sample_id']

    for i in sample:
        if 'chem_type' in i:
            new_row[i['chem_type']] = i['content']

            if 'volume' in i:
                new_row[f"{i['chem_type']}_volume"] = i['volume']

            if 'molarity' in i:
                new_row[f"{i['chem_type']}_molarity"] = i['molarity']

        if 'action' in i:
            if i['action'] == 'drop':
                new_row['drop_air_gap'] = bool(i['drop_air_gap'])
                new_row['drop_blow_out'] = bool(i['drop_blow_out'])
                new_row['drop_height'] = i['drop_height']
                new_row['drop_rate'] = i['drop_rate']
                new_row['drop_reuse_tip'] = bool(i['drop_reuse_tip'])
                new_row['drop_slow_retract'] = bool(i['drop_slow_retract'])
                new_row['drop_slow_travel'] = bool(i['drop_slow_travel'])

            if i['action'] == 'spin':
                new_row['spin_acceleration'] = i['spin_acceleration']
                new_row['spin_rpm'] = i['spin_rpm']
                new_row['spin_duration'] = i['spin_duration']

            if i['action'] == 'anneal':
                new_row['anneal_duration'] = i['anneal_duration']
                new_row['anneal_temperature'] = i['anneal_temperature']

            if i['action'] == 'rest':
                new_row['rest_duration'] = i['rest_duration']

            if i['action'] == 'fitted_metrics':
                new_row['bf_inhomogeneity_0'] = i['bf_inhomogeneity_0']
                if "df_median_0" in i:
                    new_row['df_median_0'] = i['df_median_0']
                new_row['pl_fwhm_0'] = i['pl_fwhm_0']
                new_row['pl_intensity_0'] = i['pl_intensity_0']
                new_row['pl_peakev_0'] = i['pl_peakev_0']    
    return new_row

def tabularize_samples(samples):
    rows = [create_row(grab_sample(*i)) for i in samples]
    df = pd.DataFrame(rows)
    if 'antisolvent' in df:
        elem_df = df['antisolvent'].apply(extract_elem_dict).apply(pd.Series).fillna(0)
        elem_df.columns = [f"anti_solvent_{i}" for i in elem_df.columns]
        num_elems = len(elem_df.columns)
        df = pd.concat([df,elem_df], axis = 1)
        col_index = df.columns.to_list().index('antisolvent')
        df = df.drop('antisolvent', axis = 1)
        col_list = df.columns.to_list()
        new_col_order = col_list[:col_index] + col_list[-num_elems:] + col_list[col_index:-num_elems]
        df = df.loc[:,new_col_order]
        
    return df
