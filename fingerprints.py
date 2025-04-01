#importing libraries
import pandas as pd
import numpy as np
np.product = np.prod  # Patch the import issue
from padelpy import padeldescriptor
from rdkit.Chem import AllChem as Chem
from rdkit import DataStructs
import os
import custom_preprocessing as cp
from glob import glob
from astartes.molecules import train_test_split_molecules
import sys



def canonical_smiles(df, smiles_column):
    def get_canonical_smiles(smiles):
        # Ensure that the value is a string
        if isinstance(smiles, str):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return Chem.MolToSmiles(mol)
            else:
                return None  # Return None for invalid SMILES
        else:
            return None  # Return None if it's not a string

    df['canonical_smiles'] = df[smiles_column].apply(get_canonical_smiles)
    # Remove rows with invalid or non-string SMILES
    df = df[df['canonical_smiles'].notnull()]
    return df
def rational_split (df,x, y ):
    x_train, x_test, y_train, y_test, train_index, test_index = train_test_split_molecules(molecules=x, 
    y=y,
    test_size=0.3,
    train_size=0.7,
    fingerprint="daylight_fingerprint",
    fprints_hopts={
        "minPath": 2,
        "maxPath": 5,
        "fpSize": 200,
        "bitsPerHash": 4,
        "useHs": 1,
        "tgtDensity": 0.4,
        "minSize": 64,
    },
    sampler="scaffold",
    random_state=0,
    hopts={
        "shuffle": True,
    },)
    x_train_df = pd.DataFrame(x_train, y_train.index, columns=['canonical_smiles'])
    x_test_df = pd.DataFrame(x_test, y_test.index, columns=['canonical_smiles'])
    y_train_df = pd.DataFrame(y_train, y_train.index, columns=['Class'])
    y_test_df = pd.DataFrame(y_test, y_test.index, columns=['Class'])
    return x_train_df, x_test_df, y_train_df, y_test_df
def compute_fps(df, name, fol_path): 
    xml_files = glob("*.xml")
    xml_files.sort()
    FP_list = [
    'AP2DC','AD2D','EState','CDKExt','CDK','CDKGraph','KRFPC','KRFP','MACCS','PubChem','SubFPC','SubFP', 'descriptors']
    fp = dict(zip(FP_list, xml_files))
    df['canonical_smiles'].to_csv(name+'.smi', sep='\t', index=False, header=False)
    #Calculate fingerprints
    for i in FP_list:
        padeldescriptor(mol_dir=name+'.smi',
                    d_file=i+'.csv',
                    descriptortypes= fp[i],
                    retainorder=True, 
                    removesalt=True,
                    threads=2,
                    detectaromaticity=True,
                    standardizetautomers=True,
                    standardizenitro=True,
                    fingerprints=True
                    )
        Fingerprint = pd.read_csv(i+'.csv').set_index(df.index)
        Fingerprint = Fingerprint.drop('Name', axis=1)
        Fingerprint.to_csv(i+'.csv')
        print(i+'.csv', 'done')
    #load at pc

    fp_at = pd.read_csv('AD2D.csv'     , index_col=0)
    fp_es = pd.read_csv('EState.csv'   , index_col=0)
    fp_ke = pd.read_csv('KRFP.csv'     , index_col=0)
    fp_pc = pd.read_csv('PubChem.csv'  , index_col=0)
    fp_ss = pd.read_csv('SubFP.csv'    , index_col=0)
    fp_cd = pd.read_csv('CDKGraph.csv' , index_col=0)
    fp_cn = pd.read_csv('CDK.csv'      , index_col=0)
    fp_kc = pd.read_csv('KRFPC.csv'    , index_col=0)
    fp_ce = pd.read_csv('CDKExt.csv'   , index_col=0)
    fp_sc = pd.read_csv('SubFPC.csv'   , index_col=0)
    fp_ac = pd.read_csv('AP2DC.csv'    , index_col=0)
    fp_ma = pd.read_csv('MACCS.csv'    , index_col=0)
    fp_de = pd.read_csv('descriptors.csv'    , index_col=0)
    ''
    #save
    fp_at.to_csv(os.path.join(fol_path, 'xat_' + fol_path +  '.csv'))
    fp_es.to_csv(os.path.join(fol_path, 'xes_' + fol_path +  '.csv'))
    fp_ke.to_csv(os.path.join(fol_path, 'xke_' + fol_path +  '.csv'))
    fp_pc.to_csv(os.path.join(fol_path, 'xpc_' + fol_path +  '.csv'))
    fp_ss.to_csv(os.path.join(fol_path, 'xss_' + fol_path +  '.csv'))
    fp_cd.to_csv(os.path.join(fol_path, 'xcd_' + fol_path +  '.csv'))
    fp_cn.to_csv(os.path.join(fol_path, 'xcn_' + fol_path +  '.csv'))
    fp_kc.to_csv(os.path.join(fol_path, 'xkc_' + fol_path +  '.csv'))
    fp_ce.to_csv(os.path.join(fol_path, 'xce_' + fol_path +  '.csv'))
    fp_sc.to_csv(os.path.join(fol_path, 'xsc_' + fol_path +  '.csv'))
    fp_ac.to_csv(os.path.join(fol_path, 'xac_' + fol_path +  '.csv'))
    fp_ma.to_csv(os.path.join(fol_path, 'xma_' + fol_path +  '.csv'))
    fp_de.to_csv(os.path.join(fol_path, 'xde_' + fol_path +  '.csv'))
    return fp_at, fp_es, fp_ke, fp_pc, fp_ss, fp_cd, fp_cn, fp_kc, fp_ce, fp_sc, fp_ac, fp_ma, fp_de
def main ():
    df = pd.read_csv("ampk_compounds_pre-processed.csv", index_col=0)
    df = canonical_smiles(df, "Smiles")
    df = cp.remove_inorganic(df, "canonical_smiles")
    df = cp.remove_mixtures(df, "canonical_smiles")
    df = df.drop_duplicates(subset = ["canonical_smiles"], keep=False)
    df.to_csv("ampk_compounds_pre-processed.csv")
    x_train, x_test, y_train, y_test = rational_split (df, df["canonical_smiles"], df["Class"])
    fp_at, fp_es, fp_ke, fp_pc, fp_ss, fp_cd, fp_cn, fp_kc, fp_ce, fp_sc, fp_ac, fp_ma, fp_de = compute_fps(x_train, "classification/train", "train")
    fp_at, fp_es, fp_ke, fp_pc, fp_ss, fp_cd, fp_cn, fp_kc, fp_ce, fp_sc, fp_ac, fp_ma, fp_de = compute_fps(x_test,"classification/test","test")
    y_train.to_csv(os.path.join('train_ampk', 'y_train' +'.csv'))
    x_train.to_csv(os.path.join('train_ampk', 'x_train' +'.csv'))
    x_test.to_csv(os.path.join('test_ampk', 'x_test' + '.csv'))
    y_test.to_csv(os.path.join('test_ampk', 'y_test' + '.csv'))
    #print (df)
    print (x_train, x_test)
    print(sys.path)
if __name__ == "__main__":
    main()
