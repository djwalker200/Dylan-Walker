import numpy as np 
import pandas as pd
import re
import os
import shutil
from lxml import etree
import xml.etree.ElementTree as ET
from pathlib import Path
import sys
import argparse
from tqdm import tqdm
from definitions import DefinedTerm,FoundTerm
from xml_helpers import find_in_text,markup_terms

os.environ["PYTHONHASHSEED"] = "0"

def loadPhrases(filename):
    with open(filename) as f:
        phrases = [line.rstrip() for line in f]
    return phrases

def cleanString(string):
    delims = ['.',':',',',';']
     
    for d in delims:
        idx = string.find(d)
        if idx != -1:
            string = string[:idx]

    return string

# Forms default scope by returning scope of nearest parent
def getDefaultScope(idnum):
    inilist = [m.start() for m in re.finditer(r":", idnum)]
    parent = idnum[ : inilist[-2]]
    

    return parent

# Takes a scoping phrase string and disambiguates it by looking for keywords
# Returns default scope if no scope match is found

def disambiguateScope(scoping_tags,idnum):
    
    
    if len(scoping_tags) == 0:
        return getDefaultScope(idnum),True
    
    scoping_string = ''.join([str(s) + " " for s in  scoping_tags])
        
    keywords = ["section","chapter"," part","article","subchapter","subpart","division","department","agency","title"]
    scoping_string = scoping_string.lower()
    for keyword in keywords:
        if keyword in scoping_string and keyword in idnum:
            
            idx = idnum.find(keyword)
            
            scope_prefix = idnum[:idx]
            
            scope = idnum[idx:]
            end = scope.find(":",len(keyword) + 1)
            
            if end != -1:
                scope = scope[:end] 
            
            return (scope_prefix + scope,False)
    

    # If no well-defined scope is found use default parent scope
    return (getDefaultScope(idnum),True)



def getText(filepath):
    
    text = ""
    tree = ET.parse(filepath)
    root = tree.getroot()

    for s in root.iter('subsect'):
        for txt in s.itertext():
            text = text + " " + txt

    for s in root.iter('para'):
        for txt in s.itertext():
            text = text + " " + txt

    return text

def findScoping(filepath,phrases):

    scoping = {}
    text = getText(filepath)

    for phrase in phrases:
        idx = text.find(phrase)
        
        if idx != -1:
            
            phrase_len = len(phrase.split())
            full_phrase = text[idx:idx + len(phrase) + 20]

            full_phrase = full_phrase.split()[:phrase_len + 1]
            
            string = ''.join([str(s) + " " for s in full_phrase])
            
            string = cleanString(string)
            
            scoping[string] = idx

    return scoping

def countScopingFiles(df,phrases):  
    
    filenames = df['datapath']
    scoping = {}

    for f in filenames:

        scoping[f] = findScoping(f,phrases)
        

    return scoping


def findLocalScoping(df,scoping_dict):

    filenames = list(df['datapath'])
    terms = list(df['Term'])

    local = {}

    frac = 0.0
    for word,fname in zip(terms,filenames):
        present = len(scoping_dict[fname]) > 0
        if local.get(word) is None:
            local[word] = []
        if present:
            frac += 1.0 
            local[word].append(fname)
    
    frac /= len(df)
    return frac 




# Stores all instances of well-scoped definitions in a dictionary 
# Inputs:
# df : DataFrame of extracted definition information for a given state
# phrases : list(str) containing each of the pre-defined scoping phrases
# 
# Output:
# definition_dictionary : dictionary of form {term : defined_terms}
# type(term) = str
# type(defined_terms) = list(DefinedTerm)

        
def getDefinitionDictionary(df,phrases,store_dataframe = False):  
    
    filenames = df['datapath']
    terms = df['Term']
    IDs = df['idnum']
    definitions = df['Definition']
    scoping = []

    definition_dictionary = {}

    for f,term,definition,idnum in zip(filenames,terms,definitions,IDs):
        
        scoping_dict = findScoping(f,phrases)
        scoping_tags = list(scoping_dict.keys())
        
        scope,default_scope = disambiguateScope(scoping_tags,idnum)
        
        defined_term = DefinedTerm(term,definition,idnum,scope=scope,default_scope = default_scope)
    
        if definition_dictionary.get(term) is None:
            definition_dictionary[term] = [defined_term]
        else:
            definition_dictionary[term].append(defined_term)

        if store_dataframe:
            scoping.append([term,f,idnum,scope])
            
    
    if store_dataframe:
        scoping_df = pd.DataFrame(data=scoping,columns=["Term","Filepath","IDNum",'Scope'])
        scoping_df.to_csv("definition_dictionary.csv")
    
    return definition_dictionary



# Runs markup on a single state 
# Inputs: 
# definition_dictionary: dictionary of well-scoped defined terms sorted by term length
    # {term : defined_terms}
    # type(term) = str
    # type(defined_terms) = list(DefinedTerm)
# state_path: path to the state corpus
# write_path: path to write modified_xml files

def stateMarkup(definition_dictionary,state_path,write_path):
    
    allfiles = list(Path(state_path).rglob("*.[xX][mM][lL]"))
    
    # If write directory already exists then clear it out
    if os.path.isdir(write_path):
            shutil.rmtree(write_path)
            os.mkdir(write_path)
    else:
        os.mkdir(write_path)
            
    results = []
    
    for filepath in tqdm(allfiles):
        
        try:
            tree = etree.parse(filepath)
        except Exception:
            continue
            
                
        root = tree.getroot()
        idnum = root[0].attrib['idnums']
        
        new_xml = markup_terms(tree,definition_dictionary,idnum)
    
        # WRITE XML FILE
        with open(f"{write_path}/{idnum}.xml", "wb") as f:
            f.write(new_xml)
    
    results = pd.DataFrame(data=results,columns=['Term','IDNum','Start','End','Definition','Definition ID'])
         
    return results
   
    

def computeStateFrequencies(states,term_directory,phrases,display=False):
    
    results = []

    for state in tqdm(states):

        csv_path = term_directory + "/" + state + "/dictionary.csv"
        df = pd.read_csv(csv_path)

        scoping_dict = countScopingFiles(df,phrases)

        frac = findLocalScoping(df,scoping_dict)
        
        if display:
            print(f"Percentage of terms with scoping language for {state}: {round(100 * frac,2)}")
        
        results.append([state,frac])

    results = pd.DataFrame(data=results,columns=["State","Scoping Rate"])

    results.to_csv("state_scoping_rates.csv")
        

def computeTermScoping(states,term_directory,phrases,output_directory):
    
    top_scoping = []
    for state in tqdm(states):

        csv_path = term_directory + "/" + state + "/dictionary.csv"
        df = pd.read_csv(csv_path)

        term_df = getDefinitionDictionary(df,phrases,store_dataframe=True)
    
        scoping_results = term_df['Scoping Phrase']
        
        top_phrase = None
        count = None
        if not scoping_results.mode().empty:
            
            top_phrase = scoping_results.mode()[0]
            count = scoping_results.value_counts()[0]
            
        term_df.to_csv(f"{output_directory}/{state}_term_scoping.csv")
        top_scoping.append([state,top_phrase,count])
        
    
        
    top_scoping = pd.DataFrame(data=top_scoping,columns=["State","Top Phrase","Occurences"])
    top_scoping.to_csv("state_scoping_phrases.csv")
    


def computeMarkup(states,term_directory,phrases,corpus_directory,output_directory):
    
    # Loop over each state
    for state in tqdm(states):

        # Load state extracted definitions dataframe
        csv_path = term_directory + "/" + state + "/dictionary.csv"
        df = pd.read_csv(csv_path)
        
        # Stores all well-scoped definitions into a dictionary
        definition_dictionary = getDefinitionDictionary(df,phrases)
        
        # Sort terms longest to shortest
        definition_dictionary = dict(sorted(list(definition_dictionary.items()), key = lambda key : len(key[0]),reverse=True))
        
        # Markup each file and assign definitions
        state_path = corpus_directory + "/" + state
        write_path = output_directory + "/" + state
        results = stateMarkup(definition_dictionary,state_path,write_path)
        
        if len(results) > 0:
            results.to_csv(f"{output_directory}/{state}_markup.csv")
        
   
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', type=str, default = "term_scoping")
    parser.add_argument('--all_states',action='store_true',default = False)
    parser.add_argument('--states', nargs='+')
    parser.add_argument('--corpus_directory',type=str,default="States")
    parser.add_argument('--term_directory',type=str,default="extraction_output")
    parser.add_argument('--phrases_file',type=str,default="scopingPhrases.txt")
    args = parser.parse_args()


    corpus_directory = args.corpus_directory
    term_directory = args.term_directory
    phrases = loadPhrases(args.phrases_file)
    
    if args.all_states:
        states = [s for s in os.listdir(term_directory) if s != ".DS_Store"]

    else:
        states = args.states


    if args.mode == "frequency":
        output_directory = "frequency_results"
        computeStateFrequencies(states,term_directory,phrases)
        
    if args.mode == "term_scoping":
        output_directory = "scoping_results"
        computeTermScoping(states,term_directory,phrases,output_directory)
        
    if args.mode == "markup":
        output_directory = "markup_results"
        computeMarkup(states,term_directory,phrases,corpus_directory,output_directory)

        
    
