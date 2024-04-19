import sys
from yachalk import chalk
sys.path.append("..")

import json
import ollama.client as client
import tqdm


def extractConcepts(prompt: str, metadata={}, model="mistral-openorca:latest"):
    SYS_PROMPT = (
        "Your task is extract the key concepts (and non personal entities) mentioned in the given context. "
        "Extract only the most important and atomistic concepts, if  needed break the concepts down to the simpler concepts."
        "Categorize the concepts in one of the following categories: "
        "[event, concept, place, object, document, organisation, condition, misc]\n"
        "Format your output as a list of json with the following format:\n"
        "[\n"
        "   {\n"
        '       "entity": The Concept,\n'
        '       "importance": The concontextual importance of the concept on a scale of 1 to 5 (5 being the highest),\n'
        '       "category": The Type of Concept,\n'
        "   }, \n"
        "{ }, \n"
        "]\n"
    )
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=prompt)
    try:
        result = json.loads(response)
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    return result


def graphPrompt(input: str, generate, metadata={}, #model="mistral-openorca:latest",
                repeat_refine=0,verbatim=False,
               ):
    
    SYS_PROMPT_GRAPHMAKER = (
        "You are a network ontology graph maker who extracts terms and their relations from a given context, using category theory. "
        "You are provided with a context chunk (delimited by ```) Your task is to extract the ontology "
        "of terms mentioned in the given context. These terms should represent the key concepts as per the context, including well-defined and widely used names of materials, systems, methods. \n\n"
        "Format your output as a list of JSON. Each element of the list contains a pair of terms"
        "and the relation between them, like the follwing: \n"
        "[\n"
        "   {\n"
        '       "node_1": "A concept from extracted ontology",\n'
        '       "node_2": "A related concept from extracted ontology",\n'
        '       "edge": "Relationship between the two concepts, node_1 and node_2, succinctly described"\n'
        "   }, {...}\n"
        "]"
        ""
        "Examples:"
        "Context: ```Alice is Marc's mother.```\n"
        "[\n"
        "   {\n"
        '       "node_1": "Alice",\n'
        '       "node_2": "Marc",\n'
        '       "edge": "is mother of"\n'
        "   }, "
        "{...}\n"
        "]"
        "Context: ```Silk is a strong natural fiber used to catch prey in a web. Beta-sheets control its strength.```\n"
        "[\n"
        "   {\n"
        '       "node_1": "silk",\n'
        '       "node_2": "fiber",\n'
        '       "edge": "is"\n'
        "   }," 
        "   {\n"
        '       "node_1": "beta-sheets",\n'
        '       "node_2": "strength",\n'
        '       "edge": "control"\n'
        "   },"        
        "   {\n"
        '       "node_1": "silk",\n'
        '       "node_2": "prey",\n'
        '       "edge": "catches"\n'
        "   },"
        "{...}\n"
        "]\n\n"
        "Analyze the text carefully and produce around 10 triplets, making sure they reflect consistent ontologies.\n"
        )
        
    USER_PROMPT = f"Context: ```{input}``` \n\nOutput: "
    
    print (".", end ="")
    response  =  generate( system_prompt=SYS_PROMPT_GRAPHMAKER, prompt=USER_PROMPT)
    if verbatim:
        print ("---------------------\nFirst result: ", response)
   
    SYS_PROMPT_FORMAT = ('You respond in this format:'
                 '[\n'
                    "   {\n"
                    '       "node_1": "A concept from extracted ontology",\n'
                    '       "node_2": "A related concept from extracted ontology",\n'
                    '       "edge": "Relationship between the two concepts, node_1 and node_2, succinctly described"\n'
                    '   }, {...} ]\n'  )    
    USER_PROMPT = (f'Read this context: ```{input}```.'
                  f'Read this ontology: ```{response}```'
                 f'\n\nImprove the ontology by renaming nodes so that they have consistent labels that are widely used in the field of materials science.'''
                 '')
    response  =  generate(system_prompt=SYS_PROMPT_FORMAT,
                          prompt=USER_PROMPT)
    if verbatim:
        print ("---------------------\nAfter improve: ", response)
    
    USER_PROMPT = f"Context: ```{response}``` \n\n Fix to make sure it is proper format. "
    response  =  generate(system_prompt=SYS_PROMPT_FORMAT, prompt=USER_PROMPT)
    response =   response.replace ('\\', '' )
    if verbatim:
        print ("---------------------\nAfter clean: ", response)
    
    if repeat_refine>0:
        for rep in tqdm(range (repeat_refine)):
            

            
            USER_PROMPT = (f'Insert new triplets into the original ontology. Read this context: ```{input}```.'
                          f'Read this ontology: ```{response}```'
                          f'\n\nInsert additional triplets to the original list, in the same JSON format. Repeat original AND new triplets.\n'
                         '') 
            response  =  generate( system_prompt=SYS_PROMPT_GRAPHMAKER, 
                                  prompt=USER_PROMPT)
            if verbatim:
                print ("---------------------\nAfter adding triplets: ", response)
            USER_PROMPT = f"Context: ```{response}``` \n\n Fix to make sure it is proper format. "
            response  =  generate(system_prompt=SYS_PROMPT_FORMAT, prompt=USER_PROMPT)
            response =   response.replace ('\\', '' )
            USER_PROMPT = (f'Read this context: ```{input}```.'
                          f'Read this ontology: ```{response}```'
                         f'\n\nRevise the ontology by renaming nodes and edges so that they have consistent and concise labels.'''
                        
                         '') 
            response  =  generate(system_prompt=SYS_PROMPT_FORMAT,  
                                  prompt=USER_PROMPT)            
            if verbatim:
                print (f"---------------------\nAfter refine {rep}/{repeat_refine}: ", response)

     
    USER_PROMPT = f"Context: ```{response}``` \n\n Fix to make sure it is proper format. "
    response  =  generate(system_prompt=SYS_PROMPT_FORMAT, prompt=USER_PROMPT)
    response =   response.replace ('\\', '' )
    
    try:
        response=extract(response)
       
    except:
        print(end='')
    
    try:
        result = json.loads(response)
        print (result)
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    return result

def extract (string, start='[', end=']'):
    start_index = string.find(start)
    end_index = string.rfind(end)
     
    return string[start_index :end_index+1]

