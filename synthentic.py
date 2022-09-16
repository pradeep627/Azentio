
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network

import pandas as pd
input_data = '/home/pradeep.k@zucisystems.com/Azentio/S1/synthetic-data/syn_input/DeDupeTraintmp.csv'
mode = 'correlated_attribute_mode'
description_file = f'syn_output/description.json'
synthetic_data = f'syn_output/synthetic_data.csv'


threshold_value = 40
categorical_attributes = {'DOB': True}
candidate_keys = {'': True}
epsilon = 1
degree_of_bayesian_network = 2
num_tuples_to_generate = 1000

describer = DataDescriber(category_threshold=threshold_value)
describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data, 
                                                        epsilon=epsilon, 
                                                        k=degree_of_bayesian_network,
                                                        attribute_to_is_categorical=categorical_attributes,
                                                        attribute_to_is_candidate_key=candidate_keys)
describer.save_dataset_description_to_file(description_file)
display_bayesian_network(describer.bayesian_network)

generator = DataGenerator()
generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)
generator.save_synthetic_data(synthetic_data)


